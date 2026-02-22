#include "ground_seg_gpu.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
static constexpr int NUM_GROUPS = 6; // A-F in unified kernel

// ============================================================================
// Unified Kernel: Groups A-F
//   Groups A-E: warp-level __shfl_xor butterfly reduction
//   Group  F:   2 warps/grid, shared memory inter-warp reduction
// ============================================================================
__global__ void unifiedGridKernel(
    const float* __restrict__ z_data,
    const int* __restrict__ point_idx,
    const int* __restrict__ grid_ids,
    const int* __restrict__ counts,
    const int* __restrict__ cum_blocks,
    const int* __restrict__ data_offsets,
    const int* __restrict__ grid_offsets,
    const int* __restrict__ count_offsets,
    float* __restrict__ grid_mean_z,
    float* __restrict__ grid_var_z,
    uint8_t* __restrict__ grid_is_ground,
    uint8_t* __restrict__ labels,
    float var_threshold,
    float height_threshold)
{
    __shared__ float smem[8]; // used only by Group F (2 per grid × 4 grids/block)

    const unsigned FULL_MASK = 0xFFFFFFFF;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    int group = -1;
    for (int g = 0; g < NUM_GROUPS; g++) {
        if (blockIdx.x < cum_blocks[g]) { group = g; break; }
    }
    if (group < 0) return;

    const int block_in_group = blockIdx.x - (group > 0 ? cum_blocks[group - 1] : 0);
    const int global_warp = block_in_group * WARPS_PER_BLOCK + warp_in_block;

    const int d_off = data_offsets[group];
    const int g_off = grid_offsets[group];
    const int c_off = count_offsets[group];

    // ----------------------------------------------------------------
    // Groups A-E: generalized warp-level butterfly reduction
    //   pad = 2 << group  →  A=2, B=4, C=8, D=16, E=32
    //   grids_per_warp = 32 / pad
    // ----------------------------------------------------------------
    if (group <= 4) {
        const int pad = 2 << group;
        const int grids_per_warp = 32 / pad;
        const int base_grid = global_warp * grids_per_warp;
        const int warp_data_off = d_off + global_warp * 32;
        const int local_grid = lane / pad;
        const int lane_in_grid = lane & (pad - 1);
        const int grid_abs = base_grid + local_grid;

        float val = z_data[warp_data_off + lane];
        int p_idx = point_idx[warp_data_off + lane];
        int cnt = counts[c_off + grid_abs];

        // Butterfly sum: after all steps, every lane in a pad-group has the full sum
        float sum = val;
        for (int step = 1; step < pad; step <<= 1)
            sum += __shfl_xor_sync(FULL_MASK, sum, step);

        float mean = (cnt > 0) ? sum / static_cast<float>(cnt) : 0.0f;

        // Variance: padded lanes (lane_in_grid >= cnt) contribute 0
        float diff_sq = (lane_in_grid < cnt) ? (val - mean) * (val - mean) : 0.0f;
        float var_sum = diff_sq;
        for (int step = 1; step < pad; step <<= 1)
            var_sum += __shfl_xor_sync(FULL_MASK, var_sum, step);
        float variance = (cnt > 0) ? var_sum / static_cast<float>(cnt) : 0.0f;

        bool grid_ground = (variance < var_threshold);
        bool pt_ground = grid_ground && (fabsf(val - mean) < height_threshold);

        if (p_idx >= 0 && lane_in_grid < cnt)
            labels[p_idx] = pt_ground ? 1 : 0;

        if (lane_in_grid == 0 && cnt > 0) {
            int gid = grid_ids[g_off + grid_abs];
            grid_mean_z[gid] = mean;
            grid_var_z[gid] = variance;
            grid_is_ground[gid] = grid_ground ? 1 : 0;
        }
    }
    // ----------------------------------------------------------------
    // Group F: 33-64 elements, 2 warps per grid, 4 grids per block
    // ----------------------------------------------------------------
    else {
        const int grid_in_block = warp_in_block >> 1;
        const int warp_in_pair = warp_in_block & 1;
        const int grid_abs = block_in_group * 4 + grid_in_block;

        int cnt = counts[c_off + grid_abs];
        int element_idx = warp_in_pair * 32 + lane;

        int data_base = d_off + grid_abs * 64 + element_idx;
        float val = z_data[data_base];
        int p_idx = point_idx[data_base];

        // Warp-level partial sum via shfl_down
        float warp_sum = val;
        for (int offset = 16; offset >= 1; offset >>= 1)
            warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);

        if (lane == 0) smem[grid_in_block * 2 + warp_in_pair] = warp_sum;
        __syncthreads();

        float total_sum = smem[grid_in_block * 2] + smem[grid_in_block * 2 + 1];
        float mean = (cnt > 0) ? total_sum / static_cast<float>(cnt) : 0.0f;

        // Variance: guard padded elements
        float diff_sq = (element_idx < cnt) ? (val - mean) * (val - mean) : 0.0f;
        float warp_var = diff_sq;
        for (int offset = 16; offset >= 1; offset >>= 1)
            warp_var += __shfl_down_sync(FULL_MASK, warp_var, offset);

        if (lane == 0) smem[grid_in_block * 2 + warp_in_pair] = warp_var;
        __syncthreads();

        float total_var = smem[grid_in_block * 2] + smem[grid_in_block * 2 + 1];
        float variance = (cnt > 0) ? total_var / static_cast<float>(cnt) : 0.0f;

        bool grid_ground = (variance < var_threshold);
        bool pt_ground = grid_ground && (fabsf(val - mean) < height_threshold);

        if (p_idx >= 0 && element_idx < cnt)
            labels[p_idx] = pt_ground ? 1 : 0;

        if (warp_in_pair == 0 && lane == 0 && cnt > 0) {
            int gid = grid_ids[g_off + grid_abs];
            grid_mean_z[gid] = mean;
            grid_var_z[gid] = variance;
            grid_is_ground[gid] = grid_ground ? 1 : 0;
        }
    }
}

// Shared layout when cnt<=512: z_cache[0..511], reduce_buf[512..639]
#define LG_Z_CACHE_MAX 512
#define LG_REDUCE_OFF  512
#define LG_SHMEM_FLOATS (LG_Z_CACHE_MAX + 128)

// ============================================================================
// Large Grid Kernel (65+ points): 1 block per grid, shared memory reduction
// ============================================================================
__global__ void largeGridKernel(
    const float* __restrict__ z_data,
    const int* __restrict__ point_idx,
    const int* __restrict__ grid_ids,
    const int* __restrict__ counts,
    const int* __restrict__ offsets,
    float* __restrict__ grid_mean_z,
    float* __restrict__ grid_var_z,
    uint8_t* __restrict__ grid_is_ground,
    uint8_t* __restrict__ labels,
    float var_threshold,
    float height_threshold,
    int num_grids)
{
    extern __shared__ float sdata[];

    const int tid = threadIdx.x;
    const int gidx = blockIdx.x;
    if (gidx >= num_grids) return;

    const int cnt = counts[gidx];
    const int off = offsets[gidx];

    const bool use_cache = (cnt <= LG_Z_CACHE_MAX);
    const float* z_read = use_cache ? sdata : (z_data + off);

    if (use_cache) {
        for (int i = tid; i < cnt; i += blockDim.x)
            sdata[i] = z_data[off + i];
        __syncthreads();
    }

    // Strided sum
    float local_sum = 0.0f;
    for (int i = tid; i < cnt; i += blockDim.x)
        local_sum += z_read[i];

    float* rbuf = use_cache ? (sdata + LG_REDUCE_OFF) : sdata;
    rbuf[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) rbuf[tid] += rbuf[tid + s];
        __syncthreads();
    }
    float val = (tid < 32) ? rbuf[tid] : 0.0f;
    if (tid < 32) {
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if (tid == 0) rbuf[0] = val;
    }
    __syncthreads();

    float mean = rbuf[0] / static_cast<float>(cnt);

    // Strided variance
    float local_var = 0.0f;
    for (int i = tid; i < cnt; i += blockDim.x) {
        float d = z_read[i] - mean;
        local_var += d * d;
    }
    rbuf[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) rbuf[tid] += rbuf[tid + s];
        __syncthreads();
    }
    val = (tid < 32) ? rbuf[tid] : 0.0f;
    if (tid < 32) {
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if (tid == 0) rbuf[0] = val;
    }
    __syncthreads();
    float variance = rbuf[0] / static_cast<float>(cnt);

    bool grid_ground = (variance < var_threshold);

    if (tid == 0) {
        int gid = grid_ids[gidx];
        grid_mean_z[gid] = mean;
        grid_var_z[gid] = variance;
        grid_is_ground[gid] = grid_ground ? 1 : 0;
    }

    // Inline per-point classification
    for (int i = tid; i < cnt; i += blockDim.x) {
        int pidx = point_idx[off + i];
        if (pidx >= 0) {
            bool pt_ground = grid_ground && (fabsf(z_read[i] - mean) < height_threshold);
            labels[pidx] = pt_ground ? 1 : 0;
        }
    }
}

// ============================================================================
// Empty Grid Kernel: 3x3 neighbor average variance, reads grid_var_z from GPU
// ============================================================================
__global__ void emptyGridKernel(
    const int* __restrict__ empty_indices,
    const float* __restrict__ grid_var_z,
    uint8_t* __restrict__ grid_is_ground,
    float var_threshold,
    int num_cols, int num_rows,
    int num_empty)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_empty) return;

    int g = empty_indices[tid];
    int row = g / num_cols;
    int col = g % num_cols;

    float sum = 0.0f;
    int valid = 0;

    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            int nr = row + dr;
            int nc = col + dc;
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                sum += grid_var_z[nr * num_cols + nc];
                valid++;
            }
        }
    }

    grid_is_ground[g] = (valid > 0 && sum / static_cast<float>(valid) < var_threshold) ? 1 : 0;
}

// ============================================================================
// CPU-side: unified buffer preparation
// ============================================================================
struct UnifiedBuffers {
    std::vector<float> z_data;
    std::vector<int> point_idx;
    std::vector<int> grid_ids;
    std::vector<int> counts;
    int cum_blocks[NUM_GROUPS];
    int data_offsets[NUM_GROUPS];
    int grid_offsets[NUM_GROUPS];
    int count_offsets[NUM_GROUPS];

    std::vector<float> large_z;
    std::vector<int> large_pidx;
    std::vector<int> large_grid_ids;
    std::vector<int> large_counts;
    std::vector<int> large_data_offsets;
    int num_large;

    std::vector<int> empty_indices;
    int num_empty;

    std::vector<int> one_point_grids;
    std::vector<int> one_point_pidx;
};

static UnifiedBuffers buildUnifiedBuffers(
    const PointCloud& cloud,
    const Params& params)
{
    const int N = cloud.num_points;
    const int NG = params.num_grids;

    UnifiedBuffers buf;

    std::vector<std::vector<int>> grid_point_indices(NG);

    for (int i = 0; i < N; i++) {
        float px = cloud.x[i], py = cloud.y[i];
        if (px < params.x_min || px >= params.x_max ||
            py < params.y_min || py >= params.y_max) {
            continue;
        }
        int col = std::min(static_cast<int>((px - params.x_min) / params.grid_resolution),
                           params.num_cols - 1);
        int row = std::min(static_cast<int>((py - params.y_min) / params.grid_resolution),
                           params.num_rows - 1);
        int g = row * params.num_cols + col;
        grid_point_indices[g].push_back(i);
    }

    const int pad_sizes[] = {2, 4, 8, 16, 32, 64};
    std::vector<std::vector<int>> group_grids(NUM_GROUPS);
    std::vector<int> large_grids;
    std::vector<int> empty_grids;
    std::vector<int> one_pt_grids;

    for (int g = 0; g < NG; g++) {
        int cnt = static_cast<int>(grid_point_indices[g].size());
        if (cnt < params.point_number_threshold) {
            empty_grids.push_back(g);
            if (cnt == 1) one_pt_grids.push_back(g);
        }
        else if (cnt <= 2)  group_grids[0].push_back(g);
        else if (cnt <= 4)  group_grids[1].push_back(g);
        else if (cnt <= 8)  group_grids[2].push_back(g);
        else if (cnt <= 16) group_grids[3].push_back(g);
        else if (cnt <= 32) group_grids[4].push_back(g);
        else if (cnt <= 64) group_grids[5].push_back(g);
        else                large_grids.push_back(g);
    }

    printf("  Groups: empty=%zu, A(2)=%zu, B(3-4)=%zu, C(5-8)=%zu, "
           "D(9-16)=%zu, E(17-32)=%zu, F(33-64)=%zu, G(65+)=%zu\n",
           empty_grids.size(),
           group_grids[0].size(), group_grids[1].size(), group_grids[2].size(),
           group_grids[3].size(), group_grids[4].size(), group_grids[5].size(),
           large_grids.size());

    // Build unified z_data / point_idx / grid_ids / counts for groups A-F
    int running_data = 0;
    int running_grid = 0;
    int running_count = 0;
    int total_blocks = 0;

    for (int gi = 0; gi < NUM_GROUPS; gi++) {
        buf.data_offsets[gi] = running_data;
        buf.grid_offsets[gi] = running_grid;
        buf.count_offsets[gi] = running_count;

        int pad = pad_sizes[gi];
        int num_grids_in_group = static_cast<int>(group_grids[gi].size());

        if (gi < 5) {
            int grids_per_warp = 32 / pad;
            int num_warps = (num_grids_in_group + grids_per_warp - 1) / grids_per_warp;
            int blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            int total_warps = blocks * WARPS_PER_BLOCK;
            int padded_grids = total_warps * grids_per_warp;

            // Data layout: warp-by-warp, each warp = grids_per_warp × pad = 32 floats
            for (int w = 0; w < total_warps; w++) {
                for (int g_in_warp = 0; g_in_warp < grids_per_warp; g_in_warp++) {
                    int grid_local = w * grids_per_warp + g_in_warp;
                    for (int elem = 0; elem < pad; elem++) {
                        if (grid_local < num_grids_in_group) {
                            int g = group_grids[gi][grid_local];
                            auto& pts = grid_point_indices[g];
                            if (elem < static_cast<int>(pts.size())) {
                                buf.z_data.push_back(cloud.z[pts[elem]]);
                                buf.point_idx.push_back(pts[elem]);
                            } else {
                                buf.z_data.push_back(0.0f);
                                buf.point_idx.push_back(-1);
                            }
                        } else {
                            buf.z_data.push_back(0.0f);
                            buf.point_idx.push_back(-1);
                        }
                    }
                }
            }

            for (int i = 0; i < padded_grids; i++) {
                if (i < num_grids_in_group) {
                    int g = group_grids[gi][i];
                    buf.grid_ids.push_back(g);
                    buf.counts.push_back(static_cast<int>(grid_point_indices[g].size()));
                } else {
                    buf.grid_ids.push_back(0);
                    buf.counts.push_back(0);
                }
            }

            running_data += total_warps * 32;
            running_grid += padded_grids;
            running_count += padded_grids;
            total_blocks += blocks;
        } else {
            // Group F: 2 warps/grid, 4 grids/block, pad to 64
            int blocks = (num_grids_in_group + 3) / 4;
            int padded_grids = blocks * 4;

            for (int i = 0; i < padded_grids; i++) {
                for (int elem = 0; elem < 64; elem++) {
                    if (i < num_grids_in_group) {
                        int g = group_grids[gi][i];
                        auto& pts = grid_point_indices[g];
                        if (elem < static_cast<int>(pts.size())) {
                            buf.z_data.push_back(cloud.z[pts[elem]]);
                            buf.point_idx.push_back(pts[elem]);
                        } else {
                            buf.z_data.push_back(0.0f);
                            buf.point_idx.push_back(-1);
                        }
                    } else {
                        buf.z_data.push_back(0.0f);
                        buf.point_idx.push_back(-1);
                    }
                }
            }

            for (int i = 0; i < padded_grids; i++) {
                if (i < num_grids_in_group) {
                    int g = group_grids[gi][i];
                    buf.grid_ids.push_back(g);
                    buf.counts.push_back(static_cast<int>(grid_point_indices[g].size()));
                } else {
                    buf.grid_ids.push_back(0);
                    buf.counts.push_back(0);
                }
            }

            running_data += padded_grids * 64;
            running_grid += padded_grids;
            running_count += padded_grids;
            total_blocks += blocks;
        }

        buf.cum_blocks[gi] = total_blocks;
    }

    // Large grids (65+): pad each grid to 32-aligned for coalesced access
    buf.num_large = static_cast<int>(large_grids.size());
    int large_running = 0;
    for (int g : large_grids) {
        auto& pts = grid_point_indices[g];
        int cnt = static_cast<int>(pts.size());
        int padded = ((cnt + 31) / 32) * 32;  // pad to multiple of 32
        buf.large_grid_ids.push_back(g);
        buf.large_counts.push_back(cnt);  // real count for mean/variance
        buf.large_data_offsets.push_back(large_running);
        for (int pidx : pts) {
            buf.large_z.push_back(cloud.z[pidx]);
            buf.large_pidx.push_back(pidx);
        }
        for (int k = cnt; k < padded; k++) {
            buf.large_z.push_back(0.0f);
            buf.large_pidx.push_back(-1);
        }
        large_running += padded;
    }

    // Empty + 1-point grids
    buf.empty_indices = empty_grids;
    buf.num_empty = static_cast<int>(empty_grids.size());
    buf.one_point_grids = one_pt_grids;
    for (int g : one_pt_grids) {
        buf.one_point_pidx.push_back(grid_point_indices[g][0]);
    }

    return buf;
}

// ============================================================================
// Main GPU pipeline
// ============================================================================
PipelineResult runGPUPipeline(const PointCloud& cloud, const Params& params) {
    PipelineResult result;
    const int N = cloud.num_points;
    const int NG = params.num_grids;

    cudaFree(0); // warmup

    auto t_start = std::chrono::high_resolution_clock::now();

    auto t_start_prep = std::chrono::high_resolution_clock::now();
    // ---- CPU prep: filter + group + build unified buffers ----
    UnifiedBuffers buf = buildUnifiedBuffers(cloud, params);
    auto t_end_prep = std::chrono::high_resolution_clock::now();
    printf("Prep time: %f ms\n", std::chrono::duration<double, std::milli>(t_end_prep - t_start_prep).count());

    // ---- Allocate persistent device arrays ----
    float *d_grid_mean_z, *d_grid_var_z;
    uint8_t *d_grid_is_ground, *d_labels;

    CUDA_CHECK(cudaMalloc(&d_grid_mean_z, NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_var_z, NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_is_ground, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemset(d_grid_mean_z, 0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_var_z, 0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_is_ground, 0, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_labels, 0, N * sizeof(uint8_t)));

    // ---- H2D: unified buffers ----
    float *d_z_data = nullptr;
    int *d_point_idx = nullptr, *d_grid_ids = nullptr, *d_counts = nullptr;
    int *d_cum_blocks = nullptr, *d_data_offsets = nullptr;
    int *d_grid_offsets = nullptr, *d_count_offsets = nullptr;

    if (!buf.z_data.empty()) {
        CUDA_CHECK(cudaMalloc(&d_z_data, buf.z_data.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_point_idx, buf.point_idx.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_grid_ids, buf.grid_ids.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counts, buf.counts.size() * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_z_data, buf.z_data.data(),
                              buf.z_data.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_point_idx, buf.point_idx.data(),
                              buf.point_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid_ids, buf.grid_ids.data(),
                              buf.grid_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counts, buf.counts.data(),
                              buf.counts.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_cum_blocks, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_data_offsets, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_offsets, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count_offsets, NUM_GROUPS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_cum_blocks, buf.cum_blocks,
                          NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_offsets, buf.data_offsets,
                          NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_offsets, buf.grid_offsets,
                          NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_count_offsets, buf.count_offsets,
                          NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));

    // ---- Kernel 1: Unified (groups A-F) ----
    int total_blocks = buf.cum_blocks[NUM_GROUPS - 1];
    if (total_blocks > 0) {
        unifiedGridKernel<<<total_blocks, BLOCK_SIZE>>>(
            d_z_data, d_point_idx, d_grid_ids, d_counts,
            d_cum_blocks, d_data_offsets, d_grid_offsets, d_count_offsets,
            d_grid_mean_z, d_grid_var_z, d_grid_is_ground, d_labels,
            params.min_variance_threshold, params.height_threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Kernel 2: Large grids (65+) ----
    if (buf.num_large > 0) {
        float *d_lg_z = nullptr;
        int *d_lg_pidx = nullptr, *d_lg_gids = nullptr;
        int *d_lg_counts = nullptr, *d_lg_offsets = nullptr;

        CUDA_CHECK(cudaMalloc(&d_lg_z, buf.large_z.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lg_pidx, buf.large_pidx.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lg_gids, buf.num_large * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lg_counts, buf.num_large * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lg_offsets, buf.num_large * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_lg_z, buf.large_z.data(),
                              buf.large_z.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lg_pidx, buf.large_pidx.data(),
                              buf.large_pidx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lg_gids, buf.large_grid_ids.data(),
                              buf.num_large * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lg_counts, buf.large_counts.data(),
                              buf.num_large * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lg_offsets, buf.large_data_offsets.data(),
                              buf.num_large * sizeof(int), cudaMemcpyHostToDevice));

        const int lg_block = 128;
        largeGridKernel<<<buf.num_large, lg_block, LG_SHMEM_FLOATS * sizeof(float)>>>(
            d_lg_z, d_lg_pidx, d_lg_gids, d_lg_counts, d_lg_offsets,
            d_grid_mean_z, d_grid_var_z, d_grid_is_ground, d_labels,
            params.min_variance_threshold, params.height_threshold, buf.num_large);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_lg_z); cudaFree(d_lg_pidx); cudaFree(d_lg_gids);
        cudaFree(d_lg_counts); cudaFree(d_lg_offsets);
    }

    // ---- Kernel 3: Empty grids (reads grid_var_z already on GPU) ----
    if (buf.num_empty > 0) {
        int *d_empty_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_empty_idx, buf.num_empty * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_empty_idx, buf.empty_indices.data(),
                              buf.num_empty * sizeof(int), cudaMemcpyHostToDevice));

        int eblocks = (buf.num_empty + 255) / 256;
        emptyGridKernel<<<eblocks, 256>>>(
            d_empty_idx, d_grid_var_z, d_grid_is_ground,
            params.min_variance_threshold,
            params.num_cols, params.num_rows, buf.num_empty);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_empty_idx);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- D2H: results ----
    result.labels.resize(N, 0);
    result.grid_mean_z.resize(NG);
    result.grid_var_z.resize(NG);
    result.grid_is_ground.resize(NG);

    CUDA_CHECK(cudaMemcpy(result.labels.data(), d_labels,
                          N * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_mean_z.data(), d_grid_mean_z,
                          NG * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_var_z.data(), d_grid_var_z,
                          NG * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_is_ground.data(), d_grid_is_ground,
                          NG * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // ---- CPU post-pass: classify 1-point grid points ----
    for (size_t i = 0; i < buf.one_point_grids.size(); i++) {
        int g = buf.one_point_grids[i];
        int pidx = buf.one_point_pidx[i];
        result.grid_mean_z[g] = cloud.z[pidx];
        result.labels[pidx] = result.grid_is_ground[g] ? 1 : 0;
    }

    // ---- Cleanup ----
    if (d_z_data) cudaFree(d_z_data);
    if (d_point_idx) cudaFree(d_point_idx);
    if (d_grid_ids) cudaFree(d_grid_ids);
    if (d_counts) cudaFree(d_counts);
    cudaFree(d_cum_blocks); cudaFree(d_data_offsets);
    cudaFree(d_grid_offsets); cudaFree(d_count_offsets);
    cudaFree(d_grid_mean_z); cudaFree(d_grid_var_z);
    cudaFree(d_grid_is_ground); cudaFree(d_labels);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    result.grid_min_z.resize(NG, 0.0f);
    result.grid_max_z.resize(NG, 0.0f);
    result.grid_count.resize(NG, 0);

    return result;
}
