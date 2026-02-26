#include "ground_seg_gpu.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

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
static constexpr int NUM_GROUPS = 32;
static constexpr int LARGE_GROUP = 31;

// ============================================================================
// Phase 1: Grid Assignment & Histogram with Row Assignment
// ============================================================================

__global__ void assignAndHistogramKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int* __restrict__ grid_idx,
    int* grid_count,
    int* __restrict__ point_row,
    float x_min, float x_max, float y_min, float y_max,
    float resolution, int num_cols, int num_rows, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float px = x[i], py = y[i];
    if (px < x_min || px >= x_max || py < y_min || py >= y_max) {
        grid_idx[i] = -1;
        point_row[i] = -1;
        return;
    }
    int col = min(static_cast<int>((px - x_min) / resolution), num_cols - 1);
    int row = min(static_cast<int>((py - y_min) / resolution), num_rows - 1);
    int g = row * num_cols + col;
    grid_idx[i] = g;
    point_row[i] = atomicAdd(&grid_count[g], 1);
}

// ============================================================================
// Phase 2: Group Assignment & Group Histogram
// ============================================================================

__global__ void groupAssignAndHistogramKernel(
    const int* __restrict__ grid_count,
    int* __restrict__ grid_group,
    int* group_grid_count,
    int* __restrict__ grid_row_in_group,
    int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    int cnt = grid_count[g];
    int grp;
    if (cnt < 2)
        grp = -1;
    else if (cnt <= 32)
        grp = cnt - 2;
    else
        grp = LARGE_GROUP;
    grid_group[g] = grp;
    if (grp >= 0)
        grid_row_in_group[g] = atomicAdd(&group_grid_count[grp], 1);
    else
        grid_row_in_group[g] = -1;
}

// ============================================================================
// Phase 3: z_data Construction
// ============================================================================

__global__ void fillLargeCountsKernel(
    const int* __restrict__ grid_count,
    const int* __restrict__ grid_group,
    const int* __restrict__ grid_row_in_group,
    int* __restrict__ large_counts,
    int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    if (grid_group[g] == LARGE_GROUP)
        large_counts[grid_row_in_group[g]] = grid_count[g];
}

__global__ void scatterToZdataKernel(
    const float* __restrict__ z,
    const int* __restrict__ grid_idx,
    const int* __restrict__ grid_group,
    const int* __restrict__ grid_row_in_group,
    const int* __restrict__ point_row,
    const int* __restrict__ group_data_offset,
    const int* __restrict__ padded_sizes,
    const int* __restrict__ large_offsets,
    float* __restrict__ z_data,
    int* __restrict__ grid_id_data,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int g = grid_idx[i];
    if (g < 0) return;
    int grp = grid_group[g];
    if (grp < 0) return;

    int pos;
    if (grp < LARGE_GROUP)
        pos = group_data_offset[grp] + grid_row_in_group[g] * padded_sizes[grp] + point_row[i];
    else
        pos = group_data_offset[LARGE_GROUP] + large_offsets[grid_row_in_group[g]] + point_row[i];

    z_data[pos] = z[i];
    grid_id_data[pos] = g;
}

// ============================================================================
// Phase 4: Unified Reduction Kernel
//   pad > 0 (2..32) : warp-level butterfly __shfl_xor reduction
//   pad == 0         : large grid (>32 elems), shared-memory block reduction
//   Outputs: mean, variance, min, max → directly to final grid-indexed arrays
// ============================================================================

__global__ void reductionKernel(
    const float* __restrict__ z_data,
    const int* __restrict__ grid_id_data,
    const int* __restrict__ block_z_offset,
    const int* __restrict__ block_group_no,
    float* __restrict__ grid_mean_z,
    float* __restrict__ grid_var_z,
    float* __restrict__ grid_min_z,
    float* __restrict__ grid_max_z)
{
    extern __shared__ float sdata[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int z_off = block_z_offset[bid];
    const int z_end = block_z_offset[bid + 1];
    const int grp   = block_group_no[bid];

    int pad, cnt;
    if (grp < LARGE_GROUP) {
        cnt = grp + 2;
        pad = 1 << (32 - __clz(cnt - 1));
        if (cnt == 2) pad = 2;
    } else {
        pad = 0;
        cnt = z_end - z_off;
    }

    // ----------------------------------------------------------------
    // Small groups (pad 2..32): butterfly warp-level reduction
    // ----------------------------------------------------------------
    if (pad > 0) {
        const unsigned FULL_MASK = 0xFFFFFFFF;
        const int total_data = z_end - z_off;
        const int num_grids  = total_data / pad;
        const int my_grid    = tid / pad;
        const int lane_in_grid = tid & (pad - 1);
        const bool valid     = (my_grid < num_grids);
        const bool real_lane = valid && (lane_in_grid < cnt);

        float val = valid ? z_data[z_off + tid] : 0.0f;

        float sum   = val;
        float min_v = real_lane ? val : FLT_MAX;
        float max_v = real_lane ? val : -FLT_MAX;
        for (int step = 1; step < pad; step <<= 1) {
            sum   += __shfl_xor_sync(FULL_MASK, sum, step);
            min_v  = fminf(min_v, __shfl_xor_sync(FULL_MASK, min_v, step));
            max_v  = fmaxf(max_v, __shfl_xor_sync(FULL_MASK, max_v, step));
        }

        float mean = (valid && cnt > 0) ? sum / static_cast<float>(cnt) : 0.0f;

        float diff_sq = real_lane ? (val - mean) * (val - mean) : 0.0f;
        float var_sum = diff_sq;
        for (int step = 1; step < pad; step <<= 1)
            var_sum += __shfl_xor_sync(FULL_MASK, var_sum, step);
        float variance = (valid && cnt > 0)
                         ? var_sum / static_cast<float>(cnt) : 0.0f;

        if (valid && lane_in_grid == 0) {
            int real_gid = grid_id_data[z_off + my_grid * pad];
            grid_mean_z[real_gid] = mean;
            grid_var_z[real_gid]  = variance;
            grid_min_z[real_gid]  = min_v;
            grid_max_z[real_gid]  = max_v;
        }
        return;
    }

    // ----------------------------------------------------------------
    // Large group (>32 elements): shared-memory block-level reduction
    //   sdata layout: [0..BS) = general purpose
    //                 [BS..2*BS) = min scratch
    //                 [2*BS..3*BS) = max scratch
    // ----------------------------------------------------------------
    float* s_min = sdata + blockDim.x;
    float* s_max = sdata + 2 * blockDim.x;
    const int n = z_end - z_off;

    // --- sum reduction → mean ---
    float local_sum = 0.0f;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x) {
        float zv = z_data[z_off + i];
        local_sum += zv;
        local_min  = fminf(local_min, zv);
        local_max  = fmaxf(local_max, zv);
    }

    sdata[tid] = local_sum;
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            s_min[tid]  = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid]  = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float v = (tid < 32) ? sdata[tid] : 0.0f;
    float mn = (tid < 32) ? s_min[tid] : FLT_MAX;
    float mx = (tid < 32) ? s_max[tid] : -FLT_MAX;
    if (tid < 32) {
        for (int step = 16; step >= 1; step >>= 1) {
            v  += __shfl_down_sync(0xFFFFFFFF, v, step);
            mn  = fminf(mn, __shfl_down_sync(0xFFFFFFFF, mn, step));
            mx  = fmaxf(mx, __shfl_down_sync(0xFFFFFFFF, mx, step));
        }
        if (tid == 0) {
            sdata[0] = v;
            s_min[0] = mn;
            s_max[0] = mx;
        }
    }
    __syncthreads();
    float mean = sdata[0] / static_cast<float>(n);
    float final_min = s_min[0];
    float final_max = s_max[0];

    // --- variance reduction ---
    float local_var = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = z_data[z_off + i] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    v = (tid < 32) ? sdata[tid] : 0.0f;
    if (tid < 32) {
        for (int step = 16; step >= 1; step >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, step);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    float variance = sdata[0] / static_cast<float>(n);

    if (tid == 0) {
        int real_gid = grid_id_data[z_off];
        grid_mean_z[real_gid] = mean;
        grid_var_z[real_gid]  = variance;
        grid_min_z[real_gid]  = final_min;
        grid_max_z[real_gid]  = final_max;
    }
}

// ============================================================================
// Phase 5: Ground Classification (2D, 3x3 patch with shared memory halo)
// ============================================================================

static constexpr int TILE = 16;

__global__ void groundClassificationKernel(
    const float* __restrict__ grid_var_z,
    const int*   __restrict__ grid_count,
    uint8_t*     __restrict__ is_grid_ground,
    float var_threshold,
    int   point_threshold,
    int   num_cols,
    int   num_rows)
{
    __shared__ float smem_var[TILE + 2][TILE + 2];
    __shared__ int   smem_cnt[TILE + 2][TILE + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE + tx;
    const int gy = blockIdx.y * TILE + ty;
    const int sx = tx + 1;
    const int sy = ty + 1;

    auto load = [&](int col, int row, int si, int sj) {
        if (col >= 0 && col < num_cols && row >= 0 && row < num_rows) {
            int idx = row * num_cols + col;
            smem_var[sj][si] = grid_var_z[idx];
            smem_cnt[sj][si] = grid_count[idx];
        } else {
            smem_var[sj][si] = 0.0f;
            smem_cnt[sj][si] = 0;
        }
    };

    load(gx, gy, sx, sy);

    if (tx == 0)           load(gx - 1, gy, 0, sy);
    if (tx == TILE - 1)    load(gx + 1, gy, TILE + 1, sy);
    if (ty == 0)           load(gx, gy - 1, sx, 0);
    if (ty == TILE - 1)    load(gx, gy + 1, sx, TILE + 1);
    if (tx == 0 && ty == 0)                     load(gx - 1, gy - 1, 0, 0);
    if (tx == TILE - 1 && ty == 0)              load(gx + 1, gy - 1, TILE + 1, 0);
    if (tx == 0 && ty == TILE - 1)              load(gx - 1, gy + 1, 0, TILE + 1);
    if (tx == TILE - 1 && ty == TILE - 1)       load(gx + 1, gy + 1, TILE + 1, TILE + 1);

    __syncthreads();

    if (gx >= num_cols || gy >= num_rows) return;

    int cnt = smem_cnt[sy][sx];

    if (cnt >= point_threshold) {
        is_grid_ground[gy * num_cols + gx] =
            (smem_var[sy][sx] < var_threshold) ? 1 : 0;
    } else {
        float var_sum = 0.0f;
        int   valid   = 0;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int nx = gx + dc, ny = gy + dr;
                if (nx >= 0 && nx < num_cols && ny >= 0 && ny < num_rows) {
                    var_sum += smem_var[sy + dr][sx + dc];
                    valid++;
                }
            }
        }
        is_grid_ground[gy * num_cols + gx] =
            (valid > 0 && var_sum / static_cast<float>(valid) < var_threshold)
                ? 1 : 0;
    }
}

// ============================================================================
// Phase 6: Point Classification
// ============================================================================

__global__ void pointClassificationKernel(
    const float*   __restrict__ z,
    const int*     __restrict__ grid_idx,
    const int*     __restrict__ grid_count,
    const uint8_t* __restrict__ is_grid_ground,
    const float*   __restrict__ grid_mean_z,
    uint8_t*       __restrict__ labels,
    float height_threshold,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int g = grid_idx[i];
    if (g < 0)                { labels[i] = 0; return; }
    if (!is_grid_ground[g])   { labels[i] = 0; return; }
    if (grid_count[g] < 2)   { labels[i] = 1; return; }
    labels[i] = (fabsf(z[i] - grid_mean_z[g]) < height_threshold) ? 1 : 0;
}

// ============================================================================
// Host helpers
// ============================================================================

static inline int nextPow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ============================================================================
// Main GPU Pipeline
// ============================================================================

PipelineResult runGPUPipeline(const PointCloud& cloud, const Params& params) {
    PipelineResult result;
    const int N  = cloud.num_points;
    const int NG = params.num_grids;

    cudaFree(0);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_prev  = t_start;

    auto lap = [&](const char* tag) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto now = std::chrono::high_resolution_clock::now();
        printf("  [TIME] %-40s %7.3f ms\n", tag,
               std::chrono::duration<double, std::milli>(now - t_prev).count());
        t_prev = now;
    };

    // ---- Upload point cloud ------------------------------------------------
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, cloud.x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, cloud.y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, cloud.z.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Allocate per-point & per-grid buffers -----------------------------
    int *d_grid_idx, *d_point_row, *d_grid_count;
    int *d_grid_group, *d_grid_row_in_group, *d_group_grid_count;
    CUDA_CHECK(cudaMalloc(&d_grid_idx,          N  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_point_row,         N  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_count,        NG * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_group,        NG * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_row_in_group, NG * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_grid_count,  NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_grid_count,       0, NG * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_group_grid_count, 0, NUM_GROUPS * sizeof(int)));
    lap("H2D upload + initial alloc");

    const int blocks_N  = (N  + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_NG = (NG + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---- Phase 1: assign grids + histogram (fused) ---------------------------
    assignAndHistogramKernel<<<blocks_N, BLOCK_SIZE>>>(
        d_x, d_y, d_grid_idx, d_grid_count, d_point_row,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.grid_resolution, params.num_cols, params.num_rows, N);
    CUDA_CHECK(cudaGetLastError());
    lap("assignAndHistogramKernel");

    // ---- Phase 2: group assign + histogram (fused) -------------------------
    groupAssignAndHistogramKernel<<<blocks_NG, BLOCK_SIZE>>>(
        d_grid_count, d_grid_group, d_group_grid_count,
        d_grid_row_in_group, NG);
    CUDA_CHECK(cudaGetLastError());
    lap("groupAssignAndHistogramKernel");

    // ---- Tiny D2H: 32 ints (128 bytes) ------------------------------------
    int h_ggc[NUM_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_ggc, d_group_grid_count,
                          NUM_GROUPS * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- CPU: padded sizes & offsets ---------------------------------------
    int h_pad[NUM_GROUPS];
    for (int i = 0; i < LARGE_GROUP; i++)
        h_pad[i] = nextPow2(i + 2);
    h_pad[LARGE_GROUP] = 0;

    int h_gdo[NUM_GROUPS];
    int run = 0;
    for (int i = 0; i < LARGE_GROUP; i++) {
        h_gdo[i] = run;
        run += h_ggc[i] * h_pad[i];
    }
    h_gdo[LARGE_GROUP] = run;

    lap("CPU: padded sizes & offsets");

    // ---- Phase 3: large-group offsets (>32) --------------------------------
    int num_large = h_ggc[LARGE_GROUP];
    int total_large_data = 0;
    int *d_large_counts = nullptr, *d_large_offsets = nullptr;
    std::vector<int> h_large_off, h_large_cnt;

    if (num_large > 0) {
        CUDA_CHECK(cudaMalloc(&d_large_counts,  num_large * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_large_offsets, num_large * sizeof(int)));

        fillLargeCountsKernel<<<blocks_NG, BLOCK_SIZE>>>(
            d_grid_count, d_grid_group, d_grid_row_in_group,
            d_large_counts, NG);
        CUDA_CHECK(cudaGetLastError());

        thrust::device_ptr<int> lc(d_large_counts);
        thrust::device_ptr<int> lo(d_large_offsets);
        thrust::exclusive_scan(lc, lc + num_large, lo);

        h_large_off.resize(num_large);
        h_large_cnt.resize(num_large);
        CUDA_CHECK(cudaMemcpy(h_large_off.data(), d_large_offsets,
                              num_large * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_large_cnt.data(), d_large_counts,
                              num_large * sizeof(int), cudaMemcpyDeviceToHost));
        total_large_data = h_large_off[num_large - 1] + h_large_cnt[num_large - 1];
    }
    lap("large-group offsets (scan+D2H)");

    const int z_data_size = h_gdo[LARGE_GROUP] + total_large_data;

    // if (num_large > 0)
    //     printf("  Large group: %d grids, %d points (no padding)\n",
    //            num_large, total_large_data);
    // printf("  z_data total: %d floats (%.1f KB)\n",
    //        z_data_size, z_data_size * sizeof(float) / 1024.0);

    // ---- CPU: build block offset table -------------------------------------
    std::vector<int> bzo, bgn;

    for (int i = 0; i < LARGE_GROUP; i++) {
        if (h_ggc[i] == 0) continue;
        int pad = h_pad[i];
        int gpb = BLOCK_SIZE / pad;
        int ng  = h_ggc[i];
        int nb  = (ng + gpb - 1) / gpb;
        for (int b = 0; b < nb; b++) {
            bzo.push_back(h_gdo[i] + b * gpb * pad);
            bgn.push_back(i);
        }
    }
    for (int b = 0; b < num_large; b++) {
        bzo.push_back(h_gdo[LARGE_GROUP] + h_large_off[b]);
        bgn.push_back(LARGE_GROUP);
    }
    bzo.push_back(z_data_size);

    const int total_blocks = static_cast<int>(bgn.size());

    // ---- Allocate z_data, grid_id_data, final output buffers ----------------
    float *d_z_data = nullptr;
    int   *d_grid_id_data = nullptr;

    if (z_data_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_z_data,       z_data_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_z_data, 0,     z_data_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grid_id_data, z_data_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_grid_id_data, 0, z_data_size * sizeof(int)));
    }

    float   *d_grid_mean_z, *d_grid_var_z, *d_grid_min_z, *d_grid_max_z;
    uint8_t *d_grid_is_ground, *d_labels;
    CUDA_CHECK(cudaMalloc(&d_grid_mean_z,    NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_var_z,     NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_min_z,     NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_max_z,     NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_is_ground, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_labels,         N  * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_grid_mean_z,    0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_var_z,     0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_min_z,     0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_max_z,     0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_is_ground, 0, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_labels,         0, N  * sizeof(uint8_t)));

    // ---- Upload metadata arrays --------------------------------------------
    int *d_gdo, *d_ps;
    CUDA_CHECK(cudaMalloc(&d_gdo, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ps,  NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_gdo, h_gdo, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ps,  h_pad, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));

    int *d_bzo = nullptr, *d_bgn = nullptr;
    if (total_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&d_bzo, (total_blocks + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bgn, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bzo, bzo.data(), (total_blocks + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bgn, bgn.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice));
    }
    lap("CPU metadata + alloc bufs + H2D tables");

    // ---- Phase 3: scatter z values + grid IDs into organized buffers -------
    if (z_data_size > 0) {
        scatterToZdataKernel<<<blocks_N, BLOCK_SIZE>>>(
            d_z, d_grid_idx, d_grid_group, d_grid_row_in_group, d_point_row,
            d_gdo, d_ps, d_large_offsets, d_z_data, d_grid_id_data, N);
        CUDA_CHECK(cudaGetLastError());
    }
    lap("scatterToZdataKernel");

    // ---- Phase 4: reduction (mean, var, min, max → final grid arrays) ------
    if (total_blocks > 0) {
        reductionKernel<<<total_blocks, BLOCK_SIZE,
                          BLOCK_SIZE * 3 * sizeof(float)>>>(
            d_z_data, d_grid_id_data, d_bzo, d_bgn,
            d_grid_mean_z, d_grid_var_z, d_grid_min_z, d_grid_max_z);
        CUDA_CHECK(cudaGetLastError());
    }
    lap("reductionKernel");

    // ---- Phase 5: ground classification (2D, 3x3 patch) --------------------
    dim3 tile_block(TILE, TILE);
    dim3 tile_grid((params.num_cols + TILE - 1) / TILE,
                   (params.num_rows + TILE - 1) / TILE);
    groundClassificationKernel<<<tile_grid, tile_block>>>(
        d_grid_var_z, d_grid_count, d_grid_is_ground,
        params.min_variance_threshold, params.point_number_threshold,
        params.num_cols, params.num_rows);
    CUDA_CHECK(cudaGetLastError());
    lap("groundClassificationKernel");

    // ---- Phase 6: point classification -------------------------------------
    pointClassificationKernel<<<blocks_N, BLOCK_SIZE>>>(
        d_z, d_grid_idx, d_grid_count, d_grid_is_ground,
        d_grid_mean_z, d_labels, params.height_threshold, N);
    CUDA_CHECK(cudaGetLastError());
    lap("pointClassificationKernel");

    // ---- D2H: final results ------------------------------------------------
    result.labels.resize(N);
    result.grid_mean_z.resize(NG);
    result.grid_var_z.resize(NG);
    result.grid_min_z.resize(NG);
    result.grid_max_z.resize(NG);
    result.grid_is_ground.resize(NG);
    result.grid_count.resize(NG);

    CUDA_CHECK(cudaMemcpy(result.labels.data(),         d_labels,
                          N  * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_mean_z.data(),    d_grid_mean_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_var_z.data(),     d_grid_var_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_min_z.data(),     d_grid_min_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_max_z.data(),     d_grid_max_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_is_ground.data(), d_grid_is_ground,
                          NG * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_count.data(),     d_grid_count,
                          NG * sizeof(int),     cudaMemcpyDeviceToHost));
    lap("D2H final results");

    // ---- Cleanup -----------------------------------------------------------
    if (d_large_counts)  cudaFree(d_large_counts);
    if (d_large_offsets) cudaFree(d_large_offsets);
    if (d_z_data)        cudaFree(d_z_data);
    if (d_grid_id_data)  cudaFree(d_grid_id_data);
    cudaFree(d_gdo);  cudaFree(d_ps);
    if (d_bzo) cudaFree(d_bzo);
    if (d_bgn) cudaFree(d_bgn);
    cudaFree(d_grid_mean_z);    cudaFree(d_grid_var_z);
    cudaFree(d_grid_min_z);     cudaFree(d_grid_max_z);
    cudaFree(d_grid_is_ground); cudaFree(d_labels);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}
