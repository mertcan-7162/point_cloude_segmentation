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

__global__ void assignGridKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int* __restrict__ grid_idx,
    float x_min, float x_max, float y_min, float y_max,
    float resolution, int num_cols, int num_rows, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float px = x[i], py = y[i];
    if (px < x_min || px >= x_max || py < y_min || py >= y_max) {
        grid_idx[i] = -1;
        return;
    }
    int col = min(static_cast<int>((px - x_min) / resolution), num_cols - 1);
    int row = min(static_cast<int>((py - y_min) / resolution), num_rows - 1);
    grid_idx[i] = row * num_cols + col;
}

__global__ void histogramWithRowKernel(
    const int* __restrict__ grid_idx,
    int* grid_count,
    int* __restrict__ point_row,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int g = grid_idx[i];
    if (g >= 0)
        point_row[i] = atomicAdd(&grid_count[g], 1);
    else
        point_row[i] = -1;
}

// ============================================================================
// Phase 2: Group Assignment & Group Histogram
// ============================================================================

__global__ void groupAssignKernel(
    const int* __restrict__ grid_count,
    int* __restrict__ grid_group,
    int point_threshold,
    int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    int cnt = grid_count[g];
    if (cnt < point_threshold || cnt < 2)
        grid_group[g] = -1;
    else if (cnt <= 32)
        grid_group[g] = cnt - 2;
    else
        grid_group[g] = LARGE_GROUP;
}

__global__ void groupHistogramWithRowKernel(
    const int* __restrict__ grid_group,
    int* group_grid_count,
    int* __restrict__ grid_row_in_group,
    int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    int grp = grid_group[g];
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
}

// ============================================================================
// Phase 4: Unified Reduction Kernel
//   pad > 0 (2..32) : warp-level butterfly __shfl_xor reduction
//   pad == 0         : large grid (>32 elems), shared-memory block reduction
// ============================================================================

__global__ void reductionKernel(
    const float* __restrict__ z_data,
    const int* __restrict__ block_z_offset,
    const int* __restrict__ block_padded_size,
    const int* __restrict__ block_actual_count,
    const int* __restrict__ block_grid_output_offset,
    float* __restrict__ grid_mean_z_buf,
    float* __restrict__ grid_var_z_buf,
    uint8_t* __restrict__ grid_is_ground_buf,
    uint8_t* __restrict__ labels_buf,
    float var_threshold,
    float height_threshold)
{
    extern __shared__ float sdata[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int z_off = block_z_offset[bid];
    const int z_end = block_z_offset[bid + 1];
    const int pad   = block_padded_size[bid];
    const int cnt   = block_actual_count[bid];
    const int g_off = block_grid_output_offset[bid];

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

        float val = valid ? z_data[z_off + tid] : 0.0f;

        float sum = val;
        for (int step = 1; step < pad; step <<= 1)
            sum += __shfl_xor_sync(FULL_MASK, sum, step);

        float mean = (valid && cnt > 0) ? sum / static_cast<float>(cnt) : 0.0f;

        float diff_sq = (valid && lane_in_grid < cnt)
                        ? (val - mean) * (val - mean) : 0.0f;
        float var_sum = diff_sq;
        for (int step = 1; step < pad; step <<= 1)
            var_sum += __shfl_xor_sync(FULL_MASK, var_sum, step);
        float variance = (valid && cnt > 0)
                         ? var_sum / static_cast<float>(cnt) : 0.0f;

        if (valid) {
            bool ground = (variance < var_threshold);
            if (lane_in_grid < cnt) {
                bool pt_ground = ground &&
                                 (fabsf(val - mean) < height_threshold);
                labels_buf[z_off + tid] = pt_ground ? 1 : 0;
            }
            if (lane_in_grid == 0) {
                int out_idx = g_off + my_grid;
                grid_mean_z_buf[out_idx] = mean;
                grid_var_z_buf[out_idx]  = variance;
                grid_is_ground_buf[out_idx] = ground ? 1 : 0;
            }
        }
        return;
    }

    // ----------------------------------------------------------------
    // Large group (>32 elements): shared-memory block-level reduction
    // ----------------------------------------------------------------
    const int n = z_end - z_off;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        local_sum += z_data[z_off + i];

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float v = (tid < 32) ? sdata[tid] : 0.0f;
    if (tid < 32) {
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    float mean = sdata[0] / static_cast<float>(n);

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
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    float variance = sdata[0] / static_cast<float>(n);
    bool ground = (variance < var_threshold);

    if (tid == 0) {
        grid_mean_z_buf[g_off]      = mean;
        grid_var_z_buf[g_off]       = variance;
        grid_is_ground_buf[g_off]   = ground ? 1 : 0;
    }

    for (int i = tid; i < n; i += blockDim.x) {
        bool pt = ground && (fabsf(z_data[z_off + i] - mean) < height_threshold);
        labels_buf[z_off + i] = pt ? 1 : 0;
    }
}

// ============================================================================
// Phase 5: Remap & Empty Grid Handling
// ============================================================================

__global__ void remapGridStatsKernel(
    const int* __restrict__ grid_group,
    const int* __restrict__ grid_row_in_group,
    const int* __restrict__ group_grid_offset,
    const float* __restrict__ grid_mean_z_buf,
    const float* __restrict__ grid_var_z_buf,
    const uint8_t* __restrict__ grid_is_ground_buf,
    float* __restrict__ grid_mean_z,
    float* __restrict__ grid_var_z,
    uint8_t* __restrict__ grid_is_ground,
    int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    int grp = grid_group[g];
    if (grp < 0) return;
    int buf_pos = group_grid_offset[grp] + grid_row_in_group[g];
    grid_mean_z[g]    = grid_mean_z_buf[buf_pos];
    grid_var_z[g]     = grid_var_z_buf[buf_pos];
    grid_is_ground[g] = grid_is_ground_buf[buf_pos];
}

__global__ void emptyGridKernel(
    const int* __restrict__ grid_group,
    const float* __restrict__ grid_var_z,
    uint8_t* __restrict__ grid_is_ground,
    float var_threshold,
    int num_cols, int num_rows, int NG)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= NG) return;
    if (grid_group[g] >= 0) return;

    int row = g / num_cols;
    int col = g % num_cols;

    float sum = 0.0f;
    int valid = 0;
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            int nr = row + dr, nc = col + dc;
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                sum += grid_var_z[nr * num_cols + nc];
                valid++;
            }
        }
    }
    grid_is_ground[g] =
        (valid > 0 && sum / static_cast<float>(valid) < var_threshold) ? 1 : 0;
}

__global__ void remapLabelsKernel(
    const float* __restrict__ z,
    const int* __restrict__ grid_idx,
    const int* __restrict__ grid_group,
    const int* __restrict__ grid_count,
    const int* __restrict__ grid_row_in_group,
    const int* __restrict__ point_row,
    const int* __restrict__ group_data_offset,
    const int* __restrict__ padded_sizes,
    const int* __restrict__ large_offsets,
    const uint8_t* __restrict__ grid_is_ground,
    const uint8_t* __restrict__ labels_buf,
    uint8_t* __restrict__ labels,
    float* grid_mean_z,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int g = grid_idx[i];
    if (g < 0) { labels[i] = 0; return; }

    int grp = grid_group[g];
    if (grp < 0) {
        labels[i] = grid_is_ground[g] ? 1 : 0;
        if (grid_count[g] == 1)
            grid_mean_z[g] = z[i];
        return;
    }

    int pos;
    if (grp < LARGE_GROUP)
        pos = group_data_offset[grp]
            + grid_row_in_group[g] * padded_sizes[grp]
            + point_row[i];
    else
        pos = group_data_offset[LARGE_GROUP]
            + large_offsets[grid_row_in_group[g]]
            + point_row[i];

    labels[i] = labels_buf[pos];
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

    // ---- Phase 1: assign grids ---------------------------------------------
    assignGridKernel<<<blocks_N, BLOCK_SIZE>>>(
        d_x, d_y, d_grid_idx,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.grid_resolution, params.num_cols, params.num_rows, N);
    CUDA_CHECK(cudaGetLastError());
    lap("assignGridKernel");

    cudaFree(d_x);
    cudaFree(d_y);

    // ---- Phase 1: histogram + row assignment -------------------------------
    histogramWithRowKernel<<<blocks_N, BLOCK_SIZE>>>(
        d_grid_idx, d_grid_count, d_point_row, N);
    CUDA_CHECK(cudaGetLastError());
    lap("histogramWithRowKernel");

    // ---- Phase 2: group assignment -----------------------------------------
    groupAssignKernel<<<blocks_NG, BLOCK_SIZE>>>(
        d_grid_count, d_grid_group, params.point_number_threshold, NG);
    CUDA_CHECK(cudaGetLastError());

    // ---- Phase 2: group histogram ------------------------------------------
    groupHistogramWithRowKernel<<<blocks_NG, BLOCK_SIZE>>>(
        d_grid_group, d_group_grid_count, d_grid_row_in_group, NG);
    CUDA_CHECK(cudaGetLastError());
    lap("groupAssign + groupHistogram");

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

    int h_ggo[NUM_GROUPS];
    int grun = 0;
    for (int i = 0; i < NUM_GROUPS; i++) {
        h_ggo[i] = grun;
        grun += h_ggc[i];
    }
    int total_output_grids = grun;

    printf("  Groups: ");
    for (int i = 0; i < NUM_GROUPS; i++)
        if (h_ggc[i] > 0) printf("[cnt=%d]=%d ", (i < LARGE_GROUP ? i+2 : 33), h_ggc[i]);
    printf("\n");

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
    lap("large-group offsets (fill+scan+D2H)");

    const int z_data_size = h_gdo[LARGE_GROUP] + total_large_data;

    // ---- CPU: build block offset table -------------------------------------
    std::vector<int> bzo, bps, bac, bgo;

    for (int i = 0; i < LARGE_GROUP; i++) {
        if (h_ggc[i] == 0) continue;
        int pad = h_pad[i];
        int act = i + 2;
        int gpb = BLOCK_SIZE / pad;
        int ng  = h_ggc[i];
        int nb  = (ng + gpb - 1) / gpb;
        for (int b = 0; b < nb; b++) {
            int sg = b * gpb;
            bzo.push_back(h_gdo[i] + sg * pad);
            bps.push_back(pad);
            bac.push_back(act);
            bgo.push_back(h_ggo[i] + sg);
        }
    }
    for (int b = 0; b < num_large; b++) {
        bzo.push_back(h_gdo[LARGE_GROUP] + h_large_off[b]);
        bps.push_back(0);
        bac.push_back(h_large_cnt[b]);
        bgo.push_back(h_ggo[LARGE_GROUP] + b);
    }
    bzo.push_back(z_data_size);

    const int total_blocks = static_cast<int>(bps.size());

    // ---- Allocate z_data, labels_buf, intermediate output ------------------
    float   *d_z_data = nullptr;
    uint8_t *d_labels_buf = nullptr;
    float   *d_gmz_buf = nullptr, *d_gvz_buf = nullptr;
    uint8_t *d_gig_buf = nullptr;

    if (z_data_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_z_data,     z_data_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_z_data, 0,   z_data_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels_buf, z_data_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(d_labels_buf, 0, z_data_size * sizeof(uint8_t)));
    }
    if (total_output_grids > 0) {
        CUDA_CHECK(cudaMalloc(&d_gmz_buf, total_output_grids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gvz_buf, total_output_grids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gig_buf, total_output_grids * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(d_gmz_buf, 0, total_output_grids * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gvz_buf, 0, total_output_grids * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gig_buf, 0, total_output_grids * sizeof(uint8_t)));
    }

    // ---- Upload metadata arrays --------------------------------------------
    int *d_gdo, *d_ps, *d_ggo;
    CUDA_CHECK(cudaMalloc(&d_gdo, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ps,  NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ggo, NUM_GROUPS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_gdo, h_gdo, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ps,  h_pad, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ggo, h_ggo, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice));

    int *d_bzo = nullptr, *d_bps = nullptr, *d_bac = nullptr, *d_bgo = nullptr;
    if (total_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&d_bzo, (total_blocks + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bps, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bac, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bgo, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bzo, bzo.data(), (total_blocks + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bps, bps.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bac, bac.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bgo, bgo.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice));
    }
    lap("CPU metadata + alloc bufs + H2D tables");

    // ---- Phase 3: scatter z values into organized z_data -------------------
    if (z_data_size > 0) {
        scatterToZdataKernel<<<blocks_N, BLOCK_SIZE>>>(
            d_z, d_grid_idx, d_grid_group, d_grid_row_in_group, d_point_row,
            d_gdo, d_ps, d_large_offsets, d_z_data, N);
        CUDA_CHECK(cudaGetLastError());
    }
    lap("scatterToZdataKernel");

    // ---- Phase 4: reduction ------------------------------------------------
    if (total_blocks > 0) {
        reductionKernel<<<total_blocks, BLOCK_SIZE,
                          BLOCK_SIZE * sizeof(float)>>>(
            d_z_data, d_bzo, d_bps, d_bac, d_bgo,
            d_gmz_buf, d_gvz_buf, d_gig_buf, d_labels_buf,
            params.min_variance_threshold, params.height_threshold);
        CUDA_CHECK(cudaGetLastError());
    }
    lap("reductionKernel");

    // ---- Phase 5: remap grid stats → real grid-indexed arrays --------------
    float   *d_grid_mean_z, *d_grid_var_z;
    uint8_t *d_grid_is_ground, *d_labels;
    CUDA_CHECK(cudaMalloc(&d_grid_mean_z,    NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_var_z,     NG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_is_ground, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_labels,         N  * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_grid_mean_z,    0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_var_z,     0, NG * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid_is_ground, 0, NG * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_labels,         0, N  * sizeof(uint8_t)));

    if (total_output_grids > 0) {
        remapGridStatsKernel<<<blocks_NG, BLOCK_SIZE>>>(
            d_grid_group, d_grid_row_in_group, d_ggo,
            d_gmz_buf, d_gvz_buf, d_gig_buf,
            d_grid_mean_z, d_grid_var_z, d_grid_is_ground, NG);
        CUDA_CHECK(cudaGetLastError());
    }
    lap("alloc output + remapGridStatsKernel");

    // ---- Phase 5: empty-grid neighbour classification ----------------------
    emptyGridKernel<<<blocks_NG, BLOCK_SIZE>>>(
        d_grid_group, d_grid_var_z, d_grid_is_ground,
        params.min_variance_threshold,
        params.num_cols, params.num_rows, NG);
    CUDA_CHECK(cudaGetLastError());
    lap("emptyGridKernel");

    // ---- Phase 5: remap labels (+ set mean_z for 1-point grids) ------------
    remapLabelsKernel<<<blocks_N, BLOCK_SIZE>>>(
        d_z, d_grid_idx, d_grid_group, d_grid_count,
        d_grid_row_in_group, d_point_row,
        d_gdo, d_ps, d_large_offsets,
        d_grid_is_ground, d_labels_buf,
        d_labels, d_grid_mean_z, N);
    CUDA_CHECK(cudaGetLastError());
    lap("remapLabelsKernel");

    // ---- D2H: final results ------------------------------------------------
    result.labels.resize(N);
    result.grid_mean_z.resize(NG);
    result.grid_var_z.resize(NG);
    result.grid_is_ground.resize(NG);
    result.grid_count.resize(NG);

    CUDA_CHECK(cudaMemcpy(result.labels.data(),         d_labels,
                          N  * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_mean_z.data(),    d_grid_mean_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_var_z.data(),     d_grid_var_z,
                          NG * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_is_ground.data(), d_grid_is_ground,
                          NG * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grid_count.data(),     d_grid_count,
                          NG * sizeof(int),     cudaMemcpyDeviceToHost));
    lap("D2H final results");

    // ---- Cleanup -----------------------------------------------------------
    cudaFree(d_z);
    cudaFree(d_grid_idx);       cudaFree(d_point_row);
    cudaFree(d_grid_count);     cudaFree(d_grid_group);
    cudaFree(d_grid_row_in_group); cudaFree(d_group_grid_count);
    if (d_large_counts)  cudaFree(d_large_counts);
    if (d_large_offsets) cudaFree(d_large_offsets);
    if (d_z_data)        cudaFree(d_z_data);
    if (d_labels_buf)    cudaFree(d_labels_buf);
    if (d_gmz_buf)       cudaFree(d_gmz_buf);
    if (d_gvz_buf)       cudaFree(d_gvz_buf);
    if (d_gig_buf)       cudaFree(d_gig_buf);
    cudaFree(d_gdo);  cudaFree(d_ps);  cudaFree(d_ggo);
    if (d_bzo) cudaFree(d_bzo);
    if (d_bps) cudaFree(d_bps);
    if (d_bac) cudaFree(d_bac);
    if (d_bgo) cudaFree(d_bgo);
    cudaFree(d_grid_mean_z);    cudaFree(d_grid_var_z);
    cudaFree(d_grid_is_ground); cudaFree(d_labels);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    result.grid_min_z.resize(NG, 0.0f);
    result.grid_max_z.resize(NG, 0.0f);

    return result;
}
