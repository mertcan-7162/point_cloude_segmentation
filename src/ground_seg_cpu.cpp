#include "ground_seg_cpu.h"

#include <cfloat>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>

PipelineResult runCPUPipeline(const PointCloud& cloud, const Params& params) {
    PipelineResult result;
    const int N = cloud.num_points;
    const int NG = params.num_grids;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Grid assignment + filter ----
    std::vector<int> grid_idx(N);
    for (int i = 0; i < N; i++) {
        float px = cloud.x[i];
        float py = cloud.y[i];
        if (px < params.x_min || px >= params.x_max ||
            py < params.y_min || py >= params.y_max) {
            grid_idx[i] = -1;
            continue;
        }
        int col = static_cast<int>((px - params.x_min) / params.grid_resolution);
        int row = static_cast<int>((py - params.y_min) / params.grid_resolution);
        col = std::min(col, params.num_cols - 1);
        row = std::min(row, params.num_rows - 1);
        grid_idx[i] = row * params.num_cols + col;
    }

    // ---- Build per-grid point lists + min/max ----
    std::vector<std::vector<float>> grid_points(NG);
    std::vector<float> min_z(NG, FLT_MAX);
    std::vector<float> max_z(NG, -FLT_MAX);

    for (int i = 0; i < N; i++) {
        int g = grid_idx[i];
        if (g < 0) continue;
        float z = cloud.z[i];
        grid_points[g].push_back(z);
        if (z < min_z[g]) min_z[g] = z;
        if (z > max_z[g]) max_z[g] = z;
    }

    // ---- Compute per-grid statistics ----
    result.grid_mean_z.resize(NG, 0.0f);
    result.grid_var_z.resize(NG, 0.0f);
    result.grid_min_z.resize(NG, 0.0f);
    result.grid_max_z.resize(NG, 0.0f);
    result.grid_count.resize(NG, 0);
    result.grid_is_ground.resize(NG, 0);

    for (int g = 0; g < NG; g++) {
        int cnt = static_cast<int>(grid_points[g].size());
        result.grid_count[g] = cnt;
        result.grid_min_z[g] = min_z[g];
        result.grid_max_z[g] = max_z[g];

        if (cnt == 0) continue;

        float sum = 0.0f;
        for (float v : grid_points[g]) sum += v;
        float mean = sum / cnt;
        result.grid_mean_z[g] = mean;

        if (cnt >= 2) {
            float sq = 0.0f;
            for (float v : grid_points[g]) sq += (v - mean) * (v - mean);
            result.grid_var_z[g] = sq / cnt;
        }
    }

    // ---- Grid classification ----
    // Non-empty grids (count >= threshold): variance < threshold → ground
    for (int g = 0; g < NG; g++) {
        if (result.grid_count[g] >= params.point_number_threshold) {
            result.grid_is_ground[g] =
                (result.grid_var_z[g] < params.min_variance_threshold) ? 1 : 0;
        }
    }

    // Empty grids (count < threshold): 3x3 neighbor average variance
    for (int g = 0; g < NG; g++) {
        if (result.grid_count[g] >= params.point_number_threshold) continue;

        int row = g / params.num_cols;
        int col = g % params.num_cols;

        float sum = 0.0f;
        int valid = 0;

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int nr = row + dr;
                int nc = col + dc;
                if (nr >= 0 && nr < params.num_rows && nc >= 0 && nc < params.num_cols) {
                    int ng = nr * params.num_cols + nc;
                    sum += result.grid_var_z[ng];
                    valid++;
                }
            }
        }

        if (valid > 0) {
            float avg_var = sum / valid;
            result.grid_is_ground[g] = (avg_var < params.min_variance_threshold) ? 1 : 0;
        }
    }

    // ---- Point classification ----
    result.labels.resize(N, 0);
    for (int i = 0; i < N; i++) {
        int g = grid_idx[i];
        if (g < 0) {
            result.labels[i] = 0;
            continue;
        }
        if (!result.grid_is_ground[g]) {
            result.labels[i] = 0;
            continue;
        }
        float diff = std::fabs(cloud.z[i] - result.grid_mean_z[g]);
        result.labels[i] = (diff < params.height_threshold) ? 1 : 0;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Stats
    int ground_count = 0, nonground_count = 0;
    for (int i = 0; i < N; i++) {
        if (result.labels[i] == 1) ground_count++;
        else nonground_count++;
    }
    printf("  CPU: ground=%d, nonground=%d (%.1f ms)\n",
           ground_count, nonground_count, result.elapsed_ms);

    return result;
}
