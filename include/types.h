#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

struct Params {
    float x_min, x_max;
    float y_min, y_max;
    float grid_resolution;
    float min_variance_threshold;
    int point_number_threshold;
    float height_threshold;

    int num_cols;
    int num_rows;
    int num_grids;

    void computeGridDims() {
        num_cols = static_cast<int>(std::ceil((x_max - x_min) / grid_resolution));
        num_rows = static_cast<int>(std::ceil((y_max - y_min) / grid_resolution));
        num_grids = num_cols * num_rows;
    }
};

struct PointCloud {
    std::vector<float> x, y, z;
    int num_points;
};

struct PipelineResult {
    std::vector<uint8_t> labels;
    std::vector<float> grid_mean_z;
    std::vector<float> grid_var_z;
    std::vector<float> grid_min_z;
    std::vector<float> grid_max_z;
    std::vector<int> grid_count;
    std::vector<uint8_t> grid_is_ground;
    double elapsed_ms;
};
