#include "yaml_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

Params parseYaml(const std::string& filepath) {
    YAML::Node config = YAML::LoadFile(filepath);

    Params p{};
    p.x_min = config["pointcloud_limits"]["x_min"].as<float>();
    p.x_max = config["pointcloud_limits"]["x_max"].as<float>();
    p.y_min = config["pointcloud_limits"]["y_min"].as<float>();
    p.y_max = config["pointcloud_limits"]["y_max"].as<float>();
    p.grid_resolution = config["grid_resolution"].as<float>();
    p.min_variance_threshold = config["min_variance_threshold"].as<float>();
    p.point_number_threshold = config["point_number_threshold"].as<int>();
    p.height_threshold = config["height_threshold"]["ground"].as<float>();

    p.computeGridDims();

    std::cout << "Params loaded: "
              << "x=[" << p.x_min << "," << p.x_max << "] "
              << "y=[" << p.y_min << "," << p.y_max << "] "
              << "res=" << p.grid_resolution
              << " grid=" << p.num_cols << "x" << p.num_rows
              << " (" << p.num_grids << " cells)\n";

    return p;
}
