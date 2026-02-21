#include "types.h"
#include "pcd_io.h"
#include "yaml_parser.h"
#include "ground_seg_cpu.h"
#include "ground_seg_gpu.cuh"

#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    std::string pcd_path = "../data/input_data.pcd";
    std::string yaml_path = "../config/params.yaml";
    std::string out_ground = "../output/ground.pcd";
    std::string out_nonground = "../output/nonground.pcd";

    if (argc >= 3) {
        pcd_path = argv[1];
        yaml_path = argv[2];
    }

    // ---- Load data ----
    std::cout << "=== Loading data ===\n";
    Params params = parseYaml(yaml_path);
    PointCloud cloud = readPCD(pcd_path);
    if (cloud.num_points == 0) {
        std::cerr << "No points loaded, exiting.\n";
        return 1;
    }

    // ---- CPU pipeline ----
    std::cout << "\n=== CPU Pipeline ===\n";
    PipelineResult cpu_result = runCPUPipeline(cloud, params);

    // ---- GPU pipeline ----
    std::cout << "\n=== GPU Pipeline ===\n";
    PipelineResult gpu_result = runGPUPipeline(cloud, params);

    // ---- GPU stats ----
    int gpu_ground = 0, gpu_nonground = 0;
    for (int i = 0; i < cloud.num_points; i++) {
        if (gpu_result.labels[i] == 1) gpu_ground++;
        else gpu_nonground++;
    }
    std::cout << "  GPU: ground=" << gpu_ground
              << ", nonground=" << gpu_nonground
              << " (" << gpu_result.elapsed_ms << " ms)\n";

    // ---- Comparison ----
    std::cout << "\n=== Comparison ===\n";
    int match = 0;
    for (int i = 0; i < cloud.num_points; i++) {
        if (cpu_result.labels[i] == gpu_result.labels[i]) match++;
    }
    double accuracy = 100.0 * match / cloud.num_points;
    std::cout << "  CPU vs GPU agreement: " << match << "/" << cloud.num_points
              << " (" << accuracy << "%)\n";
    std::cout << "  Speedup: " << cpu_result.elapsed_ms / gpu_result.elapsed_ms << "x\n";

    // ---- Save output (using GPU result) ----
    std::cout << "\n=== Saving output ===\n";
    writePCD(out_ground, cloud, gpu_result.labels, 1);
    writePCD(out_nonground, cloud, gpu_result.labels, 0);

    std::cout << "\nDone.\n";
    return 0;
}
