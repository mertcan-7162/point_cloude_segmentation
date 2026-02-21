cmake_minimum_required(VERSION 3.18)
project(cuda_ground_segmentation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

find_package(PCL 1.10 REQUIRED COMPONENTS common io)
find_package(yaml-cpp REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ground_seg
    src/main.cpp
    src/pcd_io.cpp
    src/yaml_parser.cpp
    src/ground_seg_cpu.cpp
    src/ground_seg_gpu.cu
)

target_link_libraries(ground_seg
    ${PCL_LIBRARIES}
    yaml-cpp
)

set_target_properties(ground_seg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
