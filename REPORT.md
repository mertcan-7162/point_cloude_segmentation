# Point Cloud Ground Segmentation on GPU - Technical Report

**Kuartis - Computer Vision, Machine Learning and Robotics**
**Calibration, Localization and Mapping Team**

---

## 1. Introduction

This report documents the design and implementation of a 3D point cloud ground segmentation algorithm. The goal is to classify each point in a LiDAR-generated point cloud as either **ground** or **nonground**. The algorithm is implemented in both **CPU (C++)** and **GPU (CUDA)**, and the results are compared for correctness and performance.

The coordinate system follows the convention specified in the assignment:
- **X axis**: right
- **Y axis**: front
- **Z axis**: up (height)

### 1.1. Libraries and Tools

| Component | Version |
|-----------|---------|
| PCL (Point Cloud Library) | 1.10+ |
| CUDA Toolkit | (system-installed) |
| GCC (g++) | (system-installed) |
| yaml-cpp | (system-installed) |
| CMake | >= 3.18 |
| Thrust | (bundled with CUDA) |

---

## 2. Project Structure

```
kuartis_project/
├── CMakeLists.txt              # Build configuration (C++ & CUDA)
├── config/
│   └── params.yaml             # Algorithm parameters
├── data/
│   └── input_data.pcd          # Input point cloud (PCD format)
├── include/
│   ├── types.h                 # Shared data structures (Params, PointCloud, PipelineResult)
│   ├── pcd_io.h                # PCD read/write interface
│   ├── yaml_parser.h           # YAML parameter parser interface
│   ├── ground_seg_cpu.h        # CPU pipeline interface
│   └── ground_seg_gpu.cuh      # GPU pipeline interface
├── src/
│   ├── main.cpp                # Entry point: runs both CPU & GPU, compares results
│   ├── pcd_io.cpp              # PCD I/O using PCL
│   ├── yaml_parser.cpp         # YAML parsing using yaml-cpp
│   ├── ground_seg_cpu.cpp      # CPU implementation
│   └── ground_seg_gpu.cu       # GPU (CUDA) implementation
└── output/
    ├── ground.pcd              # Output: ground points
    └── nonground.pcd           # Output: nonground points
```

---

## 3. Configuration Parameters

All algorithm parameters are loaded from `config/params.yaml`:

```yaml
pointcloud_limits:
  x_min: -50.0   # meters
  x_max:  50.0   # meters
  y_min:   0.0   # meters
  y_max: 100.0   # meters

grid_resolution: 0.3           # meters per grid cell
min_variance_threshold: 0.05   # Z variance threshold for ground classification
point_number_threshold: 2      # minimum points to consider a grid non-empty
height_threshold:
  ground: 0.30                 # Z difference threshold for point classification (meters)
```

With these values, the 2D grid dimensions are:
- **Columns** (X axis): ceil((50 - (-50)) / 0.3) = **334**
- **Rows** (Y axis): ceil((100 - 0) / 0.3) = **334**
- **Total grid cells**: 334 x 334 = **111,556**

---

## 4. Algorithm Overview

The pipeline consists of 6 sequential phases:

```
Input PCD  ──►  Phase 1: Filter & Grid Assignment
               ──►  Phase 2: Per-Grid Statistics (mean, var, min, max of Z)
                    ──►  Phase 3: Grid Classification (ground vs nonground via 3x3 patch)
                         ──►  Phase 4: Point Classification (Z difference check)
                              ──►  Output: ground.pcd + nonground.pcd
```

---

## 5. Data Structures

Three core data structures are defined in `include/types.h`:

### Params
Holds all configuration parameters (limits, resolution, thresholds) and computes the grid dimensions (`num_cols`, `num_rows`, `num_grids`).

### PointCloud
Stores point cloud data in **Structure of Arrays (SoA)** layout:
- `x`, `y`, `z`: separate `std::vector<float>` for each coordinate
- `num_points`: total number of points

**Design choice**: SoA layout is preferred over Array of Structures (AoS) because it enables **coalesced memory access** on the GPU. When a kernel reads all X values, consecutive threads access consecutive memory addresses, maximizing memory bandwidth utilization.

### PipelineResult
Contains all outputs: per-point labels (ground=1, nonground=0), per-grid statistics (mean, variance, min, max of Z, point count, ground flag), and elapsed time.

---

## 6. I/O Modules

### 6.1. PCD Reader/Writer (`pcd_io.cpp`)

Uses PCL's `pcl::io::loadPCDFile` to read `PointXYZI` data and extracts only the X, Y, Z coordinates into our SoA `PointCloud` structure. Output is written using `pcl::io::savePCDFileBinary` by filtering points based on their label.

### 6.2. YAML Parser (`yaml_parser.cpp`)

Uses `yaml-cpp` to load all parameters from the YAML file and computes derived grid dimensions via `Params::computeGridDims()`.

---

## 7. CPU Implementation (`ground_seg_cpu.cpp`)

The CPU implementation serves as the **reference (baseline)** for verifying GPU correctness.

### Step 1: Grid Assignment & Filtering

Each point is checked against the X/Y limits. If within bounds, its grid cell index is computed:
```
col = floor((x - x_min) / resolution)
row = floor((y - y_min) / resolution)
grid_index = row * num_cols + col
```
Out-of-bounds points receive `grid_idx = -1`.

### Step 2: Per-Grid Statistics

For each grid cell, a list of Z values is collected. Then:
- **Count**: number of points in the cell
- **Mean Z**: sum(z) / count
- **Variance Z**: sum((z - mean)^2) / count
- **Min Z / Max Z**: tracked during collection

### Step 3: Grid Classification (3x3 Patch)

For each grid cell:
- If `count >= point_number_threshold`: the cell is ground if `variance < min_variance_threshold`
- If `count < point_number_threshold` (empty/sparse): compute the **average variance** of all valid cells in the 3x3 neighborhood, and classify as ground if that average is below the threshold

### Step 4: Point Classification

For each point on a **ground grid**: if `|z - mean_z| < height_threshold`, it is classified as ground. All points on nonground grids or out-of-bounds are classified as nonground.

---

## 8. GPU Implementation (`ground_seg_gpu.cu`)

The GPU implementation parallelizes the entire pipeline using **6 CUDA kernels**. A key design challenge is computing per-grid reductions (mean, variance, min, max) efficiently when grid cell populations vary widely (from 2 to hundreds of points).

### 8.1. Phase 1: Grid Assignment & Histogram (`assignAndHistogramKernel`)

**One thread per point.** Each thread:
1. Checks if the point is within X/Y bounds
2. Computes `col`, `row`, `grid_index`
3. Uses `atomicAdd` on `grid_count[grid_index]` to simultaneously count points per grid and obtain a per-point **row index** within its grid

This fused approach avoids a separate counting pass.

### 8.2. Phase 2: Group Assignment (`groupAssignAndHistogramKernel`)

**One thread per grid cell.** Grid cells are grouped by their point count to enable efficient batched reductions later:

| Point Count | Group ID | Strategy |
|------------|----------|----------|
| 0-1 | -1 (skipped) | Too few points |
| 2-32 | 0-30 | Warp-level shuffle reduction |
| >32 | 31 (LARGE_GROUP) | Shared-memory block reduction |

Each grid cell atomically registers itself within its group.

### 8.3. Phase 3: Z-Data Construction (`scatterToZdataKernel`)

**One thread per point.** Points are scattered into a contiguous `z_data` buffer organized by group. For groups 0-30 (2-32 points), Z values are padded to the next power of 2 to enable efficient warp-level butterfly reduction. For the large group (>32 points), an exclusive prefix scan (Thrust `exclusive_scan`) computes offsets.

The data layout in `z_data`:

```
[ Group 0 (pad=2) | Group 1 (pad=4) | ... | Group 30 (pad=32) | Large Group (variable) ]
  ↑ each grid's Z values padded to power-of-2
```

### 8.4. Phase 4: Unified Reduction Kernel (`reductionKernel`)

This is the core computation kernel. It computes **mean, variance, min, max** for each grid cell. Two strategies are used within the same kernel:

#### Small Groups (2-32 points): Warp-Level Butterfly Reduction

Multiple grid cells are packed into a single warp (32 threads). Uses `__shfl_xor_sync` for register-level reduction with zero shared memory cost:

1. **Sum reduction** via butterfly XOR shuffle → compute **mean**
2. Compute `(z - mean)^2` per element
3. **Sum of squared differences** via butterfly shuffle → compute **variance**
4. **Min/Max** via shuffle + `fminf`/`fmaxf`

This approach is extremely efficient for small groups since all data stays in registers.

#### Large Groups (>32 points): Block-Level Shared Memory Reduction

Each large grid cell gets its own thread block. Uses classic shared memory parallel reduction:

1. Each thread accumulates partial sum, min, max across its strided elements
2. Shared memory tree reduction down to warp level
3. Final warp-level `__shfl_down_sync` reduction
4. Second pass for variance: compute `(z - mean)^2` and reduce again

Shared memory layout: `[sum | min | max]` = `3 * blockDim.x` floats.

### 8.5. Phase 5: Ground Classification (`groundClassificationKernel`)

**2D thread blocks (16x16 tiles).** Uses **shared memory with halo** for the 3x3 neighborhood access pattern:

- Each tile loads a `(TILE+2) x (TILE+2)` region into shared memory
- Border threads load the halo cells (1-pixel border around the tile)
- Corner threads load the diagonal halo cells
- After `__syncthreads()`, each thread evaluates the 3x3 patch from shared memory

This avoids redundant global memory reads for overlapping neighborhoods.

### 8.6. Phase 6: Point Classification (`pointClassificationKernel`)

**One thread per point.** Simple kernel:
- If grid is nonground → label = 0
- If grid is ground and `|z - mean_z| < height_threshold` → label = 1
- Otherwise → label = 0

### 8.7. Memory Management

All GPU memory is allocated with `cudaMalloc` and freed after the pipeline completes. A `CUDA_CHECK` macro wraps every CUDA API call to catch errors with file/line information.

---

## 9. Build System

The `CMakeLists.txt` configures a single executable target `ground_seg` that compiles both C++ (`.cpp`) and CUDA (`.cu`) source files:

```cmake
project(cuda_ground_segmentation LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
```

Key settings:
- **CUDA_SEPARABLE_COMPILATION**: enabled for linking CUDA device code across translation units
- **CUDA_ARCHITECTURES**: set to 89 (Ada Lovelace / RTX 4000 series)
- **Dependencies**: PCL (common, io), yaml-cpp

### Build Instructions

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run

```bash
./ground_seg ../data/input_data.pcd ../config/params.yaml
```

---

## 10. Results and Performance

The main program runs both CPU and GPU pipelines on the same input data and compares:

1. **Correctness**: Per-point label agreement between CPU and GPU
2. **Performance**: Wall-clock time for each pipeline
3. **Output**: Ground and nonground point clouds saved as separate PCD files

### 10.1. Timing Breakdown (GPU)

The GPU pipeline measures each phase individually:

| Phase | Description |
|-------|-------------|
| H2D upload + alloc | Host-to-device data transfer |
| assignAndHistogramKernel | Grid assignment with atomic histogram |
| groupAssignAndHistogramKernel | Group bucketing |
| Large-group offsets (scan) | Thrust prefix scan for large groups |
| CPU metadata + tables | Offset/block table computation on CPU |
| scatterToZdataKernel | Z value scatter into organized buffer |
| reductionKernel | Mean, variance, min, max computation |
| groundClassificationKernel | 3x3 patch ground decision |
| pointClassificationKernel | Final per-point labeling |
| D2H final results | Device-to-host result transfer |

---

## 11. Design Decisions and Trade-offs

### 11.1. SoA vs AoS Memory Layout

**Choice**: Structure of Arrays (SoA) — separate arrays for X, Y, Z.

**Rationale**: GPU kernels typically process one coordinate at a time (e.g., only X and Y in grid assignment). SoA ensures coalesced global memory accesses, which is critical for GPU bandwidth utilization. The downside is slightly more complex host code, but the performance benefit on GPU far outweighs this.

### 11.2. Group-Based Reduction Strategy

**Choice**: Bucket grid cells by point count into 32 groups.

**Rationale**: Grid cells have highly variable point counts. A one-size-fits-all reduction would waste threads for small cells or lack parallelism for large cells. The group-based approach:
- **Groups 0-30 (2-32 points)**: Pack multiple cells into a warp using butterfly shuffle — no shared memory, no synchronization barriers, maximum throughput
- **Group 31 (>32 points)**: Full block-level shared memory reduction

This dual strategy achieves near-optimal occupancy across all cell sizes.

### 11.3. Shared Memory Halo for 3x3 Classification

**Choice**: 16x16 tiles with 18x18 shared memory (including 1-cell halo).

**Rationale**: Each cell's ground classification depends on its 3x3 neighborhood (9 global memory reads per cell). By loading the tile + halo into shared memory once, each cell only requires fast shared memory reads. The 16x16 tile size balances shared memory usage with occupancy.

### 11.4. Fused Kernels

Grid assignment and histogram computation are fused into a single kernel (`assignAndHistogramKernel`) to avoid an extra pass over all points. Similarly, group assignment and group histogram are fused.

---

## 12. Conclusion

The implemented ground segmentation pipeline successfully classifies LiDAR point cloud data into ground and nonground categories using a grid-based variance analysis approach. The GPU implementation leverages multiple CUDA optimization techniques:

- **Coalesced memory access** via SoA layout
- **Warp-level shuffle reductions** for small grid cells
- **Shared memory halo pattern** for neighborhood-based classification
- **Fused kernels** to minimize kernel launch overhead and memory traffic
- **Group-based batching** to handle variable workload sizes efficiently

The CPU implementation provides a reliable baseline for correctness validation, and the comparison between CPU and GPU results ensures that the CUDA parallelization preserves algorithmic correctness while delivering significant speedup.
