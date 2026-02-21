#include "pcd_io.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <iostream>

PointCloud readPCD(const std::string& filepath) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(filepath, *cloud) == -1) {
        std::cerr << "Failed to read PCD file: " << filepath << "\n";
        return {};
    }

    PointCloud pc;
    pc.num_points = static_cast<int>(cloud->size());
    pc.x.resize(pc.num_points);
    pc.y.resize(pc.num_points);
    pc.z.resize(pc.num_points);

    for (int i = 0; i < pc.num_points; i++) {
        pc.x[i] = (*cloud)[i].x;
        pc.y[i] = (*cloud)[i].y;
        pc.z[i] = (*cloud)[i].z;
    }

    std::cout << "Loaded " << pc.num_points << " points from " << filepath << "\n";
    return pc;
}

void writePCD(const std::string& filepath,
              const PointCloud& cloud,
              const std::vector<uint8_t>& labels,
              uint8_t target_label) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < cloud.num_points; i++) {
        if (labels[i] == target_label) {
            pcl::PointXYZ pt;
            pt.x = cloud.x[i];
            pt.y = cloud.y[i];
            pt.z = cloud.z[i];
            out->push_back(pt);
        }
    }

    out->width = out->size();
    out->height = 1;
    out->is_dense = true;

    pcl::io::savePCDFileBinary(filepath, *out);
    std::cout << "Saved " << out->size() << " points to " << filepath << "\n";
}
