#pragma once

#include "types.h"
#include <string>

PointCloud readPCD(const std::string& filepath);

void writePCD(const std::string& filepath,
              const PointCloud& cloud,
              const std::vector<uint8_t>& labels,
              uint8_t target_label);
