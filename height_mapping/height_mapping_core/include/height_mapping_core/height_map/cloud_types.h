/*
 * cloud_types.h
 *
 *  Created on: Dec 4, 2024
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#pragma once

#include <pcl/point_types.h>
// ============= new ===============
namespace lesta_types {
// 1. 定义包含我们所有自定义字段的结构体
struct EIGEN_ALIGN16 PointXYZILRGB {
  PCL_ADD_POINT4D;       // 自动添加 x, y, z 及其内存对齐
  float intensity;       // 雷达强度
  float semantic_label;  // 语义标签 ID
  float rgb;             // 包含颜色信息的 float
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
} // namespace lesta_types

// 2. 向 PCL 注册这个结构体，字段名称必须与 Python 脚本中设置的完全一致
POINT_CLOUD_REGISTER_POINT_STRUCT(
    lesta_types::PointXYZILRGB,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, semantic_label, semantic_label)
    (float, rgb, rgb)
)
// ================================

// 使用自定义类型
using Laser = lesta_types::PointXYZILRGB;
// using Laser = pcl::PointXYZI;
using Color = pcl::PointXYZRGB;
