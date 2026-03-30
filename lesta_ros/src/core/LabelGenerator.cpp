/*
 * LabelGenerator.cpp
 *
 * Created on: Feb 10, 2025
 * Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 * Email: tre0430@korea.ac.kr
 */

#include "lesta/core/LabelGenerator.h"

// 确保包含类型定义
#include "lesta/types/traversability.h"

namespace lesta {

LabelGenerator::LabelGenerator(const Config &cfg) : cfg(cfg) {}

void LabelGenerator::ensureLabelLayers(HeightMap &map) {
  map.addLayer(layers::Label::FOOTPRINT, 0.0f);
  map.addLayer(layers::Label::TRAVERSABILITY);
}

void LabelGenerator::addFootprint(HeightMap &map, grid_map::Position &robot_position) {
  ensureLabelLayers(map);

  grid_map::CircleIterator iterator(map, robot_position, cfg.footprint_radius);
  for (iterator; !iterator.isPastEnd(); ++iterator) {
    if (map.isEmptyAt(*iterator))
      continue;

    // 强行写入足迹并标记为可通行
    map.at(layers::Label::FOOTPRINT, *iterator) = 1.0f;
    // 使用显式的命名空间，或者直接写 1.0f
    map.at(layers::Label::TRAVERSABILITY, *iterator) = (float)lesta_types::Traversability::TRAVERSABLE; 
  }
}

// 新增标签生成逻辑，物理经验绝对优先
void LabelGenerator::addObstacles(HeightMap &map,
                                  const std::vector<grid_map::Index> &measured_indices) {

  ensureLabelLayers(map);

  for (const auto &index : measured_indices) {

    if (map.isEmptyAt(layers::Feature::SLOPE, index))
      continue;

    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0f) < 1e-3;

    // --- 修复的核心部分：使用 lesta_types:: 确保枚举被正确解析 ---
    if (has_footprint) {
      // 1. 只要被机器人碾压过，强制标记为可通行
      map.at(layers::Label::TRAVERSABILITY, index) = (float)lesta_types::Traversability::TRAVERSABLE;
    } 
    else if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step) {
      // 2. 没有轨迹时，才依靠几何特征（STEP）判定为障碍物
      map.at(layers::Label::TRAVERSABILITY, index) = (float)lesta_types::Traversability::NON_TRAVERSABLE;
    } 
    else {
      // 3. 其他安全的未知区域
      map.at(layers::Label::TRAVERSABILITY, index) = (float)lesta_types::Traversability::UNKNOWN;
    }
  }
}

} // namespace lesta