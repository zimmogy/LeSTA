/*
 * LabelGenerator.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/core/LabelGenerator.h"

namespace lesta {

LabelGenerator::LabelGenerator(const Config &cfg) : cfg(cfg) {}

void LabelGenerator::ensureLabelLayers(HeightMap &map) {

  map.addLayer(layers::Label::FOOTPRINT, 0.0f);
  map.addLayer(layers::Label::TRAVERSABILITY);
}

void LabelGenerator::addFootprint(HeightMap &map, grid_map::Position &robot_position) {

  ensureLabelLayers(map);

  // Iterate over the footprint radius
  grid_map::CircleIterator iterator(map, robot_position, cfg.footprint_radius);
  for (iterator; !iterator.isPastEnd(); ++iterator) {
    if (map.isEmptyAt(*iterator))
      continue;

    // pass if recoreded as obstacle to prevent noisy label generation
    auto is_non_traversable = std::abs(map.at(layers::Label::TRAVERSABILITY, *iterator) -
                                       (float)Traversability::NON_TRAVERSABLE) < 1e-3;
    if (is_non_traversable)
      continue;

    map.at(layers::Label::FOOTPRINT, *iterator) = 1.0;
    map.at(layers::Label::TRAVERSABILITY, *iterator) = (float)Traversability::TRAVERSABLE;
  }
}
/*
void LabelGenerator::addObstacles(HeightMap &map,
                                  const std::vector<grid_map::Index> &measured_indices) {

  ensureLabelLayers(map);

  for (const auto &index : measured_indices) {

    if (map.isEmptyAt(layers::Feature::SLOPE, index))
      continue;

    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0) < 1e-3;

    if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step)
      map.at(layers::Label::TRAVERSABILITY, index) =
          (float)Traversability::NON_TRAVERSABLE;
    else if (has_footprint) // Noisy label removal
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    else
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
  }
}
*/
// 新增标签生成逻辑，物理经验绝对优先
void LabelGenerator::addObstacles(HeightMap &map,
                                  const std::vector<grid_map::Index> &measured_indices) {
  ensureLabelLayers(map);

  for (const auto &index : measured_indices) {
    if (map.isEmptyAt(layers::Feature::SLOPE, index))
      continue;

    // 检查是否有轨迹覆盖
    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0) < 1e-3;

    // --- 修改开始：调整判断优先级 ---
    if (has_footprint) {
      // 1. 经验绝对优先：被碾压过的区域，强制标记为可通行
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    } 
    else if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step) {
      // 2. 无轨迹区域，再根据高度差(STEP)判定是否为障碍物
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::NON_TRAVERSABLE;
    } 
    else {
      // 3. 几何安全且无轨迹的未知区域
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
    }
    // --- 修改结束 ---
  }
}
} // namespace lesta
