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
/*
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
*/
void LabelGenerator::addFootprint(HeightMap &map, grid_map::Position &robot_position) {

  ensureLabelLayers(map);

  // Iterate over the footprint radius
  grid_map::CircleIterator iterator(map, robot_position, cfg.footprint_radius);
  for (iterator; !iterator.isPastEnd(); ++iterator) {
    if (map.isEmptyAt(*iterator))
      continue;

    // --- 修改的核心部分：注释掉阻碍轨迹写入的保护机制 ---
    // auto is_non_traversable = std::abs(map.at(layers::Label::TRAVERSABILITY, *iterator) -
    //                                    (float)Traversability::NON_TRAVERSABLE) < 1e-3;
    // if (is_non_traversable)
    //   continue;
    // --------------------------------------------------

    // 强行写入足迹并标记为可通行
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

    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0) < 1e-3;

    // --- 修改的核心部分：物理经验拥有最高优先级 ---
    if (has_footprint) {
      // 1. 只要被机器人碾压过，强制标记为可通行
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    } 
    else if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step) {
      // 2. 没有轨迹时，才依靠几何特征（STEP）判定为障碍物
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::NON_TRAVERSABLE;
    } 
    else {
      // 3. 其他安全的未知区域
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
    }
  }
}

} // namespace lesta
