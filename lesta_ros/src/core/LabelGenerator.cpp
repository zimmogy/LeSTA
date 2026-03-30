/*
 * LabelGenerator.cpp
 *
 * Created on: Feb 10, 2025
 * Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 * Email: tre0430@korea.ac.kr
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

    // 【修改点 1：移除对已存在障碍物标签的避让逻辑】
    // 原代码中，如果该网格被识别为非可通行，则不会写入 footprint。
    // 为了让野外环境中的高草（被误判为障碍物）能被碾压经验覆写，必须删除这部分阻挡逻辑。
    /* 删除以下代码：
    auto is_non_traversable = std::abs(map.at(layers::Label::TRAVERSABILITY, *iterator) -
                                       (float)Traversability::NON_TRAVERSABLE) < 1e-3;
    if (is_non_traversable)
      continue;
    */

    // 强制写入轨迹和可通行标签
    map.at(layers::Label::FOOTPRINT, *iterator) = 1.0;
    map.at(layers::Label::TRAVERSABILITY, *iterator) = (float)Traversability::TRAVERSABLE;
  }
}

void LabelGenerator::addObstacles(HeightMap &map,
                                  const std::vector<grid_map::Index> &measured_indices) {

  ensureLabelLayers(map);

  for (const auto &index : measured_indices) {

    if (map.isEmptyAt(layers::Feature::SLOPE, index))
      continue;

    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0) < 1e-3;

    // 【修改点 2：倒置判定逻辑，实现经验绝对优先】
    // 1. 优先判断是否有轨迹覆盖（如高草被碾压），如果有，强制标记为可通行
    if (has_footprint) {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    } 
    // 2. 在无轨迹覆盖的前提下，再根据几何高度差(STEP)判定是否为刚性障碍物
    else if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step) {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::NON_TRAVERSABLE;
    } 
    // 3. 几何安全且无轨迹的区域，标记为未知
    else {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
    }
  }
}

} // namespace lesta