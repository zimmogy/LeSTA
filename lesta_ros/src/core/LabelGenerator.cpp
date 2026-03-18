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

void LabelGenerator::addFootprint(HeightMap &map, grid_map::Position &robot_position, float traversability_score) {

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
    map.at(layers::Label::TRAVERSABILITY, *iterator) = traversability_score;
  }
}

void LabelGenerator::addObstacles(HeightMap &map,
                                  const std::vector<grid_map::Index> &measured_indices) {

  ensureLabelLayers(map);
  // ensure that map include vision layers
  if (!map.exists(layers::Visual::COST)) 
    {map.addLayer(layers::Visual::COST, std::nanf(""));}
  for (const auto &index : measured_indices) {

    if (map.isEmptyAt(layers::Feature::SLOPE, index))
      continue;

    bool has_footprint = std::abs(map.at(layers::Label::FOOTPRINT, index) - 1.0) < 1e-3;
    float step = map.at(layers::Feature::STEP, index);
    float roughness = map.at(layers::Feature::ROUGHNESS, index);  
    float visual_cost = map.at(layers::Visual::COST, index);

    // ======================================
    // Hybrid Pseudo-labeling Strategy
    // ======================================

    // 1. Hard-negative sample from absolute physical obbstacle
    if (step > cfg.max_traversable_step) {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::NON_TRAVERSABLE;
    }
    // 2. Soft positive/negative sample: if there is trajectory, keep  historical score calculated by IMU.
    else if (has_footprint) {
      // nothing todo, keep the score
      continue; 
    }
    // 3. Hard-positive sample with adjustable parameter
    else if (step < 0.03 && roughness < 0.01 && !std::isnan(visual_cost) && visual_cost < cfg.max_traversable_visual_cost) {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    }
    // 4. 其他未知区域：由于没有被探索过，交给之后的网络自己去泛化
    else {
      map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
    }
    
    // if (map.at(layers::Feature::STEP, index) > cfg.max_traversable_step)
    //   map.at(layers::Label::TRAVERSABILITY, index) =
    //       (float)Traversability::NON_TRAVERSABLE;
    // else if (has_footprint) // Noisy label removal
    //   map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::TRAVERSABLE;
    // else
    //   map.at(layers::Label::TRAVERSABILITY, index) = (float)Traversability::UNKNOWN;
  }
}

} // namespace lesta
