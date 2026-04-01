/*
 * FeatureExtractor.cpp
 *
 *  Created on: Feb 07, 2025
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/core/FeatureExtractor.h"
#include <Eigen/Dense>

namespace lesta {

FeatureExtractor::FeatureExtractor(const Config &cfg) : cfg(cfg) {}

void FeatureExtractor::ensureFeatureLayers(HeightMap &map) {
  // Basic feature layers
  map.addLayer(layers::Feature::STEP);
  map.addLayer(layers::Feature::SLOPE);
  map.addLayer(layers::Feature::ROUGHNESS);
  map.addLayer(layers::Feature::CURVATURE);
  
  // 确保新的特征层被初始化
  map.addLayer(layers::Feature::VARIANCE);
  map.addLayer(layers::Feature::INTENSITY_MEAN);
  map.addLayer(layers::Feature::INTENSITY_VAR);
  map.addLayer(layers::Feature::SPARSITY);

  // Layers for visualization of normal vector
  map.addLayer(layers::Feature::NORMAL_X);
  map.addLayer(layers::Feature::NORMAL_Y);
  map.addLayer(layers::Feature::NORMAL_Z);
}

void FeatureExtractor::extractFeatures(HeightMap &map) {

  ensureFeatureLayers(map);
  // TODO: implement feature extraction for entire map (less efficient)
}

void FeatureExtractor::extractFeatures(
    HeightMap &map,
    const std::vector<grid_map::Index> &measured_indices) {

  ensureFeatureLayers(map);

  for (const auto &index : measured_indices) {
    if (map.isEmptyAt(index))
      continue;

    const auto &neighbors = map.getNeighborHeights(index, cfg.pca_radius);
    if (neighbors.size() < 4)
      continue;

    // 1. Compute Covariance Matrix
    Eigen::Matrix3d covariance;
    Eigen::Vector3d sum_neighbors(Eigen::Vector3d::Zero());
    Eigen::Matrix3d squared_sum_neighbors(Eigen::Matrix3d::Zero());
    for (const auto &neighbor : neighbors) {
      sum_neighbors += neighbor;
      squared_sum_neighbors.noalias() += neighbor * neighbor.transpose();
    }
    const auto mean_neighbors = sum_neighbors / neighbors.size();
    covariance = squared_sum_neighbors / neighbors.size() -
                 mean_neighbors * mean_neighbors.transpose();

    // Check if covariance matrix is degenerated using trace
    if (covariance.trace() < std::numeric_limits<float>::epsilon())
      continue;

    // Compute Eigenvectors and Eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.computeDirect(covariance, Eigen::DecompositionOptions::ComputeEigenvectors);
    const auto &eigenvectors = solver.eigenvectors();
    const auto &eigenvalues = solver.eigenvalues();

    // Line feature: second eigen value is near zero -> normal is not defined
    if (eigenvalues(1) < 1e-8)
      continue;

    // Check direction of the normal vector and flip the sign towards the user defined
    // direction.
    Eigen::Vector3d normal_vector = eigenvectors.col(0);
    Eigen::Vector3d positive_normal_vector(Eigen::Vector3d::UnitZ());
    if (normal_vector.dot(positive_normal_vector) < 0.0)
      normal_vector *= -1;

    // Calculate step
    auto minMax =
        std::minmax_element(neighbors.begin(),
                            neighbors.end(),
                            [](const Eigen::Vector3d &lhs, const Eigen::Vector3d &rhs) {
                              return lhs(2) < rhs(2); // Compare z-components.
                            });

    double minZ = (*minMax.first)(2);  // Minimum z-component.
    double maxZ = (*minMax.second)(2); // Maximum z-component.
    map.at(layers::Feature::STEP, index) = maxZ - minZ;

    map.at(layers::Feature::SLOPE, index) =
        std::acos(std::abs(normal_vector(2))) * 180 / M_PI;
    map.at(layers::Feature::ROUGHNESS, index) = std::sqrt(eigenvalues(0));
    map.at(layers::Feature::CURVATURE, index) =
        std::abs(eigenvalues(0) / covariance.trace());
    // 补充：几何方差（Z轴高度方差）
    map.at(layers::Feature::VARIANCE, index) = covariance(2, 2);
    // 新增 1：计算局部致密度 (Density) 代替稀疏度，解决长尾分布导致的 Logit 爆炸问题
    // 设定邻域内有效点数的上限（例如 100 个点），将局部点数线性映射到 0.0 ~ 1.0 的安全区间
    const float MAX_POINTS_IN_RADIUS = 100.0f;
    float point_count = static_cast<float>(neighbors.size());
    float density = std::min(point_count, MAX_POINTS_IN_RADIUS) / MAX_POINTS_IN_RADIUS;
    
    // 注意：图层名仍保持 SPARSITY 以兼容现有的网络配置文件，但其物理意义已变为稳定的 Density
    map.at(layers::Feature::SPARSITY, index) = density;
    map.at(layers::Feature::NORMAL_X, index) = normal_vector(0);
    map.at(layers::Feature::NORMAL_Y, index) = normal_vector(1);
    map.at(layers::Feature::NORMAL_Z, index) = normal_vector(2);
    // 新增 2：预留强度特征槽位
    // 假设上游的 HeightMapper 已经将 INTENSITY_MEAN 和 INTENSITY_VAR 写入 GridMap，
    // 这里只需确保提取时包含这些层即可。如果没有写入，可以在此处设置默认值避免报错：
    if (!map.isValid(index, layers::Feature::INTENSITY_MEAN)) {
        map.at(layers::Feature::INTENSITY_MEAN, index) = 0.0f; // 或对接你的点云强度提取逻辑
        map.at(layers::Feature::INTENSITY_VAR, index) = 0.0f;
  }
}
}
} // namespace lesta
