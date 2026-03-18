/*
 * HeightMapping.cpp
 *
 *  Created on: Aug 17, 2023
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/core/HeightMapper.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

namespace lesta {

HeightMapper::HeightMapper(const Config &cfg)
    : cfg{cfg}, heightFilter_{cfg.min_height, cfg.max_height} {

  initMap();
  initHeightEstimator();
}

void HeightMapper::initMap() {

  // Check parameter validity
  if (cfg.grid_resolution <= 0) {
    throw std::invalid_argument(
        "[height_mapping::HeightMapper]: Grid resolution must be positive");
  }
  if (cfg.map_length_x <= 0 || cfg.map_length_y <= 0) {
    throw std::invalid_argument(
        "[height_mapping::HeightMapper]: Map dimensions must be positive");
  }

  // Initialize map geometry
  map_.setFrameId(cfg.frame_id);
  map_.setGeometry(grid_map::Length(cfg.map_length_x, cfg.map_length_y),
                   cfg.grid_resolution);
}

void HeightMapper::initHeightEstimator() {

  // Set height estimator
  // - Kalman Filter
  // - Moving Average
  // - StatMean (by default)

  if (cfg.estimator_type == "KalmanFilter") {
    height_estimator_ = std::make_unique<height_mapping::KalmanEstimator>();

    std::cout << "\033[1;33m[height_mapping::HeightMapper]: Height estimator "
                 "type --> KalmanFilter \033[0m\n";

  } else if (cfg.estimator_type == "MovingAverage") {
    height_estimator_ = std::make_unique<height_mapping::MovingAverageEstimator>();

    std::cout << "\033[1;33m[height_mapping::HeightMapper]: Height estimator "
                 "type --> MovingAverage \033[0m\n";

  } else if (cfg.estimator_type == "StatMean") {
    height_estimator_ = std::make_unique<height_mapping::StatMeanEstimator>();

    std::cout << "\033[1;33m[height_mapping::HeightMapper]: Height estimator "
                 "type --> StatisticalMeanEstimator "
                 "\033[0m\n";

  } else {
    height_estimator_ = std::make_unique<height_mapping::StatMeanEstimator>();

    std::cout << "\033[1;33m[height_mapping::HeightMapper] Invalid height "
                 "estimator type. Set Default: StatMeanEstimator \033[0m\n";
  }
}

template <typename PointT>
typename boost::shared_ptr<pcl::PointCloud<PointT>> HeightMapper::heightMapping(
    const typename boost::shared_ptr<pcl::PointCloud<PointT>> &cloud) {

  // 1. Rasterize pointcloud
  auto cloud_rasterized = cloudRasterization<PointT>(cloud, cfg.grid_resolution);
  if (cloud_rasterized->empty()) {
    std::cout << "\033[1;31m[height_mapping::HeightMapper]: Height estimation failed. "
                 "Rasterized cloud is empty\033[0m\n";
    return nullptr;
  }

  // 2. Update height map
  height_estimator_->estimate(map_, *cloud_rasterized);
  return cloud_rasterized;
}

template <typename PointT>
void HeightMapper::fastHeightFilter(
    const typename boost::shared_ptr<pcl::PointCloud<PointT>> &cloud,
    typename boost::shared_ptr<pcl::PointCloud<PointT>> &filtered_cloud) {

  heightFilter_.filter<PointT>(cloud, filtered_cloud);
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
HeightMapper::cloudRasterization(const typename pcl::PointCloud<PointT>::Ptr &cloud,
                                 float gridSize) {

  if (cloud->empty()) {
    return cloud;
  }

  // grid cell: first, second -> (x_index, y_index), point
  std::unordered_map<std::pair<int, int>, PointT, pair_hash> grid_map;
  for (const auto &point : *cloud) {
    int x_index = std::floor(point.x / gridSize);
    int y_index = std::floor(point.y / gridSize);

    auto grid_key = std::make_pair(x_index, y_index);
    auto [iter, inserted] = grid_map.try_emplace(grid_key, point);

    if (!inserted && point.z > iter->second.z) {
      iter->second = point;
    }
  }

  auto cloud_downsampled = boost::make_shared<pcl::PointCloud<PointT>>();
  cloud_downsampled->reserve(grid_map.size());
  for (const auto &grid_cell : grid_map) {
    cloud_downsampled->points.emplace_back(grid_cell.second);
  }

  cloud_downsampled->header = cloud->header;
  return cloud_downsampled;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
HeightMapper::cloudRasterizationAlt(const typename pcl::PointCloud<PointT>::Ptr &cloud,
                                    float gridSize) {

  if (cloud->empty()) {
    return cloud;
  }

  std::unordered_map<std::pair<int, int>, PointT, pair_hash> gridCells;
  grid_map::Position measuredPosition;
  grid_map::Index measuredIndex;

  for (const auto &point : *cloud) {
    measuredPosition << point.x, point.y;

    if (!map_.getIndex(measuredPosition, measuredIndex)) {
      continue; // Skip points outside map bounds
    }

    auto gridIndex = std::make_pair(measuredIndex.x(), measuredIndex.y());
    auto [iter, inserted] = gridCells.try_emplace(gridIndex, point);

    if (inserted) {
      // If new point, get exact grid center position
      iter->second = point;
      map_.getPosition(measuredIndex, measuredPosition);
      iter->second.x = measuredPosition.x();
      iter->second.y = measuredPosition.y();
    } else if (point.z > iter->second.z) {
      // If higher point, update z while keeping grid center position
      iter->second = point;
      map_.getPosition(measuredIndex, measuredPosition);
      iter->second.x = measuredPosition.x();
      iter->second.y = measuredPosition.y();
    }
  }

  auto cloudDownsampled = boost::make_shared<pcl::PointCloud<PointT>>();
  cloudDownsampled->reserve(gridCells.size());
  for (const auto &gridCell : gridCells) {
    cloudDownsampled->points.emplace_back(gridCell.second);
  }

  cloudDownsampled->header = cloud->header;
  return cloudDownsampled;
}

template <typename PointT>
void HeightMapper::raycasting(
    const Eigen::Vector3f &sensorOrigin,
    const typename boost::shared_ptr<pcl::PointCloud<PointT>> &cloud) {
  if (cloud->empty()) {
    return;
  }
  raycaster_.correctHeight(map_, *cloud, sensorOrigin);
}

void HeightMapper::moveMapOrigin(const grid_map::Position &position) {
  map_.move(position);
}

/////////////////////////////////////////////////////////////////////////////
///////////////// EXPLICIT INSTANTIATION OF TEMPLATE FUNCTIONS //////////////
/////////////////////////////////////////////////////////////////////////////
// Laser
template void HeightMapper::fastHeightFilter<Laser>(
    const typename pcl::PointCloud<Laser>::Ptr &cloud,
    typename pcl::PointCloud<Laser>::Ptr &filtered_cloud);
template typename pcl::PointCloud<Laser>::Ptr
HeightMapper::heightMapping<Laser>(const pcl::PointCloud<Laser>::Ptr &cloud);

template pcl::PointCloud<Laser>::Ptr
HeightMapper::cloudRasterization<Laser>(const pcl::PointCloud<Laser>::Ptr &cloud,
                                        float gridSize);

template pcl::PointCloud<Laser>::Ptr
HeightMapper::cloudRasterizationAlt<Laser>(const pcl::PointCloud<Laser>::Ptr &cloud,
                                           float gridSize);

template void
HeightMapper::raycasting<Laser>(const Eigen::Vector3f &sensorOrigin,
                                const typename pcl::PointCloud<Laser>::Ptr &cloud);

// Color
template void HeightMapper::fastHeightFilter<Color>(
    const typename pcl::PointCloud<Color>::Ptr &cloud,
    typename pcl::PointCloud<Color>::Ptr &filtered_cloud);
template typename pcl::PointCloud<Color>::Ptr
HeightMapper::heightMapping<Color>(const pcl::PointCloud<Color>::Ptr &cloud);

template pcl::PointCloud<Color>::Ptr
HeightMapper::cloudRasterization<Color>(const pcl::PointCloud<Color>::Ptr &cloud,
                                        float gridSize);

template pcl::PointCloud<Color>::Ptr
HeightMapper::cloudRasterizationAlt<Color>(const pcl::PointCloud<Color>::Ptr &cloud,
                                           float gridSize);

template void
HeightMapper::raycasting<Color>(const Eigen::Vector3f &sensorOrigin,
                                const typename pcl::PointCloud<Color>::Ptr &cloud);
// [new]
// ============================================================================
void HeightMapper::integrateVisualCost(const pcl::PointCloud<Laser>::Ptr& cloud_base,
                                       const cv::Mat& visual_cost_img,
                                       const Eigen::Matrix4f& T_cam_base,
                                       const cv::Mat& K) {
  if (cloud_base->empty()) return;

  // 1. initialize rellis-3d camera intrinsics K
  cv::Mat K = cv::Mat::zeros(3,3, CV_64F);
  K.at<double>(0,0) = 2813.643275; // fx
  K.at<double>(1,1) = 2808.326079; // fy
  K.at<double>(0,2) = 969.285772;       // cx
  K.at<double>(1,2) = 624.049972;       // cy
  K.at<double>(2,2) = 1.0;

  // 2. initialize Rellis-3d lidar-camera extrinsics matrix
  Eigen::Quaternionf q(-0.50507811, 0.51206185, 0.49024953, -0.49228464); // (w, x, y, z)
  Eigen::Vector3f t(-0.13165462, 0.03870398, -0.17253834);

  Eigen::Matrix4f RT = Eigen::Matrix4f::Identity();
  RT.block<3, 3>(0, 0) = q.toRotationMatrix();
  RT.block<3, 1>(0, 3) = t;
  
  // acording to the Rellis-3d dataset, we need the inverse transform from base to camera
  Eigen::Matrix4f T_cam_base = RT.inverse();

  // ensure GridMap has the visual cost layer, initialized to 0.0 (absolute flat/safe)
  if(!map_.exists(layers::Visual::COST)) {
    map_.addLayer(layers::Visual::COST, 0.0); 
  }

  // 3. 遍历当前帧雷达点云 (已转到 Base 系)
  for (const auto& pt : cloud_base->points) {
    
    if (std::abs(pt.x) < 0.1 && std::abs(pt.y) <0.1) continue;

    // 将 LiDAR 系下的点转换到相机坐标系  
    Eigen::Vector4f p_l(pt.x, pt.y, pt.z, 1.0);
    Eigen::Vector4f p_c = T_cam_base * p_l;

    // 剔除相机后方的点
    if (p_c.z() <= 0.0) continue;

    // 利用 RELLIS-3D 相机内参投影到像素平面:
    double u = (K.at<double>(0,0) * p_c.x() + K.at<double>(0,2) * p_c.z()) / p_c.z();
    double v = (K.at<double>(1,1) * p_c.y() + K.at<double>(1,2) * p_c.z()) / p_c.z();

    int px = std::round(u);
    int py = std::round(v);

    // 4. 检查是否在图像有效视野范围内
    if (px >= 0 && px < visual_cost_img.cols && py >= 0 && py < visual_cost_img.rows) {
      
      // 采样对应的视觉代价
      float cost = visual_cost_img.at<float>(py, px);

      // 4. 将代价记录到对应的 2.5D GridMap 栅格中
      grid_map::Position position(pt.x, pt.y);
      grid_map::Index index;
      
      // 只有当点落在当前地图边界内时才更新
      if (map_.getIndex(position, index)) {
        // 保守更新策略：如果同一个栅格落入多个点，取最大的 visual_cost
        float current_cost = map_.at(layers::Visual::COST, index);
        if (std::isnan(current_cost) || cost > current_cost) {
            map_.at(layers::Visual::COST, index) = cost;
        }
      }
    }
  }
}
// ============================================================================
} // namespace lesta
