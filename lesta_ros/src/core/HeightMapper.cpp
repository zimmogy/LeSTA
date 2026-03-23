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
// [new]
#include "lesta/types/layer_definitions.h"

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
void HeightMapper::integrateVisualCost(const pcl::PointCloud<Laser>::Ptr& cloud_map,
                                       const cv::Mat& visual_cost_img,
                                       const Eigen::Matrix4f& T_map_to_sensor) {
  if (cloud_map->empty()) return;

  // 1. RELLIS-3D 相机内参保持不变
  cv::Mat K_intrinsic = cv::Mat::zeros(3,3, CV_64F);
  K_intrinsic.at<double>(0,0) = 2813.643275; 
  K_intrinsic.at<double>(1,1) = 2808.326079; 
  K_intrinsic.at<double>(0,2) = 969.285772;       
  K_intrinsic.at<double>(1,2) = 624.049972;       
  K_intrinsic.at<double>(2,2) = 1.0;

  // 2. RELLIS-3D 雷达-相机外参保持不变
  Eigen::Quaternionf q(-0.50507811, 0.51206185, 0.49024953, -0.49228464);
  Eigen::Vector3f t(-0.13165462, 0.03870398, -0.17253834);
  Eigen::Matrix4f RT = Eigen::Matrix4f::Identity();
  RT.block<3, 3>(0, 0) = q.toRotationMatrix();
  RT.block<3, 1>(0, 3) = t;
  Eigen::Matrix4f T_lidar_to_cam = RT.inverse();

  // 确保地图包含 cost 图层
  if(!map_.exists(lesta::layers::Visual::COST)) {
    map_.addLayer(lesta::layers::Visual::COST, std::nanf("")); // 默认设为 NaN 更好，避免误判为 0(平坦)
  }

  // 3. 遍历当前帧点云
  for (const auto& pt_map : cloud_map->points) {
    
    // 【修改点A】先将 Map 系点转换回 LiDAR 系，才能给相机用
    Eigen::Vector4f p_m(pt_map.x, pt_map.y, pt_map.z, 1.0);
    Eigen::Vector4f p_sensor = T_map_to_sensor * p_m;

    // 在 sensor 系下剔除雷达近处死角的点
    if (std::abs(p_sensor.x()) < 0.1 && std::abs(p_sensor.y()) < 0.1) continue;

    // 将 LiDAR 系下的点转换到相机坐标系  
    Eigen::Vector4f p_c = T_lidar_to_cam * p_sensor;

    // 剔除相机后方的点
    if (p_c.z() <= 0.0) continue;

    // 投影到像素平面
    double u = (K_intrinsic.at<double>(0,0) * p_c.x() + K_intrinsic.at<double>(0,2) * p_c.z()) / p_c.z();
    double v = (K_intrinsic.at<double>(1,1) * p_c.y() + K_intrinsic.at<double>(1,2) * p_c.z()) / p_c.z();

    int px = std::round(u);
    int py = std::round(v);

    // 4. 检查是否在图像视野内并更新地图
    if (px >= 0 && px < visual_cost_img.cols && py >= 0 && py < visual_cost_img.rows) {
      float cost = visual_cost_img.at<float>(py, px);

      // 【修改点B】写回栅格时，必须使用原始的 Map 系坐标 pt_map !
      grid_map::Position position(pt_map.x, pt_map.y);
      grid_map::Index index;
      
      if (map_.getIndex(position, index)) {
        float current_cost = map_.at(lesta::layers::Visual::COST, index);
        // 保守策略：同一个栅格被多个点击中时，保留危险系数 (cost) 最高的
        if (std::isnan(current_cost) || cost > current_cost) {
            map_.at(lesta::layers::Visual::COST, index) = cost;
        }
      }
    }
  }
}
// ============================================================================
} // namespace lesta
