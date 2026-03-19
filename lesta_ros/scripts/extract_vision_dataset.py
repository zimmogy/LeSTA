"""
# 该脚本用于从 RELLIS-3D 数据集中提取视觉数据集，生成图像和对应的稀疏轨迹掩码。
# 主要功能：
# 1. 订阅 RELLIS-3D 的图像和 IMU数据。
# 2. 对每个图像，等待设定的延迟时间（默认 3 秒），以便收集足够长的“未来”轨迹。
# 3. 使用 TF 监听器获取未来轨迹点在当前图像坐标系中的位置，并根据轨迹点的分布计算软标签。
# 4. 将图像和对应的稀疏轨迹掩码保存到指定目录，供后续训练使用。
"""
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import math
from collections import deque
import tf2_ros
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge

class VisionDatasetExtractor:
    def __init__(self):
        rospy.init_node('vision_dataset_extractor', anonymous=True)
        
        # 配置参数：延迟 3.0 秒处理图像，以便收集足够长的“未来”轨迹
        self.delay_time = rospy.get_param('~delay_time', 3.0) 
        self.output_dir = rospy.get_param('~output_dir', '/tmp/rellis_vision_dataset/')
        self.image_dir = os.path.join(self.output_dir, 'train/images')
        self.mask_dir = os.path.join(self.output_dir, 'train/sparse_masks')
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        # 载入 RELLIS-3D 真实的相机内参矩阵 (fx, fy, cx, cy) 
        self.K = np.array([
            [2813.643275, 0.0,         969.285772],
            [0.0,         2808.326079, 624.049972],
            [0.0,         0.0,         1.0       ]
        ])
        
        # TF 监听器：缓存 15 秒的历史 TF 树
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        
        # 数据双端队列
        self.image_queue = deque()
        self.imu_queue = deque()
        
        # IMU 软标签参数
        self.lambda_decay = 0.5
        self.min_soft_label = 0.2
        self.imu_window_size = 0.5 
        
        # 订阅 RELLIS-3D 话题
        rospy.Subscriber('/pylon_camera_node/image_raw', Image, self.image_callback)
        rospy.Subscriber('/vectornav/IMU', Imu, self.imu_callback)
        
        # 定时器：以 10Hz 的频率处理积压的缓冲图像
        rospy.Timer(rospy.Duration(0.1), self.process_queue)
        
        rospy.loginfo(f"Vision Dataset Extractor started. Output: {self.output_dir}")

    def imu_callback(self, msg):
        self.imu_queue.append(msg)
        # 维护 IMU 队列长度，丢弃过于陈旧的数据
        while self.imu_queue and (msg.header.stamp - self.imu_queue[0].header.stamp).to_sec() > 10.0:
            self.imu_queue.popleft()

    def image_callback(self, msg):
        self.image_queue.append(msg)

    def calculate_soft_label(self, target_time):
        valid_imus = [m.linear_acceleration.z for m in self.imu_queue 
                      if 0 <= (target_time - m.header.stamp).to_sec() <= self.imu_window_size]
        
        if not valid_imus:
            return 1.0
            
        mean_z = sum(valid_imus) / len(valid_imus)
        variance_z = sum((z - mean_z) ** 2 for z in valid_imus) / len(valid_imus)
        
        score = math.exp(-self.lambda_decay * variance_z)
        return max(self.min_soft_label, float(score))

    def process_queue(self, event):
        now = rospy.Time.now()
        
        while self.image_queue:
            img_msg = self.image_queue[0]
            img_time = img_msg.header.stamp
            
            # 检查是否已达到设定的延迟时间
            if (now - img_time).to_sec() < self.delay_time:
                break
                
            self.image_queue.popleft()
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            except Exception:
                continue
                
            image_shape = cv_image.shape[:2]
            mask = np.full(image_shape, -1.0, dtype=np.float32) 
            has_trajectory = False
            
            # 获取未来轨迹点
            for dt in np.arange(0.0, self.delay_time, 0.1):
                future_time = img_time + rospy.Duration(dt)
                
                try:
                    # 使用提取到的真实坐标系名称：计算未来 os1_cloud_node (LiDAR基准) 在当前 pylon_camera_node (相机) 中的位置
                    trans = self.tf_buffer.lookup_transform_full(
                        target_frame='pylon_camera_node', 
                        target_time=img_time,
                        source_frame='os1_cloud_node',
                        source_time=future_time,
                        fixed_frame='odom',
                        timeout=rospy.Duration(0.05)
                    )
                    
                    x = trans.transform.translation.x
                    y = trans.transform.translation.y
                    z = trans.transform.translation.z
                    
                    if z <= 0.1: # 剔除相机后方的点
                        continue
                        
                    # 利用真实内参进行透视投影
                    u = int((self.K[0,0] * x) / z + self.K[0,2])
                    v = int((self.K[1,1] * y) / z + self.K[1,2])
                    
                    if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                        cost = self.calculate_soft_label(future_time)
                        cv2.circle(mask, (u, v), radius=25, color=float(cost), thickness=-1)
                        has_trajectory = True
                        
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue
            
            if not has_trajectory:
                continue
                
            timestamp_str = f"{img_time.to_sec():.6f}"
            cv2.imwrite(os.path.join(self.image_dir, f"{timestamp_str}.jpg"), cv_image)
            np.save(os.path.join(self.mask_dir, f"{timestamp_str}.npy"), mask)
            rospy.loginfo(f"Extracted image-mask pair: {timestamp_str}")

if __name__ == '__main__':
    try:
        VisionDatasetExtractor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass