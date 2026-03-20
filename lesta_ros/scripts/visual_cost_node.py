#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pylesta')))
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lesta.core.models.vision_cost_net import MobileNetV3CostNet

class VisualCostInferNode:
    def __init__(self):
        rospy.init_node('visual_cost_mobilenet_node', anonymous=True)
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载定制的 MobileNetV3 模型
        weight_path = rospy.get_param('~model_path', 'mobilenetv3_cost_epoch_30.pth')
        self.model = MobileNetV3CostNet(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        
        rospy.loginfo(f"MobileNetV3 Cost Model loaded from {weight_path}")
        
        # 2. 订阅 RELLIS-3D 图像与发布 32FC1 代价图
        camera_topic = rospy.get_param('~camera_topic', '/pylon_camera_node/image_raw')
        
        # 【修改 3】：扩大队列并增加 buff_size，防止 Socket 阻塞导致大幅丢帧饿死同步器
        self.sub_img = rospy.Subscriber(camera_topic, Image, self.image_callback, 
                                        queue_size=5, buff_size=2**24)
        self.pub_cost = rospy.Publisher("/visual_cost_map", Image, queue_size=5)

    def image_callback(self, msg):
        # 【修改 2】：把静默退出改为报错退出，查清数据到底有没有进来
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(1.0, f"[VisualCostNode] imgmsg_to_cv2 转换失败! 检查图像编码: {e}")
            return

        # BGR转RGB并送入 GPU
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 高速推理
        with torch.no_grad():
            cost_map_tensor = self.model(img_tensor)
            
        # 【修改 1】：使用 .squeeze() 去除所有的 1 维 (包括 batch 和 channel)，确保输出是纯正的 2D 数组 [H, W]
        cost_map_np = cost_map_tensor.squeeze().cpu().numpy().astype(np.float32)

        # 封装为 ROS 32位单通道浮点数格式，维持严格的时间戳同步
        try:
            # 现在 cost_map_np 是纯 2D [H, W]，cv2_to_imgmsg 不会再崩溃了
            cost_msg = self.bridge.cv2_to_imgmsg(cost_map_np, encoding="32FC1")
            
            # 【完美保留】：直接复用原图 header，确保 C++ 端 ApproximateTime 同步成功
            cost_msg.header = msg.header 
            self.pub_cost.publish(cost_msg)
            
        except Exception as e:
            rospy.logerr(f"[VisualCostNode] 代价图发布失败: {e}")

if __name__ == '__main__':
    try:
        node = VisualCostInferNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass