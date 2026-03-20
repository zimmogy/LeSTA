#!/usr/bin/env python3
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pylesta')))
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
import os
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
        self.sub_img = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)
        self.pub_cost = rospy.Publisher("/visual_cost_map", Image, queue_size=1)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        # BGR转RGB并送入 GPU
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 高速推理
        with torch.no_grad():
            cost_map_tensor = self.model(img_tensor)
            
        cost_map_np = cost_map_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # 封装为 ROS 32位单通道浮点数格式，维持严格的时间戳同步
        try:
            cost_msg = self.bridge.cv2_to_imgmsg(cost_map_np, encoding="32FC1")
            cost_msg.header = msg.header 
            self.pub_cost.publish(cost_msg)
        except Exception as e:
            rospy.logerr(f"Publishing Error: {e}")

if __name__ == '__main__':
    try:
        node = VisualCostInferNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass