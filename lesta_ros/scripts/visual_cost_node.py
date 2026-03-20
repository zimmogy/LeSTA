#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pylesta')))
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image

from lesta.core.models.vision_cost_net import MobileNetV3CostNet

class VisualCostInferNode:
    def __init__(self):
        rospy.init_node('visual_cost_mobilenet_node', anonymous=True)
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
        # ==========================================
        # 1. 纯 Numpy 解析输入图像 (彻底抛弃 cv_bridge)
        # ==========================================
        try:
            # 直接读取 ROS Image 的字节流并转化为 numpy 数组
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            
            # RELLIS-3D 的前视相机通常是 bgr8 或 rgb8 编码
            if msg.encoding == 'bgr8':
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif msg.encoding == 'rgb8':
                img_rgb = img_np
            else:
                # 兼容其他未知格式，强行截取前3通道
                img_rgb = img_np[..., :3] 
        except Exception as e:
            rospy.logerr_throttle(1.0, f"[VisualCostNode] 内存解析图像失败: {e}")
            return

        # ==========================================
        # 2. PyTorch 高速推理
        # ==========================================
        # BGR转RGB并送入 GPU
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            cost_map_tensor = self.model(img_tensor)
            
        # 降维得到纯正的 2D 数组 [H, W]
        cost_map_np = cost_map_tensor.squeeze().cpu().numpy().astype(np.float32)

        # ==========================================
        # 3. 手动封装输出代价图 (彻底抛弃 cv_bridge)
        # ==========================================
        try:
            cost_msg = Image()
            cost_msg.header = msg.header 
            cost_msg.height = cost_map_np.shape[0]
            cost_msg.width = cost_map_np.shape[1]
            cost_msg.encoding = "32FC1"
            cost_msg.is_bigendian = 0
            cost_msg.step = cost_msg.width * 4 # 32位单精度浮点数，每个像素占 4 bytes
            cost_msg.data = cost_map_np.tobytes() # 直接转成二进制字节流
            
            self.pub_cost.publish(cost_msg)
        except Exception as e:
            rospy.logerr(f"[VisualCostNode] 代价图发布失败: {e}")
            
if __name__ == '__main__':
    try:
        node = VisualCostInferNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass