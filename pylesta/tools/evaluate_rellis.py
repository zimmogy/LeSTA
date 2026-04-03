#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import DataLoader

# 假设你的代码库结构如下，请根据实际路径调整导入
from lesta.core.datasets.pcd_dataset.dataset import PCDDataset
from lesta.core.models.mlp_classifier import MLPClassifier
from utils.param import yaml

# 导入你截图中的评估类
from utils.pytorch.evaluation import BinaryClassificationMetrics

def evaluate_model(config_path, model_ckpt_path, val_pcd_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # ==========================================
    # 1. 加载配置与模型
    # ==========================================
    cfg = yaml.load(config_path)
    
    # 实例化你修改过输入维度（例如 8 通道）的模型
    model = MLPClassifier(cfg=cfg['model'])
    
    # 加载训练好的权重
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    # 兼容有些保存了 optimizer 状态的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print(f"✅ 模型加载成功: {model_ckpt_path}")

    # ==========================================
    # 2. 构建验证集 DataLoader
    # ==========================================
    # 临时修改 config 中的数据路径为你的评估集路径
    val_cfg = cfg['DATASET'].copy()
    val_cfg['training_data'] = val_pcd_path # 借用原有的加载逻辑
    
    print(f"📦 正在加载验证集 PCD: {val_pcd_path}")
    val_dataset = PCDDataset(val_cfg)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    # ==========================================
    # 3. 运行推理与收集数据
    # ==========================================
    all_preds = []
    all_targets = []

    print("🧠 开始前向推理计算...")
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["feats"].to(device)    # [Batch, Channels]
            labels = batch["label"].to(device)   # [Batch, 1]

            # 网络输出通常是经过 Sigmoid 的概率 (0.0 ~ 1.0)
            preds_prob = model(feats)

            all_preds.append(preds_prob.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # 拼接所有的 batch
    preds_np = np.vstack(all_preds).squeeze()
    targets_np = np.vstack(all_targets).squeeze()

    # ==========================================
    # 4. 数据清洗：核心的物理掩码 (Masking)
    # ==========================================
    # 你的验证集中包含 -1 (Void/Sky/未知) 的区域，这些绝对不能参与指标计算
    valid_mask = (targets_np != -1.0)
    
    clean_targets = targets_np[valid_mask]
    clean_preds_prob = preds_np[valid_mask]
    
    # 将概率二值化为硬标签 (阈值设定为 0.5)
    clean_preds_hard = (clean_preds_prob >= 0.5).astype(np.float32)

    print(f"🧹 清洗完毕! 有效评估网格数: {len(clean_targets)} (剔除了 {len(targets_np) - len(clean_targets)} 个未知网格)")

    # ==========================================
    # 5. 计算并打印评估指标
    # ==========================================
    # 将 numpy array 转回 Tensor，以适配你截图中的 BinaryClassificationMetrics
    preds_tensor = torch.from_numpy(clean_preds_hard)
    targets_tensor = torch.from_numpy(clean_targets)

    # 实例化并计算指标
    metrics_calculator = BinaryClassificationMetrics()
    metrics = metrics_calculator.compute(preds_tensor, targets_tensor)

    print("\n" + "="*40)
    print("📈 越野环境通过性 (Traversability) 评估报告")
    print("="*40)
    print(f"Accuracy (整体准确率): \t{metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (精确率): \t{metrics['precision'] * 100:.2f}%")
    print(f"Recall (召回率): \t{metrics['recall'] * 100:.2f}%")
    print(f"F1-Score (F1分数): \t{metrics['f1_score'] * 100:.2f}%")
    print(f"IoU (交并比): \t\t{metrics['iou'] * 100:.2f}%")
    print("="*40)
    
    return metrics

if __name__ == "__main__":
    # 配置你的路径
    CONFIG_YAML = "../configs/lesta.yaml"
    BEST_MODEL_CKPT = "../checkpoints/epoch_best.pt"
    # 上一步生成的包含全局硬真值的评估 PCD
    EVAL_PCD = "/home/whr/Data/dataset/00000/val_set_with_gt/rellis_00000_global_eval.pcd" 
    
    evaluate_model(CONFIG_YAML, BEST_MODEL_CKPT, EVAL_PCD)