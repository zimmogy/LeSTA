import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pylesta.lesta.core.models.vision_cost_net import MobileNetV3CostNet
from pylesta.lesta.core.loss_fns.masked_loss import MaskedSmoothL1Loss
from pylesta.lesta.core.datasets.pcd_dataset.vision_dataset.dataset import RellisVisionDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start training on: {device}")

    # 1. 实例化数据集
    train_dataset = RellisVisionDataset(
        image_dir='/home/whr/Data/dataset/00000/rellis/train/images',       # 【修改这里】
        mask_dir='/home/whr/Data/dataset/00000/rellis/train/sparse_masks'   # 【修改这里】
    )
    # 增加 num_workers 提升读取速度
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # 2. 实例化 MobileNetV3 骨干网络
    model = MobileNetV3CostNet(pretrained=True).to(device)
    criterion = MaskedSmoothL1Loss(ignore_index=-1.0)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    num_epochs = 30
    
    # 3. 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            
            # 跳过全图无轨迹的无效批次
            if loss.item() == 0.0:
                continue
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        print(f"--> Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"mobilenetv3_cost_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()