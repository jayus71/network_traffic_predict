import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, 
                learning_rate=0.001, epochs=100, patience=10,
                model_dir='models', model_name='model.pt'):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备 (CPU/GPU)
        learning_rate: 学习率
        epochs: 训练轮数
        patience: 早停耐心值
        model_dir: 模型保存目录
        model_name: 模型文件名
        
    返回:
        训练好的模型和训练历史
    """
    # 确保模型目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=0.0001)
    
    # 早停设置
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # 将数据移动到设备
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # 将数据移动到设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                output = model(data)
                
                # 计算损失
                loss = criterion(output, target)
                
                # 累加损失
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
            print(f"Model saved to {os.path.join(model_dir, model_name)}")
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history