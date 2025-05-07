import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_loader, device, scaler=None):
    """
    评估模型
    
    参数:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 评估设备 (CPU/GPU)
        scaler: 用于反归一化的缩放器
        
    返回:
        评估指标字典
    """
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # 收集预测值和目标值
            if scaler:
                # 转换到CPU并转换为NumPy数组
                pred_np = output.cpu().numpy()
                target_np = target.cpu().numpy()
                
                # 反归一化
                pred_np = scaler.inverse_transform(pred_np)
                target_np = scaler.inverse_transform(target_np)
                
                all_preds.extend(pred_np)
                all_targets.extend(target_np)
            else:
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    # 计算平均测试损失
    test_loss /= len(test_loader)
    
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100
    
    metrics = {
        'test_loss': test_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    return metrics