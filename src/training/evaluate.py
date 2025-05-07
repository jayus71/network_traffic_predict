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
    
    # 判断是否为集成模型（检查第一个批次的数据类型）
    sample_batch = next(iter(test_loader))
    is_ensemble = isinstance(sample_batch[0], list)
    
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到设备
            if is_ensemble:
                # 集成模型的情况
                data = [d.to(device) for d in data]
            else:
                # 普通模型的情况
                data = data.to(device)
                
            target = target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # 收集预测值和目标值
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算平均测试损失
    test_loss /= len(test_loader)
    
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 反归一化数据（如果提供了缩放器）
    if scaler:
        # 检查是否为多步预测
        if len(all_preds.shape) > 1 and all_preds.shape[1] > 1:
            # 多步预测，需要对每个步长单独反归一化
            all_preds_reshaped = all_preds.reshape(-1, 1)
            all_targets_reshaped = all_targets.reshape(-1, 1)
            
            all_preds_reshaped = scaler.inverse_transform(all_preds_reshaped)
            all_targets_reshaped = scaler.inverse_transform(all_targets_reshaped)
            
            all_preds = all_preds_reshaped.reshape(all_preds.shape)
            all_targets = all_targets_reshaped.reshape(all_targets.shape)
        else:
            # 单步预测
            all_preds = scaler.inverse_transform(all_preds)
            all_targets = scaler.inverse_transform(all_targets)
    
    # 计算评估指标
    # 如果是多步预测，先平均每个步长的指标
    if len(all_preds.shape) > 1 and all_preds.shape[1] > 1:
        # 计算每个步长的指标
        step_metrics = []
        for step in range(all_preds.shape[1]):
            step_preds = all_preds[:, step]
            step_targets = all_targets[:, step]
            
            mae = mean_absolute_error(step_targets, step_preds)
            rmse = np.sqrt(mean_squared_error(step_targets, step_preds))
            r2 = r2_score(step_targets, step_preds)
            mape = np.mean(np.abs((step_targets - step_preds) / (step_targets + 1e-8))) * 100
            
            step_metrics.append({
                'step': step + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            })
        
        # 计算所有步长的平均指标
        mae = np.mean([m['mae'] for m in step_metrics])
        rmse = np.mean([m['rmse'] for m in step_metrics])
        r2 = np.mean([m['r2'] for m in step_metrics])
        mape = np.mean([m['mape'] for m in step_metrics])
        
        # 返回总体指标和每个步长的指标
        metrics = {
            'test_loss': test_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'step_metrics': step_metrics
        }
    else:
        # 单步预测
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