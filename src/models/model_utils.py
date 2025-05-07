import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def save_model(model, path):
    """
    保存模型
    
    参数:
        model: PyTorch模型
        path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")

def load_model(model, path, device):
    """
    加载模型
    
    参数:
        model: PyTorch模型实例
        path: 模型路径
        device: 设备 (CPU/GPU)
        
    返回:
        加载好参数的模型
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件: {path}")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"模型已从 {path} 加载")
    return model

def evaluate_and_visualize(model, test_loader, device, scaler=None, save_path=None):
    """
    评估模型并可视化结果
    
    参数:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 评估设备 (CPU/GPU)
        scaler: 用于反归一化的缩放器
        save_path: 图像保存路径
        
    返回:
        评估指标字典和预测结果
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 收集预测值和目标值
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 如果有缩放器，进行反归一化
    if scaler:
        all_preds = scaler.inverse_transform(all_preds)
        all_targets = scaler.inverse_transform(all_targets)
    
    # 计算评估指标
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # 绘制预测值和实际值
    x = np.arange(len(all_targets))
    plt.plot(x, all_targets, label='实际值', alpha=0.7)
    plt.plot(x, all_preds, label='预测值', alpha=0.7)
    
    # 添加标题和标签
    plt.title(f'预测结果 (MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('流量值')
    plt.legend()
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return metrics, (all_preds, all_targets)