import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .lstm_model import LSTMModel
from .cnn_model import CNNModel
from .transformer_model import TransformerModel
from .ensemble_model import EnsembleModel

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
            if isinstance(data, list):
                # 集成模型的情况
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
                
            target = target.to(device)
            
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

def create_model(model_type, input_size, seq_length=24, output_size=1, config=None):
    """
    创建模型
    
    参数:
        model_type: 模型类型 ('lstm', 'cnn', 'transformer', 或 'ensemble')
        input_size: 输入特征维度
        seq_length: 输入序列长度
        output_size: 输出维度
        config: 模型配置
        
    返回:
        创建的模型实例
    """
    if config is None:
        config = {}
    
    if model_type == 'lstm':
        return LSTMModel(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            output_size=output_size
        )
    elif model_type == 'cnn':
        return CNNModel(
            input_size=input_size,
            seq_length=seq_length,
            filters=config.get('filters', 64),
            kernel_size=config.get('kernel_size', 3),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.2),
            output_size=output_size
        )
    elif model_type == 'transformer':
        return TransformerModel(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 2),
            nhead=config.get('nhead', 4),
            dropout=config.get('dropout', 0.2),
            output_size=output_size
        )
    elif model_type == 'ensemble':
        # 获取集成模型配置
        ensemble_config = config.get('ensemble', {})
        
        # 获取降采样比例并计算实际序列长度
        ratios = config.get('downsample', {}).get('ratios', [1.0, 0.5, 0.25])
        seq_lengths = [max(int(seq_length * ratio), 1) for ratio in ratios]
        seq_lengths = sorted(seq_lengths, reverse=True)  # 降序排列
        
        # 创建集成模型
        return EnsembleModel(
            input_size=input_size,
            seq_lengths=seq_lengths,
            hidden_size=config.get('hidden_size', 128),
            cnn_config=ensemble_config.get('cnn'),
            lstm_config=ensemble_config.get('lstm'),
            transformer_config=ensemble_config.get('transformer'),
            dropout=config.get('dropout', 0.2),
            output_size=output_size
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")