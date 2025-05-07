import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_data, create_dataloaders
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import (
    create_time_features, 
    create_lag_features,
    create_window_features, 
    create_sequences
)
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNNModel
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='网络流量预测训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='数据文件路径')
    parser.add_argument('--model_type', type=str, default=None, 
                        choices=['lstm', 'gru', 'cnn'], help='模型类型(覆盖配置文件)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录(覆盖配置文件)')
    parser.add_argument('--cell_name', type=str, default=None, 
                        help='基站名称(仅combined_data.csv)')
    parser.add_argument('--city', type=str, default=None, 
                        help='城市名称(仅dataA_fill.csv)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.model_type:
        config['model']['type'] = args.model_type
    
    if args.output_dir:
        config['training']['model_dir'] = args.output_dir
    
    # 创建输出目录
    os.makedirs(config['training']['model_dir'], exist_ok=True)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    df = load_data(args.data_path)
    print(f"从 {args.data_path} 加载数据，形状: {df.shape}")
    
    # 根据数据类型处理
    if 'combined_data.csv' in args.data_path:
        # 基站数据
        if args.cell_name:
            df = df[df['CellName'] == args.cell_name]
            print(f"筛选基站 {args.cell_name} 的数据，形状: {df.shape}")
        
        # 数据预处理
        df = preprocess_data(df)
        
        # 排序数据
        df = df.sort_values(by=['Date', 'Hour'])
        print(f"前五个数据:\n{df.head()}")
        # 特征工程
        target_col = 'Traffic'
        
        # 创建时间特征
        df['Time'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
        df = create_time_features(df, 'Time')
        
        # 对于单一基站，创建滞后特征和窗口特征
        if args.cell_name:
            df = create_lag_features(df, target_col, config['features']['lag_periods'])
            df = create_window_features(df, target_col, config['features']['window_sizes'])
    
    elif 'fill.csv' in args.data_path:
        # 城市数据
        if args.city:
            df = df[df['City'] == args.city]
            print(f"筛选城市 {args.city} 的数据，形状: {df.shape}")
        
        # 数据预处理
        df = preprocess_data(df)
        
        # 排序数据
        df = df.sort_values(by=['Time'])
        
        # 特征工程
        target_col = 'Value'
        
        # 创建时间特征
        df = create_time_features(df, 'Time')
        
        # 创建滞后特征和窗口特征
        df = create_lag_features(df, target_col, config['features']['lag_periods'])
        df = create_window_features(df, target_col, config['features']['window_sizes'])
    
    else:
        raise ValueError(f"不支持的数据文件类型: {args.data_path}")
    
    # 删除含有NaN的行
    df = df.dropna()
    print(f"数据预处理和特征工程后，形状: {df.shape}")
    
    # 准备特征和目标变量
    # 排除非数值类型的列，如'CellName'，以及不需要的时间列
    features = [col for col in df.columns if col != 'Time' and col != 'Date' 
                and col != 'CellName' and df[col].dtype != 'object']
    print(f"使用特征: {features}")
        
    X = df[features].values
    y = df[target_col].values
    
    # 数据缩放
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    
    # 创建序列数据
    seq_length = config['features']['sequence_length']
    X_seq, y_seq = create_sequences(np.column_stack([X_scaled, y_scaled]), seq_length)
    

    
    print(f"序列数据形状: X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
    
    # 划分训练集和测试集
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    
    # 先划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, shuffle=False, random_state=config['data']['random_state']
    )
    
    # 再从测试集中划分验证集
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=val_size, shuffle=False, random_state=config['data']['random_state']
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        batch_size=config['training']['batch_size']
    )
    
    # 创建模型
    model_type = config['model']['type']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    
    input_size = X_train.shape[2]  # 特征维度
    
    if model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        model_name = 'lstm_model.pt'
    elif model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        model_name = 'gru_model.pt'
    elif model_type == 'cnn':
        model = CNNModel(
            input_size=input_size,
            seq_length=seq_length,
            filters=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        model_name = 'cnn_model.pt'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print(f"创建 {model_type.upper()} 模型")
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        model_dir=config['training']['model_dir'],
        model_name=model_name
    )
    
    # 评估模型
    metrics = evaluate_model(model, test_loader, device, y_scaler)
    # 在main.py中的评估代码之后添加
    # 获取预测结果用于可视化
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 收集预测值和目标值
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 反归一化
    if y_scaler:
        all_preds = y_scaler.inverse_transform(all_preds)
        all_targets = y_scaler.inverse_transform(all_targets)

    # 使用visualize.py中的函数进行可视化
    from src.visualization.visualize import plot_prediction_results, plot_time_series, plot_loss_history

    # 预测结果与真实值对比图
    viz_save_path = os.path.join(config['evaluation']['visualization_dir'], f"{model_type}_predictions.png")
    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
    plot_prediction_results(
        all_targets.flatten(), 
        all_preds.flatten(), 
        title=f"{model_type.upper()} 模型预测结果",
        save_path=viz_save_path
    )

    # 时间序列图 - 仅展示部分数据点以提高可读性
    sample_size = min(500, len(all_preds))
    indices = np.arange(len(all_preds))[-sample_size:]
    plt.figure(figsize=(12, 6))
    plt.plot(indices, all_targets.flatten()[-sample_size:], label='真实值', alpha=0.7)
    plt.plot(indices, all_preds.flatten()[-sample_size:], label='预测值', alpha=0.7, linestyle='--')
    plt.title(f'{model_type.upper()} 模型时间序列预测')
    plt.xlabel('样本索引')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(config['evaluation']['visualization_dir'], f"{model_type}_time_series.png"))
    plt.show()

    # 绘制训练和验证损失曲线
    loss_save_path = os.path.join(config['evaluation']['visualization_dir'], f"{model_type}_loss.png")
    plot_loss_history(history, save_path=loss_save_path)

    print(f"所有可视化结果已保存到 {config['evaluation']['visualization_dir']} 目录")
    print("\n评估结果:")
    print(f"测试损失: {metrics['test_loss']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    # 保存评估结果
    results_file = os.path.join(config['training']['model_dir'], 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"数据文件: {args.data_path}\n")
        if 'combined_data.csv' in args.data_path and args.cell_name:
            f.write(f"基站: {args.cell_name}\n")
        elif 'dataA_fill.csv' in args.data_path and args.city:
            f.write(f"城市: {args.city}\n")
        
        f.write(f"模型类型: {model_type}\n")
        f.write(f"隐藏层大小: {hidden_size}\n")
        f.write(f"层数: {num_layers}\n")
        f.write(f"序列长度: {seq_length}\n")
        f.write("\n评估结果:\n")
        f.write(f"测试损失: {metrics['test_loss']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
    
    print(f"评估结果已保存到 {results_file}")

if __name__ == "__main__":
    main()