import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_series(df, time_col, value_col, title=None, figsize=(12, 6), save_path=None):
    """
    绘制时间序列图
    
    参数:
        df: DataFrame对象
        time_col: 时间列名
        value_col: 值列名
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 绘制时间序列
    plt.plot(df[time_col], df[value_col], marker='.', linestyle='-', alpha=0.8)
    
    # 设置标题和标签
    if title:
        plt.title(title)
    else:
        plt.title(f'{value_col}随{time_col}的变化')
    
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    plt.show()

def plot_multiple_cells(df, cell_names, date_col='Date', hour_col='Hour', value_col='Traffic', figsize=(14, 8), save_path=None):
    """
    绘制多个基站的流量对比图
    
    参数:
        df: DataFrame对象
        cell_names: 基站名称列表
        date_col: 日期列名
        hour_col: 小时列名
        value_col: 值列名
        figsize: 图表大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    for cell_name in cell_names:
        # 筛选数据
        cell_df = df[df['CellName'] == cell_name].copy()
        
        # 创建完整时间列
        cell_df['Datetime'] = pd.to_datetime(cell_df[date_col]) + pd.to_timedelta(cell_df[hour_col], unit='h')
        
        # 绘制时间序列
        plt.plot(cell_df['Datetime'], cell_df[value_col], marker='.', linestyle='-', alpha=0.8, label=cell_name)
    
    # 设置标题和标签
    plt.title('多基站流量对比')
    plt.xlabel('时间')
    plt.ylabel(value_col)
    
    # 添加图例
    plt.legend()
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    plt.show()

def plot_prediction_results(y_true, y_pred, title='预测结果对比', figsize=(12, 6), save_path=None):
    """
    绘制预测结果与真实值对比图
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 计算评估指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    r2 = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    
    # 绘制散点图和对角线
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 获取坐标轴范围
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    padding = (max_val - min_val) * 0.1
    
    # 设置相同的坐标轴范围
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    
    # 绘制对角线(理想情况下预测值应在此线上)
    plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--')
    
    # 设置标题和标签
    plt.title(f'{title} (MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f})')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    plt.show()

def plot_loss_history(history, figsize=(10, 6), save_path=None):
    """
    绘制训练过程中的损失曲线
    
    参数:
        history: 训练历史字典，包含'train_loss'和'val_loss'
        figsize: 图表大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 绘制训练损失和验证损失
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    
    # 设置标题和标签
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    
    # 添加图例
    plt.legend()
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    plt.show()