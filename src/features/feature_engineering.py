import pandas as pd
import numpy as np

def create_time_features(df, time_col):
    """创建时间特征"""
    df = df.copy()
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    df['day'] = pd.to_datetime(df[time_col]).dt.day
    df['weekday'] = pd.to_datetime(df[time_col]).dt.weekday
    df['month'] = pd.to_datetime(df[time_col]).dt.month
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def create_lag_features(df, target_col, lag_periods=[1, 2, 3, 24]):
    """创建滞后特征"""
    df = df.copy()
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_window_features(df, target_col, windows=[3, 6, 24]):
    """创建滑动窗口特征"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_mean_{window}'] = df[target_col].rolling(window).mean().shift(1)
        df[f'{target_col}_std_{window}'] = df[target_col].rolling(window).std().shift(1)
        df[f'{target_col}_max_{window}'] = df[target_col].rolling(window).max().shift(1)
        df[f'{target_col}_min_{window}'] = df[target_col].rolling(window).min().shift(1)
    return df

def create_sequences(data, input_length, output_length=1):
    """
    创建序列数据，支持多步预测
    
    参数:
        data: 输入数据，最后一列为目标变量
        input_length: 输入序列长度
        output_length: 输出序列长度（预测步长）
        
    返回:
        X: 输入序列，shape [样本数, 输入序列长度, 特征维度]
        y: 目标值，shape [样本数, 输出序列长度]
    """
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        # 输入序列
        X.append(data[i:i+input_length])
        
        # 目标序列（单步或多步预测）
        if output_length == 1:
            y.append(data[i+input_length, -1])  # 单步预测，目标变量在最后一列
        else:
            y.append(data[i+input_length:i+input_length+output_length, -1])  # 多步预测
            
    return np.array(X), np.array(y)