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

def create_sequences(data, seq_length):
    """创建序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  # 假设目标变量在最后一列
    return np.array(X), np.array(y)