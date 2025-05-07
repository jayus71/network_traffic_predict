import numpy as np

def downsample_sequence(sequence, target_length):
    """
    对序列进行降采样到目标长度
    
    参数:
        sequence: 原始序列，形状为 [seq_length, feature_dim]
        target_length: 目标序列长度
        
    返回:
        降采样后的序列，形状为 [target_length, feature_dim]
    """
    original_length = sequence.shape[0]
    
    # 如果目标长度大于或等于原始长度，则不需要降采样
    if target_length >= original_length:
        return sequence
    
    # 计算采样间隔
    if target_length == 1:
        # 如果目标长度为1，则只取最后一个元素
        return sequence[-1:].reshape(1, -1)
    
    # 计算降采样索引
    indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    
    # 执行降采样
    downsampled = sequence[indices]
    
    return downsampled

def downsample_batch(batch, target_lengths):
    """
    对批量序列进行多种长度的降采样
    
    参数:
        batch: 批量序列，形状为 [batch_size, seq_length, feature_dim]
        target_lengths: 目标序列长度列表，例如 [24, 12, 6]
    
    返回:
        降采样后的序列列表，每个元素形状为 [batch_size, target_length, feature_dim]
    """
    batch_size, seq_length, feature_dim = batch.shape
    downsampled_batches = []
    
    # 原始序列也作为一种采样结果
    if seq_length in target_lengths:
        downsampled_batches.append(batch)
    
    # 对每个目标长度进行降采样
    for length in target_lengths:
        if length == seq_length:
            continue  # 原始序列已经添加
            
        result = np.zeros((batch_size, length, feature_dim))
        
        # 对批次中的每个序列进行降采样
        for i in range(batch_size):
            result[i] = downsample_sequence(batch[i], length)
            
        downsampled_batches.append(result)
    
    return downsampled_batches 