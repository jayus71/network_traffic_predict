import os
import random
import numpy as np
import torch

def set_seed(seed=42, deterministic=True):
    """
    设置随机种子，确保实验可重现性
    
    参数:
        seed: 随机种子值
        deterministic: 是否使用确定性算法（可能会影响性能）
        
    返回:
        None
    """
    # 设置Python的random模块种子
    random.seed(seed)
    
    # 设置NumPy的随机种子
    np.random.seed(seed)
    
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    
    # 设置CUDA的随机种子（如果可用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CuDNN行为
    if deterministic:
        # 使用确定性算法，可能会影响性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 使用非确定性算法，性能更好
        torch.backends.cudnn.benchmark = True
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置随机种子: {seed}, 确定性模式: {deterministic}") 