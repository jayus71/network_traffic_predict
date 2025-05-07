import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    1x1 CNN模型用于时间序列预测
    """
    def __init__(self, input_size, seq_length, filters=64, kernel_size=1, num_layers=3, dropout=0.2, output_size=1):
        """
        初始化1x1 CNN模型
        
        参数:
            input_size: 输入特征维度
            seq_length: 序列长度
            filters: 卷积核数量
            kernel_size: 卷积核大小
            num_layers: 卷积层数量
            dropout: Dropout比率
            output_size: 输出维度(预测步长)
        """
        super(CNNModel, self).__init__()
        
        self.output_size = output_size
        
        # 创建多层卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层卷积
        self.conv_layers.append(nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=0
        ))
        self.conv_layers.append(nn.ReLU())
        
        # 中间卷积层
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=0
            ))
            self.conv_layers.append(nn.ReLU())
        
        # 展平层后的全连接层
        # 计算卷积后的序列长度
        self.flattened_size = filters * (seq_length - num_layers * (kernel_size - 1))
        
        # 中间层
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 100),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层 - 单步或多步预测
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, shape [batch_size, seq_length, input_size]
            
        返回:
            输出张量, shape [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # 调整输入形状以适应Conv1d [batch_size, input_size, seq_length]
        x = x.permute(0, 2, 1)
        
        # 卷积层前向计算
        for layer in self.conv_layers:
            x = layer(x)
        
        # 共享特征提取
        features = self.fc1(x)
        
        if self.output_size == 1:
            # 单步预测
            out = self.fc2(features)
        else:
            # 多步预测
            outs = []
            for _ in range(self.output_size):
                step_out = self.fc2(features)
                outs.append(step_out)
            out = torch.cat(outs, dim=1)
        
        return out