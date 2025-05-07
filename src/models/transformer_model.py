import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供序列位置信息
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    """
    Transformer模型用于时间序列预测
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, nhead=4, dropout=0.2, output_size=1):
        """
        初始化Transformer模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: Transformer编码器层数
            nhead: 多头注意力的头数
            dropout: Dropout比率
            output_size: 输出维度(预测步长)
        """
        super(TransformerModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 特征映射层
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, shape [batch_size, seq_length, input_size]
            
        返回:
            输出张量, shape [batch_size, output_size]
        """
        # 特征映射
        x = self.embedding(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        if self.output_size == 1:
            # 单步预测，只需要最后一个时间步的输出
            out = x[:, -1, :]
            out = self.fc(out)
        else:
            # 多步预测，使用最后output_size个时间步的输出
            outs = []
            for i in range(1, self.output_size + 1):
                step_out = x[:, -i, :]
                step_pred = self.fc(step_out)
                outs.append(step_pred)
            
            # 反转列表使得预测顺序正确 (从t+1到t+output_size)
            outs.reverse()
            out = torch.cat(outs, dim=1)
        
        return out 