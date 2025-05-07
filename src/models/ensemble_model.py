import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_model import CNNModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel

class EnsembleModel(nn.Module):
    """
    集成模型，结合CNN、LSTM和Transformer，使用注意力机制融合不同模型的输出
    """
    def __init__(self, input_size, seq_lengths=[24, 12, 6], hidden_size=128, 
                 cnn_config=None, lstm_config=None, transformer_config=None, 
                 dropout=0.2, output_size=1):
        """
        初始化集成模型
        
        参数:
            input_size: 输入特征维度
            seq_lengths: 不同子模型的输入序列长度列表
            hidden_size: 隐藏层大小
            cnn_config: CNN模型配置
            lstm_config: LSTM模型配置
            transformer_config: Transformer模型配置
            dropout: Dropout比率
            output_size: 输出维度(预测步长)
        """
        super(EnsembleModel, self).__init__()
        
        self.output_size = output_size
        self.seq_lengths = seq_lengths
        
        # 默认配置
        if cnn_config is None:
            cnn_config = {
                'filters': 64,
                'kernel_size': 3,
                'num_layers': 3
            }
            
        if lstm_config is None:
            lstm_config = {
                'hidden_size': hidden_size,
                'num_layers': 2
            }
            
        if transformer_config is None:
            transformer_config = {
                'hidden_size': hidden_size,
                'num_layers': 2,
                'nhead': 4
            }
        
        # 创建子模型
        # CNN模型 - 使用最短的序列
        self.cnn_model = CNNModel(
            input_size=input_size,
            seq_length=seq_lengths[2],  # 最短序列
            filters=cnn_config['filters'],
            kernel_size=cnn_config['kernel_size'],
            num_layers=cnn_config['num_layers'],
            dropout=dropout,
            output_size=1  # 内部输出单步预测
        )
        
        # LSTM模型 - 使用中等长度的序列
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=dropout,
            output_size=1  # 内部输出单步预测
        )
        
        # Transformer模型 - 使用最长的序列
        self.transformer_model = TransformerModel(
            input_size=input_size,
            hidden_size=transformer_config['hidden_size'],
            num_layers=transformer_config['num_layers'],
            nhead=transformer_config['nhead'],
            dropout=dropout,
            output_size=1  # 内部输出单步预测
        )
        
        # 每个模型的特征提取器
        self.cnn_feature = nn.Linear(1, hidden_size)
        self.lstm_feature = nn.Linear(1, hidden_size)
        self.transformer_feature = nn.Linear(1, hidden_size)
        
        # 注意力层 - 对三个模型的输出进行加权
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 3, 3),
            nn.Softmax(dim=1)
        )
        
        # 最终输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量列表 [x_long, x_medium, x_short]
               x_long: 形状为 [batch_size, seq_length_long, input_size]
               x_medium: 形状为 [batch_size, seq_length_medium, input_size]
               x_short: 形状为 [batch_size, seq_length_short, input_size]
            
        返回:
            输出张量, 形状为 [batch_size, output_size]
        """
        batch_size = x[0].size(0)
        
        # 分别通过三个模型
        cnn_out = self.cnn_model(x[2])  # 使用最短序列
        lstm_out = self.lstm_model(x[1])  # 使用中等长度序列
        transformer_out = self.transformer_model(x[0])  # 使用最长序列
        
        # 提取特征
        cnn_feat = self.cnn_feature(cnn_out)
        lstm_feat = self.lstm_feature(lstm_out)
        transformer_feat = self.transformer_feature(transformer_out)
        
        # 合并特征
        combined_features = torch.cat([cnn_feat, lstm_feat, transformer_feat], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(combined_features)  # [batch_size, 3]
        
        # 加权特征
        weighted_cnn = cnn_feat * attention_weights[:, 0].unsqueeze(1)
        weighted_lstm = lstm_feat * attention_weights[:, 1].unsqueeze(1)
        weighted_transformer = transformer_feat * attention_weights[:, 2].unsqueeze(1)
        
        # 融合特征
        fused_features = weighted_cnn + weighted_lstm + weighted_transformer
        
        # Dropout
        fused_features = self.dropout(fused_features)
        
        # 输出层
        out = self.output_layer(fused_features)
        
        return out 