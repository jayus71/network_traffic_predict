import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_model import CNNModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel

class CrossAttention(nn.Module):
    """
    跨模型注意力模块，用于模型之间的交互
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        前向传播
        
        参数:
            query: 查询张量 [batch_size, hidden_size]
            key_value: 键值张量 [batch_size, hidden_size]
            
        返回:
            输出张量 [batch_size, hidden_size]
        """
        # 调整维度以适应多头注意力 [batch_size, 1, hidden_size]
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        
        # 第一个子层: 多头注意力
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        
        # 残差连接和层归一化
        attn_output = self.dropout(attn_output)
        output1 = self.norm1(query + attn_output)
        
        # 第二个子层: 前馈网络
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout(ffn_output)
        output2 = self.norm2(output1 + ffn_output)
        
        # 移除批次维度 [batch_size, hidden_size]
        return output2.squeeze(1)

class FeatureFusionModule(nn.Module):
    """
    特征融合模块，使用残差连接和交叉注意力来融合来自不同模型的特征
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # 交叉注意力模块
        self.cnn_lstm_attn = CrossAttention(hidden_size, dropout)
        self.cnn_transformer_attn = CrossAttention(hidden_size, dropout)
        self.lstm_cnn_attn = CrossAttention(hidden_size, dropout)
        self.lstm_transformer_attn = CrossAttention(hidden_size, dropout)
        self.transformer_cnn_attn = CrossAttention(hidden_size, dropout)
        self.transformer_lstm_attn = CrossAttention(hidden_size, dropout)
        
        # 每个模型特征的融合层
        self.cnn_fusion = nn.Linear(hidden_size, hidden_size)
        self.lstm_fusion = nn.Linear(hidden_size, hidden_size)
        self.transformer_fusion = nn.Linear(hidden_size, hidden_size)
        
        # 最终特征融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 最终注意力
        self.attention = nn.Linear(hidden_size * 3, 3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, cnn_feat, lstm_feat, transformer_feat):
        """
        前向传播
        
        参数:
            cnn_feat: CNN特征 [batch_size, hidden_size]
            lstm_feat: LSTM特征 [batch_size, hidden_size]
            transformer_feat: Transformer特征 [batch_size, hidden_size]
            
        返回:
            融合特征 [batch_size, hidden_size]
        """
        batch_size = cnn_feat.size(0)
        hidden_size = cnn_feat.size(1)
        
        # 使用残差连接和交叉注意力进行模型间交互
        # CNN接收来自LSTM和Transformer的信息
        cnn_lstm_interaction = self.cnn_lstm_attn(cnn_feat, lstm_feat)
        cnn_transformer_interaction = self.cnn_transformer_attn(cnn_feat, transformer_feat)
        cnn_enhanced = cnn_feat + cnn_lstm_interaction + cnn_transformer_interaction
        cnn_enhanced = self.cnn_fusion(cnn_enhanced)
        
        # LSTM接收来自CNN和Transformer的信息
        lstm_cnn_interaction = self.lstm_cnn_attn(lstm_feat, cnn_feat)
        lstm_transformer_interaction = self.lstm_transformer_attn(lstm_feat, transformer_feat)
        lstm_enhanced = lstm_feat + lstm_cnn_interaction + lstm_transformer_interaction
        lstm_enhanced = self.lstm_fusion(lstm_enhanced)
        
        # Transformer接收来自CNN和LSTM的信息
        transformer_cnn_interaction = self.transformer_cnn_attn(transformer_feat, cnn_feat)
        transformer_lstm_interaction = self.transformer_lstm_attn(transformer_feat, lstm_feat)
        transformer_enhanced = transformer_feat + transformer_cnn_interaction + transformer_lstm_interaction
        transformer_enhanced = self.transformer_fusion(transformer_enhanced)
        
        # 拼接增强特征用于计算注意力权重
        concat_features = torch.cat([cnn_enhanced, lstm_enhanced, transformer_enhanced], dim=1)
        attention_weights = self.softmax(self.attention(concat_features))
        
        # 加权融合
        weighted_cnn = cnn_enhanced * attention_weights[:, 0].unsqueeze(1)
        weighted_lstm = lstm_enhanced * attention_weights[:, 1].unsqueeze(1)
        weighted_transformer = transformer_enhanced * attention_weights[:, 2].unsqueeze(1)
        
        # 加权特征相加
        fused_features = weighted_cnn + weighted_lstm + weighted_transformer
        
        # 最终融合
        fused_features = self.final_fusion(fused_features)
        
        return fused_features

class ModifiedCNNModel(nn.Module):
    """修改版CNN模型，输出hidden_size维度的特征"""
    def __init__(self, input_size, seq_length, filters=64, kernel_size=3, num_layers=3, dropout=0.2, hidden_size=128):
        super(ModifiedCNNModel, self).__init__()
        
        # 创建多层卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层卷积
        self.conv_layers.append(nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size//2  # 使用填充保持序列长度
        ))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.BatchNorm1d(filters))
        
        # 中间卷积层
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size//2  # 使用填充保持序列长度
            ))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm1d(filters))
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(filters, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 调整输入形状以适应Conv1d [batch_size, input_size, seq_length]
        x = x.permute(0, 2, 1)
        
        # 卷积层前向计算
        for layer in self.conv_layers:
            x = layer(x)
        
        # 全局池化得到特征
        pooled = self.global_avg_pool(x).squeeze(-1)  # [batch_size, filters]
        
        # 特征映射
        features = self.feature_map(pooled)  # [batch_size, hidden_size]
        
        return features, pooled  # 返回映射后的特征和池化特征

class ModifiedLSTMModel(nn.Module):
    """修改版LSTM模型，输出hidden_size维度的特征"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(ModifiedLSTMModel, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # LSTM前向计算
        out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_out = out[:, -1, :]  # [batch_size, hidden_size]
        
        # 特征映射
        features = self.feature_map(last_out)  # [batch_size, hidden_size]
        
        return features, last_out  # 返回特征和LSTM最后一步输出

class ModifiedTransformerModel(nn.Module):
    """修改版Transformer模型，输出hidden_size维度的特征"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, nhead=4, dropout=0.2):
        super(ModifiedTransformerModel, self).__init__()
        
        # 特征映射层
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 特征映射
        x = self.embedding(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出
        last_out = x[:, -1, :]  # [batch_size, hidden_size]
        
        # 特征映射
        features = self.feature_map(last_out)  # [batch_size, hidden_size]
        
        return features, last_out  # 返回特征和Transformer最后一步输出

class EnsembleModel(nn.Module):
    """
    改进的集成模型，结合CNN、LSTM和Transformer，并添加模型间交互和残差连接
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
        
        # 创建修改版子模型，输出维度为hidden_size
        # CNN模型 - 使用最短的序列
        self.cnn_model = ModifiedCNNModel(
            input_size=input_size,
            seq_length=seq_lengths[2],  # 最短序列
            filters=cnn_config['filters'],
            kernel_size=cnn_config['kernel_size'],
            num_layers=cnn_config['num_layers'],
            dropout=dropout,
            hidden_size=hidden_size
        )
        
        # LSTM模型 - 使用中等长度的序列
        self.lstm_model = ModifiedLSTMModel(
            input_size=input_size,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=dropout
        )
        
        # Transformer模型 - 使用最长的序列
        self.transformer_model = ModifiedTransformerModel(
            input_size=input_size,
            hidden_size=transformer_config['hidden_size'],
            num_layers=transformer_config['num_layers'],
            nhead=transformer_config['nhead'],
            dropout=dropout
        )
        
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(hidden_size, dropout)
        
        # 输入特征投影 - 用于残差连接
        self.input_projection = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(3)
        ])
        
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
        
        # 对输入进行投影，用于残差连接
        # 使用每个序列的最后一个时间步
        res_inputs = [
            self.input_projection[i](x[i][:, -1, :]) 
            for i in range(len(x))
        ]
        
        # 分别通过三个模型，每个模型输出hidden_size维度的特征和中间特征
        cnn_feat, cnn_internal = self.cnn_model(x[2])  # 使用最短序列
        lstm_feat, lstm_internal = self.lstm_model(x[1])  # 使用中等长度序列
        transformer_feat, transformer_internal = self.transformer_model(x[0])  # 使用最长序列
        
        # 添加残差连接
        cnn_feat = cnn_feat + res_inputs[2]
        lstm_feat = lstm_feat + res_inputs[1]
        transformer_feat = transformer_feat + res_inputs[0]
        
        # 使用特征融合模块进行模型间交互和特征融合
        final_features = self.feature_fusion(cnn_feat, lstm_feat, transformer_feat)
        
        # Dropout
        final_features = self.dropout(final_features)
        
        # 输出层
        out = self.output_layer(final_features)
        
        return out 