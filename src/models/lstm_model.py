import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列预测
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        初始化LSTM模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, shape [batch_size, seq_length, input_size]
            
        返回:
            输出张量, shape [batch_size, output_size]
        """
        # LSTM前向计算
        # h0和c0默认为0
        out, _ = self.lstm(x)
        
        # 我们只需要最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out