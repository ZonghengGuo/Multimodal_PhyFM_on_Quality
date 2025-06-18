import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class SignalTransformerEncoder(nn.Module):
    def __init__(self, input_channels, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(SignalTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 1. Input Embedding Layer
        # Each channel is treated as a separate sequence initially
        # We'll use a Conv1D to project the 9000 length sequence to d_model,
        # or a Linear layer if we flatten the channels for each time step.
        # For simplicity, let's assume we want to project each time step's 2 features to d_model
        # If input is (Batch, Channels, Seq_len) -> (Batch, Seq_len, Channels) for Linear
        # For a (2, 9000) input, we can view it as 9000 time steps, each with 2 features.
        self.input_projection = nn.Linear(input_channels, d_model)


        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Feature Output Layer
        # You might want to average the sequence output, or take the [CLS] token output
        # Here, we'll average the output sequence to get a single feature vector per sample
        self.output_feature_layer = nn.Linear(d_model, d_model) # Or any desired feature dimension

    def forward(self, src):
        # src shape: (batch_size, input_channels, seq_len)
        # We need to transform it to (batch_size, seq_len, input_channels) for nn.Linear
        src = src.permute(0, 2, 1) # (batch_size, seq_len, input_channels)

        # Apply input projection
        # Input shape for self.input_projection: (batch_size, seq_len, input_channels)
        # Output shape after projection: (batch_size, seq_len, d_model)
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # Apply positional encoding
        # TransformerEncoderLayer expects (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(src) # (batch_size, seq_len, d_model)

        # Feature Output Layer
        # Option 1: Average pooling across the sequence dimension
        features = torch.mean(output, dim=1) # (batch_size, d_model)

        # Option 2: Using a learnable [CLS] token (requires adding an extra token at input)
        # For simplicity, we stick to average pooling here.

        features = self.output_feature_layer(features) # (batch_size, output_feature_dimension)

        return features

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的输入数据是 (Batch_size, Channels, Sequence_length)
    # 例如：(4, 2, 9000) -> 4个样本，每个样本2个通道，序列长度9000
    batch_size = 4
    input_channels = 2
    seq_len = 9000

    # 模型参数
    d_model = 128          # 嵌入维度，Transformer内部的维度
    nhead = 8              # 多头注意力的头数
    num_encoder_layers = 3 # Transformer Encoder的层数
    dim_feedforward = 512  # 前馈网络的维度

    # 实例化模型
    model = SignalTransformerEncoder(input_channels, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward)

    # 模拟输入数据
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    print(f"输入数据 shape: {dummy_input.shape}")

    # 前向传播
    output_features = model(dummy_input)
    print(f"输出特征 shape: {output_features.shape}")

    # 验证模型输出的特征维度
    assert output_features.shape == (batch_size, d_model)
    print(f"成功构建并运行 Transformer Encoder，输出特征维度为 {d_model}")