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
        return x + self.pe[:x.size(0), :]

class TemporalConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.out_chans = out_chans
        self.conv1 = nn.Conv1d(in_chans, out_chans//100, kernel_size=201, stride=5, padding=100)
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(1, out_chans//100)
        self.conv2 = nn.Conv1d(out_chans//100, out_chans//10, kernel_size=101, stride=10, padding=50)
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(1, out_chans//10)
        self.conv3 = nn.Conv1d(out_chans//10, out_chans, kernel_size=51, stride=10, padding=25)
        self.norm3 = nn.GroupNorm(1, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs): # [1, 2, 9000]
        x = self.gelu1(self.norm1(self.conv1(x))) # [1, 10, 1800]
        x = self.gelu2(self.norm2(self.conv2(x))) # [1, 100, 180]
        x = self.gelu3(self.norm3(self.conv3(x))) # [1, 1000, 18]
        return x


class SignalTransformerEncoder(nn.Module):
    def __init__(self, input_channels, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(SignalTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.patch_embed = TemporalConv(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_feature_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.patch_embed(src) # [batchsize, , 600]
        # print(src.shape)
        src = src.permute(0, 2, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) # [1, 18, 1000]
        features = self.output_feature_layer(output)

        return features

#
class MultiModalTransformerQuality(nn.Module):
    def __init__(self, input_channels, d_model, nhead, num_encoder_layers, out_dim):
        super(MultiModalTransformerQuality, self).__init__()
        self.encoder = SignalTransformerEncoder(
            input_channels=input_channels,
            seq_len=9000,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=out_dim,
            dropout=0.1
        )

    def encode(self, signal_data):
        signal_features = self.encoder(signal_data)
        return signal_features

    def decoder(self, features):
        return


    def forward(self, signal_data):
        signal_features = self.encode(signal_data)

        return signal_features


# --- 使用示例 ---
if __name__ == "__main__":
    batch_size = 1
    input_channels = 18
    seq_len = 1000

    d_model = 1000
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 512

    # 实例化模型
    model = MultiModalTransformerQuality(input_channels, d_model, nhead, num_encoder_layers, dim_feedforward)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_M = total_params / 1_000_000
    print(f"模型的总参数量: {params_in_M:.2f} M")

    # 模拟输入数据
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    print(f"输入数据 shape: {dummy_input.shape}")

    # 前向传播
    output_features = model(dummy_input)
    print(f"输出特征 shape: {output_features.shape}")