import torch
import torch.nn as nn
import math


def create_longformer_attention_mask(seq_len, window_size, global_attention_indices):
    if window_size % 2 != 0:
        raise ValueError("Window size must be an even number.")

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    half_window = window_size // 2

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, start:end] = False

    if global_attention_indices:
        global_indices_tensor = torch.tensor(global_attention_indices, dtype=torch.long)
        mask[global_indices_tensor, :] = False
        mask[:, global_indices_tensor] = False

    return mask


class LongformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class LongformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, window_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LongformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.window_size = window_size
        self.num_layers = num_layers

    def forward(self, src, global_attention_indices=[0]):
        seq_len = src.size(1)

        attention_mask = create_longformer_attention_mask(
            seq_len, self.window_size, global_attention_indices
        ).to(src.device)

        output = src
        for layer in self.layers:
            output = layer(output, src_mask=attention_mask)

        return output


class FourierSpectrumProcessor(nn.Module):
    def __init__(self, target_sequence_length=1000, downsample_method='slice'):
        super(FourierSpectrumProcessor, self).__init__()
        self.target_sequence_length = target_sequence_length
        self.downsample_method = downsample_method
        if self.downsample_method == 'pool':
            pool_factor = 9000 // self.target_sequence_length
            if 9000 % self.target_sequence_length != 0: raise ValueError(
                "For the “pool” method, the length of the original sequence must be an integer multiple of the length of the target sequence.")
            self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)

    def std_norm(self, x):
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6)

    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude, phase = torch.abs(x_fft), torch.angle(x_fft)
        if self.downsample_method == 'slice':
            amplitude, phase = amplitude[:, :, :self.target_sequence_length], phase[:, :, :self.target_sequence_length]
        elif self.downsample_method == 'pool':
            amplitude, phase = self.pool(amplitude), self.pool(phase)
        else:
            raise ValueError("Unsupported downsampling method. Please select “slice” or “pool”.")
        return self.std_norm(amplitude), self.std_norm(phase)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=9000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TemporalConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.out_chans = out_chans
        self.conv1 = nn.Conv1d(in_chans, out_chans // 100, kernel_size=201, stride=5, padding=100)
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(1, out_chans // 100)
        self.conv2 = nn.Conv1d(out_chans // 100, out_chans // 10, kernel_size=101, stride=10, padding=50)
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(1, out_chans // 10)
        self.conv3 = nn.Conv1d(out_chans // 10, out_chans, kernel_size=51, stride=10, padding=25)
        self.norm3 = nn.GroupNorm(1, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return x


class SignalLongformerEncoder(nn.Module):
    def __init__(self, input_channels, d_model, nhead, num_encoder_layers, dim_feedforward, window_size, dropout=0.1):
        super(SignalLongformerEncoder, self).__init__()

        self.d_model = d_model
        self.patch_embed = TemporalConv(input_channels, d_model)
        self.output_seq_len = 18

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.output_seq_len)

        self.longformer_encoder = LongformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            window_size=window_size,
            dropout=dropout
        )

        self.output_feature_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.patch_embed(src)

        src = src.permute(0, 2, 1)

        src = self.pos_encoder(src)

        output = self.longformer_encoder(src)

        features = self.output_feature_layer(output)

        return features


class MultiModalLongformerQuality(nn.Module):
    def __init__(self, input_channels, d_model, nhead, num_encoder_layers, out_dim, window_size):
        super(MultiModalLongformerQuality, self).__init__()

        self.encoder = SignalLongformerEncoder(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=out_dim,
            window_size=window_size,
            dropout=0.1
        )

        self.decoder = SignalLongformerEncoder(
            input_channels=18,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=out_dim,
            window_size=window_size,
            dropout=0.1
        )

        self.decoder_amp_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

        self.decoder_pha_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

    def encode(self, signal_data):
        return self.encoder(signal_data)

    def decode(self, features):
        decoder_features = self.decoder(features)

        feat_amp = self.decoder_amp_layer(decoder_features)
        feat_pha = self.decoder_pha_layer(decoder_features)

        return feat_amp, feat_pha

    def forward(self, signal_data):
        signal_features = self.encode(signal_data)
        feat_amp, feat_pha = self.decode(signal_features)
        return signal_features, feat_amp, feat_pha


if __name__ == "__main__":
    batch_size = 2
    input_channels = 2
    seq_len = 9000

    d_model = 768
    nhead = 4
    num_encoder_layers = 10
    dim_feedforward = 512
    window_size = 8

    model = MultiModalLongformerQuality(
        input_channels, d_model, nhead, num_encoder_layers, dim_feedforward, window_size
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_M = total_params / 1_000_000
    print(f"模型的总参数量: {params_in_M:.2f} M")

    # 模拟输入数据
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    print(f"输入数据 shape: {dummy_input.shape}")

    # 前向传播
    output_features, feat_amp, feat_pha = model(dummy_input)

    # 检查输出形状是否符合预期
    # TemporalConv输出长度为18，所以特征输出长度也是18
    print(f"输出特征 shape: {output_features.shape}")
    print(f"amp shape: {feat_amp.shape}")
    print(f"pha shape: {feat_pha.shape}")