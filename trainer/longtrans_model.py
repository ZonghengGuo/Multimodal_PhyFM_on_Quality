import torch
import torch.nn as nn
import math
from longformer.longformer import LongformerSelfAttention, LongformerConfig


class FourierSpectrumProcessor(nn.Module):
    def __init__(self, target_sequence_length=1000, downsample_method='slice'):
        super(FourierSpectrumProcessor, self).__init__()
        self.target_sequence_length = target_sequence_length
        self.downsample_method = downsample_method

        if self.downsample_method == 'pool':
            pool_factor = 9000 // self.target_sequence_length
            if 9000 % self.target_sequence_length != 0:
                raise ValueError("For the “pool” method, the length of the original sequence must be an integer multiple of the length of the target sequence.")
            self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)

    def std_norm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + 1e-6)
        return normalized_x

    def forward(self, x):
        expected_seq_len = 9000
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        if self.downsample_method == 'slice':
            amplitude = amplitude[:, :, :self.target_sequence_length]
            phase = phase[:, :, :self.target_sequence_length]
        elif self.downsample_method == 'pool':
            amplitude = self.pool(amplitude)
            phase = self.pool(phase)
        else:
            raise ValueError("Unsupported downsampling method. Please select “slice” or “pool”.")

        normalized_amplitude = self.std_norm(amplitude)
        normalized_phase = self.std_norm(phase)

        return normalized_amplitude, normalized_phase

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


class LongformerEncoderLayer(nn.Module):
    def __init__(self, config: LongformerConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.attention = LongformerSelfAttention(config, layer_id=layer_id)
        self.sa_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = nn.GELU() # Longformer (基于 RoBERTa) 使用 GELU 激活函数

    def forward(
        self,
        hidden_states: torch.Tensor, # (batch_size, seq_len, hidden_size)
        attention_mask: torch.Tensor = None, # (batch_size, 1, 1, seq_len)
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        # 自注意力层
        # LongformerSelfAttention 期望 hidden_states 为 (batch_size, seq_len, embed_dim)
        # attention_mask 期望为 (batch_size, 1, 1, seq_len)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果 output_attentions=True，这里包含注意力权重

        # 残差连接和层归一化
        attention_output = self.sa_layer_norm(hidden_states + self.dropout(attention_output))

        # 前馈网络 (FFN)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        layer_output = self.output(intermediate_output)

        # 残差连接和层归一化
        layer_output = self.ffn_layer_norm(attention_output + self.dropout(layer_output))

        outputs = (layer_output,) + outputs
        return outputs


# ====================================================================
# 修改后的 SignalTransformerEncoder
# ====================================================================
class SignalTransformerEncoder(nn.Module):
    def __init__(self, input_channels, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, attention_window=8):
        super(SignalTransformerEncoder, self).__init__()

        self.d_model = d_model
        # 根据 TemporalConv 的输出 [batchsize, 1000, 18]，转置后 seq_len 为 18
        self.seq_len_after_conv = 18

        self.patch_embed = TemporalConv(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len_after_conv)

        # 构建 LongformerConfig
        # attention_window 是单侧窗口大小，例如 256 或 512。
        # 对于您转换后的序列长度 18，如果 attention_window 设置为 8，那么每个 token 将关注自身及两侧各 8 个 token (共 17 个)。
        # 这几乎覆盖了整个 18 个长度的序列，线性注意力在这里的效果不明显，但机制是正确的。
        # attention_mode='sliding_chunks' 是 Longformer 论文中建议的 PyTorch 实现
        longformer_config = LongformerConfig(
            hidden_size=d_model,
            num_attention_heads=nhead,
            intermediate_size=dim_feedforward,
            num_hidden_layers=num_encoder_layers,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            attention_window=[attention_window] * num_encoder_layers, # 所有层使用相同的窗口大小
            attention_mode='sliding_chunks', # 使用 PyTorch 实现的滑动分块注意力
            autoregressive=False # 此处为双向注意力，非自回归
        )

        # 使用自定义的 LongformerEncoderLayer 构建编码器层堆栈
        self.encoder_layers = nn.ModuleList([
            LongformerEncoderLayer(longformer_config, layer_id=i)
            for i in range(num_encoder_layers)
        ])

        self.output_feature_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        # src: [batchsize, 2, 9000]
        src = self.patch_embed(src) # [batchsize, d_model, 18] (例如，[1, 1000, 18])
        src = src.permute(0, 2, 1)  # [batchsize, 18, d_model] (例如，[1, 18, 1000])

        # 获取原始序列长度以便在填充后恢复
        original_seq_len = src.shape[1]

        # 位置编码
        src = self.pos_encoder(src) # [batchsize, 18, d_model]

        # 为 LongformerSelfAttention 准备注意力掩码
        # attention_mask 的值约定：
        #   -1 (或任意负数): 填充位置，不参与注意力
        #   0: 局部注意力 (滑动窗口)
        #   2 (或任意正数): 全局注意力
        # 此处我们只使用局部注意力，并处理填充。

        # 初始化注意力掩码，所有原始 token 都设置为 0 (局部注意力)
        attention_mask_base = torch.zeros(src.shape[0], src.shape[1], dtype=torch.long, device=src.device)

        # Longformer 的 'sliding_chunks' 注意力模式要求序列长度是 (2 * attention_window) 的倍数。
        # 获取 Longformer 配置中的单侧注意力窗口大小
        one_sided_window_size = self.encoder_layers[0].attention.attention_window
        effective_window_size_for_chunks = 2 * one_sided_window_size # 'w' in sliding_chunks.py 的含义

        current_seq_len = src.shape[1]
        padding_len = (effective_window_size_for_chunks - current_seq_len % effective_window_size_for_chunks) \
                      % effective_window_size_for_chunks

        if padding_len > 0:
            # 在序列维度上用 0.0 填充 src (特征张量)
            src = F.pad(src, (0, 0, 0, padding_len), value=0.0) # (batch, seq_len_padded, d_model)
            # 填充 attention_mask_base，新填充的位置设置为 -1 (表示无注意力)
            # 在 LongformerSelfAttention 内部，-1 的位置会被完全忽略
            attention_mask_padded = F.pad(attention_mask_base, (0, padding_len), value=-1)
        else:
            attention_mask_padded = attention_mask_base


        # 将注意力掩码转换为 LongformerSelfAttention 所需的形状：(batch_size, 1, 1, seq_len_padded)
        longformer_attention_mask = attention_mask_padded.unsqueeze(1).unsqueeze(1)

        # 应用 Longformer 编码器层
        hidden_states = src
        for layer_module in self.encoder_layers:
            # 每个层返回 (hidden_states, [attention_weights])
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=longformer_attention_mask,
                head_mask=None, # 此处未使用 head_mask
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # 如果存在填充，从输出中移除填充部分
        if padding_len > 0:
            hidden_states = hidden_states[:, :original_seq_len, :]

        # 最终的输出特征层
        output = self.output_feature_layer(hidden_states)

        return output

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

        self.decoder = SignalTransformerEncoder(
            input_channels=18,
            seq_len=1000,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=out_dim,
            dropout=0.1
        )

        self.decoder_amp_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        # Predicts phase from decoder output
        self.decoder_pha_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

    def encode(self, signal_data):
        signal_features = self.encoder(signal_data)
        return signal_features

    def decode(self, signal_data, features):
        decoder_features = self.decoder(features)

        feat_amp = self.decoder_amp_layer(decoder_features)
        feat_pha = self.decoder_pha_layer(decoder_features)

        return feat_amp, feat_pha


    def forward(self, signal_data):
        signal_features = self.encode(signal_data)
        feat_amp, feat_pha = self.decode(signal_data, signal_features)
        # amp, pha = self.spectrum(signal_data)

        return signal_features, feat_amp, feat_pha


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