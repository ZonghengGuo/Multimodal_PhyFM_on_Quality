import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def forward(self, x, **kwargs):
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return x


class SimplifiedMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, 2 * d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(d_inner, self.d_state + self.d_state, bias=False)  # for B, C
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)  # for Δ

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, self.d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm_conv = x_ssm.transpose(1, 2)
        x_ssm_conv = self.conv1d(x_ssm_conv)[:, :, :L]
        x_ssm_conv = x_ssm_conv.transpose(1, 2)
        x_ssm_activated = F.silu(x_ssm_conv)

        delta = F.softplus(self.dt_proj(x_ssm_activated))

        x_proj_out = self.x_proj(x_ssm_activated)
        B_param, C_param = x_proj_out.chunk(2, dim=-1)

        y_ssm = self.selective_scan(x_ssm_activated, delta, B_param, C_param)

        z_gate = F.silu(z)
        y = y_ssm * z_gate
        output = self.out_proj(y)

        return output

    def selective_scan(self, u, delta, B, C):
        B_size, L_size, d_inner = u.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log.float())

        h = torch.zeros(B_size, d_inner, d_state, device=u.device)
        ys = []

        for i in range(L_size):
            u_i = u[:, i, :]
            delta_i = delta[:, i, :]
            B_i = B[:, i, :]
            C_i = C[:, i, :]

            delta_A_i = torch.exp(delta_i.unsqueeze(-1) * A)
            delta_B_i = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)

            h = delta_A_i * h + delta_B_i * u_i.unsqueeze(-1)

            y_i = torch.sum(h * C_i.unsqueeze(1), dim=-1)
            ys.append(y_i)

        y_ssm = torch.stack(ys, dim=1)

        y_ssm = y_ssm + u * self.D

        return y_ssm


class MambaEncoder(nn.Module):
    def __init__(self, input_channels, d_model, num_mamba_layers, **mamba_kwargs):
        super(MambaEncoder, self).__init__()
        self.d_model = d_model
        self.patch_embed = TemporalConv(input_channels, d_model)

        self.mamba_layers = nn.ModuleList([
            SimplifiedMamba(d_model=d_model, **mamba_kwargs)
            for _ in range(num_mamba_layers)
        ])
        self.output_feature_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.patch_embed(src)
        src = src.permute(0, 2, 1)
        output = src
        for mamba_layer in self.mamba_layers:
            output = mamba_layer(output)
        features = self.output_feature_layer(output)
        return features


class MultiModalMambaQuality(nn.Module):
    def __init__(self, input_channels, d_model, num_mamba_layers, out_dim):
        super(MultiModalMambaQuality, self).__init__()
        self.encoder = MambaEncoder(
            input_channels=input_channels,
            d_model=d_model,
            num_mamba_layers=num_mamba_layers,
            d_state=16,
            d_conv=4
        )
        self.decoder = MambaEncoder(
            input_channels=d_model,
            d_model=d_model,
            num_mamba_layers=num_mamba_layers,
            d_state=16,
            d_conv=4
        )
        self.decoder_amp_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.decoder_pha_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model))

    def encode(self, signal_data):
        return self.encoder(signal_data)

    def decode(self, features):
        features_permuted = features.permute(0, 2, 1)
        decoder_features = self.decoder(features_permuted)
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

    d_model = 500

    num_mamba_layers = 2
    out_dim = 256

    model = MultiModalMambaQuality(input_channels, d_model, num_mamba_layers, out_dim)

    if isinstance(model.encoder.mamba_layers[0], nn.Identity):
        print("请先安装 Mamba: pip install mamba-ssm causal-conv1d")
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_in_M = total_params / 1_000_000
        print(f"模型的总参数量: {params_in_M:.2f} M")

        dummy_input = torch.randn(batch_size, input_channels, seq_len)
        print(f"输入数据 shape: {dummy_input.shape}")

        # 前向传播
        output_features, feat_amp, feat_pha = model(dummy_input)
        print(f"输出特征 shape: {output_features.shape}")
        print(f"amp shape: {feat_amp.shape}")
        print(f"pha shape: {feat_pha.shape}")