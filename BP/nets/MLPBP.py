import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')  # 交换回来
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels=2, dim=128, num_classes=1, num_patch=9000,
                 depth=4, token_dim=256, channel_dim=512, dropout=0.2):
        super().__init__()

        assert num_patch is not None, "num_patch must be provided."
        self.num_patch = num_patch

        self.patch_embedding = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LSTM(input_size=in_channels,
                    hidden_size=int(0.5 * dim),  # 每个方向的hidden_size
                    num_layers=1,
                    bidirectional=True,  # 双向LSTM
                    batch_first=True),
        )

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, self.num_patch, token_dim, channel_dim, dropout)
            for _ in range(depth)
        ])

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x, _ = self.patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)