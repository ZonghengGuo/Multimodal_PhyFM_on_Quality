import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, channel_list=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, channel_list[0])
        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_list) - 1):
            self.down_blocks.append(Down(channel_list[i], channel_list[i + 1]))

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channel_list))
        for i in range(len(reversed_channels) - 1):
            self.up_blocks.append(Up(reversed_channels[i], reversed_channels[i + 1]))

        self.outc = nn.Conv1d(channel_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        x = skip_connections.pop()

        for block in self.up_blocks:
            skip = skip_connections.pop()
            x = block(x, skip)

        logits = self.outc(x)
        return logits


class UNet1D_for_Regression(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, **kwargs):
        super().__init__()
        f_out_ch = kwargs.get('f_out_ch', 16)
        self.feature_extractor = UNet1D(
            in_channels=in_channels,
            out_channels=f_out_ch,
            **kwargs
        )

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=f_out_ch, out_features=num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regression_head(features)