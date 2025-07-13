import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# --- 辅助模块 1: 带残差连接的卷积块 ---
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_convs, dropout_rate):
        super().__init__()
        layers = []
        for i in range(n_convs):
            # 第一个卷积层可能需要改变通道数
            current_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(current_in, out_channels, kernel_size=kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            # 对1D数据应用空间丢弃 (Spatial Dropout)
            layers.append(nn.Dropout2d(p=dropout_rate))  # Dropout2d作用于(N, C, L), 会丢弃整个通道

        self.convs = nn.Sequential(*layers)
        # 如果输入输出通道不同，需要一个1x1卷积来匹配残差连接的维度
        self.residual_conv = nn.Conv1d(in_channels, out_channels,
                                       kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.convs(x)
        return x + residual


# --- 辅助模块 2: 下采样块 ---
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, kernel_size, factor, dropout_rate):
        super().__init__()
        self.conv_block = ResConvBlock(in_channels, in_channels, kernel_size, n_convs, dropout_rate)
        self.down_conv = nn.Conv1d(in_channels, out_channels, kernel_size=factor, stride=factor)

    def forward(self, x):
        # 特征提取，用于跳跃连接
        features_for_skip = self.conv_block(x)
        # 下采样
        x_down = self.down_conv(features_for_skip)
        return x_down, features_for_skip


# --- 辅助模块 3: 上采样块 ---
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, kernel_size, factor, dropout_rate):
        super().__init__()
        self.up_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=factor, stride=factor)
        # 上采样后，通道数变为 out_channels + out_channels (来自跳跃连接)
        self.conv_block = ResConvBlock(out_channels * 2, out_channels, kernel_size, n_convs, dropout_rate)

    def forward(self, x, skip_features):
        x_up = self.up_conv(x)

        # 裁剪跳跃连接的特征以匹配上采样后的尺寸
        diff = skip_features.size(2) - x_up.size(2)
        x_up = F.pad(x_up, [diff // 2, diff - diff // 2])

        x_concat = torch.cat([skip_features, x_up], dim=1)
        return self.conv_block(x_concat)


# --- 主模型: VNet1D ---
class VNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, f_out_ch=16,
                 num_convolutions_list=(1, 2, 3), bottom_convolutions=4,
                 kernel_size=5, factor=2, dropout_rate=0.2):
        super().__init__()

        num_levels = len(num_convolutions_list)

        # 初始卷积层
        self.init_layer = ResConvBlock(in_channels, f_out_ch, kernel_size, 1, dropout_rate)

        # ---- 编码器 (下采样路径) ----
        self.down_layers = nn.ModuleList()
        current_channels = f_out_ch
        for n_convs in num_convolutions_list:
            self.down_layers.append(
                DownBlock(current_channels, current_channels * factor, n_convs, kernel_size, factor, dropout_rate)
            )
            current_channels *= factor

        # ---- 网络最底部的中间层 ----
        self.mid_layer = ResConvBlock(current_channels, current_channels, kernel_size, bottom_convolutions,
                                      dropout_rate)

        # ---- 解码器 (上采样路径) ----
        self.up_layers = nn.ModuleList()
        for n_convs in reversed(num_convolutions_list):
            self.up_layers.append(
                UpBlock(current_channels, current_channels // factor, n_convs, kernel_size, factor, dropout_rate)
            )
            current_channels //= factor

        # ---- 最终输出层 ----
        self.last_layer = nn.Conv1d(f_out_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.init_layer(x)

        skip_connections = []
        for down_layer in self.down_layers:
            x, skip = down_layer(x)
            skip_connections.append(skip)

        x = self.mid_layer(x)

        for up_layer, skip in zip(self.up_layers, reversed(skip_connections)):
            x = up_layer(x, skip)

        return self.last_layer(x)


class VNet1D_for_Regression(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, **kwargs):
        super().__init__()
        f_out_ch = kwargs.get('f_out_ch', 16)
        self.feature_extractor = VNet1D(
            in_channels=in_channels,
            out_channels=f_out_ch,  # 让U-Net的输出通道数等于它最后一层的通道数
            **kwargs
        )

        # 2. 接着，我们创建一个新的“回归头”
        self.regression_head = nn.Sequential(
            # 使用自适应平均池化将序列压缩成一个点
            # 无论输入序列多长，输出长度都是1
            nn.AdaptiveAvgPool1d(1),

            # 展平，以便送入全连接层
            nn.Flatten(),

            # 全连接层，将特征映射到最终的2个输出值
            nn.Linear(in_features=f_out_ch, out_features=num_classes)
        )

    def forward(self, x):
        # 首先用V-Net提取特征图
        # 输出形状会是 (B, C_out, L_out)，例如 (128, 16, 9000)
        features = self.feature_extractor(x)

        return self.regression_head(features)