import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# --- 主模型: 序列到值回归模型 ---
class Seq2ValueRegressionModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_classes=2,
                 n_layers=3, dropout=0.1, bidirectional_encoder=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder_directions = 2 if bidirectional_encoder else 1

        # ---- 1. 编码器 (从之前的模型中保留) ----
        # 编码器用于读取和理解整个输入序列
        self.encoder_grus = nn.ModuleList()
        encoder_input_size = input_size
        for i in range(n_layers):
            self.encoder_grus.append(
                nn.GRU(encoder_input_size, hidden_size, batch_first=True, bidirectional=bidirectional_encoder)
            )
            # 下一层 GRU 的输入是 (原始输入 + 上一层GRU输出)
            encoder_input_size += (hidden_size * self.encoder_directions)

        # ---- 2. 新的回归头 (替换掉解码器和注意力) ----
        # 这个头的作用是聚合序列信息并输出最终的2个值
        self.regression_head = nn.Sequential(
            # 自适应平均池化层，将任意长度的序列压缩为长度为1的序列
            # 它会把 (B, C, L) -> (B, C, 1)
            nn.AdaptiveAvgPool1d(1),

            # 展平以便送入全连接层
            nn.Flatten(),

            nn.BatchNorm1d(encoder_input_size),  # 对聚合后的特征进行归一化
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # 最终的全连接层，将特征映射到2个输出值
            nn.Linear(in_features=encoder_input_size, out_features=num_classes)
        )

        # 执行权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 复制原始代码中的权重初始化逻辑
        for gru in self.encoder_grus:
            for name in gru._flat_weights_names:
                if "bias_ih_l" in name:
                    bias = getattr(gru, name)
                    n = bias.size(0)
                    bias.data[0:n // 3].fill_(0.)
            for name in gru._flat_weights_names:
                if "weight_hh_l" in name:
                    weight_hh = getattr(gru, name)
                    nn.init.orthogonal_(weight_hh)

    def forward(self, x):
        # x 初始形状: (B, C, L), e.g., (128, 2, 9000)

        # ---- 编码器前向传播 ----
        # 将输入从 (B, C, L) 转换为 (B, L, C) 以适应GRU
        encoder_input = x.permute(0, 2, 1)

        # 通过编码器GRUs (与之前相同)
        encoder_outputs = encoder_input
        for gru in self.encoder_grus:
            gru_output, _ = gru(encoder_outputs)
            encoder_outputs = torch.cat((encoder_outputs, gru_output), dim=2)

        # ---- 回归头前向传播 ----
        # AdaptiveAvgPool1d 需要 (B, C, L) 格式
        # encoder_outputs 当前是 (B, L, C)，所以需要再次转换
        features_for_pooling = encoder_outputs.permute(0, 2, 1)

        # 将特征送入回归头
        # 输出形状将是 (B, 2)
        final_output = self.regression_head(features_for_pooling)

        return final_output