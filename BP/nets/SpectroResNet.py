import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB


class SingleChannelResNet(nn.Module):
    def __init__(self, in_channels=1, num_res_blocks=4, cnn_per_res=3,
                 kernel_sizes=[8, 5, 5, 3], init_filters=32, max_filters=64,
                 pool_size=2, pool_stride_size=1):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.cnn_per_res = cnn_per_res

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.pools = nn.ModuleList()

        num_filters = init_filters
        current_in_channels = in_channels

        for i in range(num_res_blocks):
            if current_in_channels != num_filters:
                shortcut = nn.Sequential(
                    nn.Conv1d(current_in_channels, num_filters, kernel_size=1, padding='same'),
                    nn.BatchNorm1d(num_filters)
                )
            else:
                shortcut = nn.BatchNorm1d(num_filters)
            self.shortcuts.append(shortcut)

            res_block_layers = nn.ModuleList()
            for j in range(cnn_per_res):
                in_c = current_in_channels if j == 0 else num_filters
                kernel = kernel_sizes[j] if j < len(kernel_sizes) else kernel_sizes[-1]

                res_block_layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_c, num_filters, kernel_size=kernel, padding='same'),
                        nn.BatchNorm1d(num_filters),
                        nn.ReLU() if j < cnn_per_res - 1 else nn.Identity()  # 最后一层后不加ReLU
                    )
                )
            self.blocks.append(res_block_layers)

            if i < 5:
                self.pools.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride_size))
            else:
                self.pools.append(nn.Identity())

            current_in_channels = num_filters
            num_filters = min(num_filters * 2, max_filters)

    def forward(self, x):
        block_input = x
        for i in range(self.num_res_blocks):
            res_conn = self.shortcuts[i](block_input)

            inner_x = block_input
            for j in range(self.cnn_per_res):
                inner_x = self.blocks[i][j](inner_x)

            x = F.relu(inner_x + res_conn)

            x = self.pools[i](x)
            block_input = x

        return x


class MidSpectrogramLayer(nn.Module):
    def __init__(self, n_dft=64, n_hop=64, out_features=32, l2_lambda=0.001):
        super().__init__()
        self.spectrogram = Spectrogram(n_fft=n_dft, hop_length=n_hop, power=2)
        self.to_db = AmplitudeToDB(stype='power', top_db=80)
        self.norm = nn.InstanceNorm2d(1)
        self.flatten = nn.Flatten()
        in_features = (n_dft // 2 + 1) * (9000 // n_hop + 1)

        self.fc_block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = self.spectrogram(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x


class RawSignalsDeepResNetPyTorch(nn.Module):
    def __init__(self, num_channels=2, seq_len=9000, l2_lambda=0.001, dropout_rate=0.25):
        super().__init__()
        self.num_channels = num_channels

        self.time_branches = nn.ModuleList([
            SingleChannelResNet(in_channels=1, init_filters=32, max_filters=64,
                                num_res_blocks=4, kernel_sizes=[8, 5, 5, 3],
                                pool_size=2, pool_stride_size=1)
            for _ in range(num_channels)
        ])

        self.freq_branches = nn.ModuleList([
            MidSpectrogramLayer(n_dft=64, n_hop=64, out_features=32)
            for _ in range(num_channels)
        ])

        self.time_bn = nn.BatchNorm1d(8996)
        self.time_gru = nn.GRU(input_size=64 * num_channels, hidden_size=65, batch_first=True)
        self.time_gru_bn = nn.BatchNorm1d(65)

        self.freq_bn = nn.BatchNorm1d(32 * num_channels)

        final_in_features = 65 + (32 * num_channels)  # time_gru_hidden + freq_features
        self.classifier = nn.Sequential(
            nn.Linear(final_in_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)  # Keras: Dense(2, activation="relu"), ReLU在回归任务中不常见，但我们遵循原模型
        )

    def forward(self, x):
        time_outputs = []
        freq_outputs = []
        for i in range(self.num_channels):
            channel_slice = x[:, i:i + 1, :]

            time_out = self.time_branches[i](channel_slice)
            time_outputs.append(time_out)

            freq_out = self.freq_branches[i](channel_slice)
            freq_outputs.append(freq_out)

        time_concat = torch.cat(time_outputs, dim=1)  # -> (B, F_out * num_channels, L_out)
        time_concat = time_concat.permute(0, 2, 1)  # -> (B, L_out, F_out * num_channels)
        time_bn = self.time_bn(time_concat)

        gru_out, _ = self.time_gru(time_bn)
        time_feature = self.time_gru_bn(gru_out[:, -1, :])

        freq_concat = torch.cat(freq_outputs, dim=1)  # -> (B, out_features * num_channels)
        freq_feature = self.freq_bn(freq_concat)

        combined_feature = torch.cat([time_feature, freq_feature], dim=1)
        output = self.classifier(combined_feature)

        return F.relu(output)