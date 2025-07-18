import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channels, d_model, output_channels, dropout_prob=0.1):
        super(ResNet, self).__init__()
        self.in_planes = d_model

        self.conv0 = nn.Conv1d(input_channels, d_model, kernel_size=201, stride=3, padding=100)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.stage0 = self._make_layer(block, d_model, num_blocks[0], stride=3, kernel_size=3, padding=1)
        self.stage1 = self._make_layer(block, d_model, num_blocks[1], stride=2, kernel_size=3, padding=1)  # stride=2
        self.stage2 = self._make_layer(block, d_model, num_blocks[2], stride=1, kernel_size=3, padding=1)  # stride=1

        if output_channels % block.expansion != 0:
            raise ValueError(
                f"output_channels ({output_channels}) must be divisible by block.expansion ({block.expansion})")
        final_planes = output_channels // block.expansion  # e.g., 18 // 2 = 9

        self.stage3 = self._make_layer(block, final_planes, num_blocks[3], stride=1, kernel_size=3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size, padding):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, kernel_size, padding))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

def resnet101(input_channels, d_model, output_channels):
    return ResNet(Bottleneck, [3, 8, 23, 3], input_channels, d_model, output_channels)

def resnet18(input_channels, d_model, output_channels):
    return ResNet(Bottleneck, [2, 2, 2, 2], input_channels, d_model, output_channels)

class MultiModalResNet101Quality(nn.Module):
    def __init__(self, input_channels, d_model, encoder_output_channels):
        super(MultiModalResNet101Quality, self).__init__()
        self.encoder = resnet101(
            input_channels=input_channels,
            d_model=d_model,
            output_channels=encoder_output_channels
        )

    def encode(self, signal_data):
        signal_features = self.encoder(signal_data)
        return signal_features


    def forward(self, signal_data):
        signal_features = self.encode(signal_data)
        return signal_features


class MultiModalResNet18Quality(nn.Module):
    def __init__(self, input_channels, d_model, encoder_output_channels):
        super(MultiModalResNet18Quality, self).__init__()
        self.encoder = resnet18(
            input_channels=input_channels,
            d_model=d_model,
            output_channels=encoder_output_channels
        )

    def encode(self, signal_data):
        signal_features = self.encoder(signal_data)
        return signal_features


    def forward(self, signal_data):
        signal_features = self.encode(signal_data)
        return signal_features


if __name__ == "__main__":
    batch_size = 4
    input_channels = 2
    seq_len = 9000
    d_model = 130

    target_channels = 18
    target_seq_len = 500

    # model = MultiModalResNet101Quality(input_channels, d_model, encoder_output_channels=target_channels)
    model = MultiModalResNet101Quality(input_channels, d_model, encoder_output_channels=target_channels)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_M = total_params / 1_000_000
    print(f"模型的总参数量: {params_in_M:.2f} M")

    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    print(f"输入数据 shape: {dummy_input.shape}\n")

    with torch.no_grad():
        output_features = model(dummy_input)

    print("--- Encoder 输出 ---")
    print(f"期望的输出特征 shape: torch.Size([{batch_size}, {target_channels}, {target_seq_len}])")
    print(f"实际的输出特征 shape: {output_features.shape}\n")

    # 验证形状是否匹配
    assert output_features.shape == (batch_size, target_channels, target_seq_len)
    print("\n✅ 成功！Encoder 的输出形状已达到目标。")
