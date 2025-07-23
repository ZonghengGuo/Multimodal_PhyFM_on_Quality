import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, dropout_prob=0.1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv0 = nn.Conv1d(2, 64, kernel_size=201, stride=2, padding=100)
        self.bn1 = nn.BatchNorm1d(64)
        self.stage0 = self._make_layer(block, 64, num_blocks[0], stride=2, kernel_size=101, padding=50)
        self.stage1 = self._make_layer(block, 128, num_blocks[1], stride=2, kernel_size=51, padding=25)
        self.stage2 = self._make_layer(block, 256, num_blocks[2], stride=2, kernel_size=25, padding=12)
        self.stage3 = self._make_layer(block, 128, num_blocks[3], stride=2, kernel_size=13, padding=6)
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.fc = nn.Linear(384 * block.expansion, num_classes) # Not used in this combined model
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
        x_pool = F.adaptive_avg_pool1d(out, 1)
        x_flat = x_pool.view(x_pool.size(0), -1)
        features = self.projector(x_flat)
        return features

def fcn(num_classes=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)