import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(Simple1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x