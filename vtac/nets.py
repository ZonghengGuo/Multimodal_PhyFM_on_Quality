import torch
import torch.nn as nn
from vtac.model import fcn
import torch
import torch.nn as nn


class FinetuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, num_classes=1, channels=6, chan_1=64, chan_2=128, chan_3=64, ks1=201,
                 ks2=101, ks3=51, dropout_prob=0.1):
        super(FinetuneModel, self).__init__()
        self.encoder = pre_trained_encoder
        pd1 = (ks1 - 1) // 2
        pd2 = (ks2 - 1) // 2
        pd3 = (ks3 - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv1d(channels, chan_1, kernel_size=ks1, stride=1, padding=pd1),
            nn.BatchNorm1d(chan_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_1, chan_2, kernel_size=ks2, stride=1, padding=pd2),
            nn.BatchNorm1d(chan_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_2, chan_3, kernel_size=ks3, stride=1, padding=pd3),
            nn.BatchNorm1d(chan_3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.AdaptiveMaxPool1d(1),
        )

        self.signal_feature = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_prob)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.encoder(x)  # [B, 6, 512]
        # pooled_features = torch.mean(features, dim=1)

        pooled_features = self.convs(features)
        print(pooled_features.shape)
        # signal = signal.view(-1, signal.size(1))
        # pooled_features = self.signal_feature(signal)

        logits = self.classifier(pooled_features)
        return logits


class ProgressiveFCN(nn.Module):
    def __init__(self, channels=6, chan_1=32, chan_2=64, chan_3=32, ks1=201, ks2=101, ks3=51, dropout_prob=0.2):
        super(ProgressiveFCN, self).__init__()

        pd1 = (ks1 - 1) // 2
        pd2 = (ks2 - 1) // 2
        pd3 = (ks3 - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv1d(channels, chan_1, kernel_size=ks1, stride=1, padding=pd1),
            nn.BatchNorm1d(chan_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_1, chan_2, kernel_size=ks2, stride=1, padding=pd2),
            nn.BatchNorm1d(chan_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_2, chan_3, kernel_size=ks3, stride=1, padding=pd3),
            nn.BatchNorm1d(chan_3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.AdaptiveMaxPool1d(1),
        )

        self.signal_feature = nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout_prob)
        )

    def forward(self, signal):
        signal = self.convs(signal).squeeze(-1)
        s_f = self.signal_feature(signal)

        return s_f


class FinetuneCNNModel(nn.Module):
    def __init__(self, backbone):
        super(FinetuneCNNModel, self).__init__()
        self.backbone = backbone

        fcn_input_dim = 6 * 512
        fcn_output_dim = 64

        self.fcn = nn.Sequential(
            nn.Linear(fcn_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, fcn_output_dim)
        )

        self.fcn_head = fcn()

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)  # [B, 6, 512]
        flattened_features = features.view(features.size(0), -1)

        encoder_output = self.fcn(flattened_features)

        resnet_features = self.fcn_head(x)

        combined_features = torch.cat([encoder_output, resnet_features], dim=1)

        logits = self.classifier(resnet_features)

        return logits
