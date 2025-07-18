import torch
import torch.nn as nn


import torch
import torch.nn as nn

class FinetuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, num_classes=1, channels=2, chan_1=128, chan_2=256, chan_3=128, ks1=51, ks2=25, ks3=13, dropout_prob=0.5):
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

    def forward(self, x, random_s=None):
        # features = self.encoder(x)
        # pooled_features = torch.mean(features, dim=1)
        signal = self.convs(x).squeeze(-1)
        signal = signal.view(-1, signal.size(1))
        pooled_features = self.signal_feature(signal)

        if random_s is not None:
            random_s = self.convs(random_s).squeeze(-1)
            random_s = random_s.view(-1, random_s.size(1))
            random_s = self.signal_feature(random_s)

            # print(pooled_features.shape, random_s.shape)
            return self.classifier(pooled_features), pooled_features, random_s

        logits = self.classifier(pooled_features)
        return pooled_features, logits

class ProgressiveFCN(nn.Module):
    def __init__(self, channels=2, chan_1=128, chan_2=256, chan_3=128, ks1=51, ks2=25, ks3=13, dropout_prob=0.5):
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

        self.classifier = nn.Sequential(nn.Dropout(dropout_prob), nn.Linear(64, 1))

        self.signal_feature = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_prob)
        )


    def forward(self, signal, random_s):
        signal = self.convs(signal).squeeze(-1)
        signal = signal.view(-1, signal.size(1))
        s_f = self.signal_feature(signal)

        random_s = self.convs(random_s).squeeze(-1)
        random_s = random_s.view(-1, random_s.size(1))
        random_s = self.signal_feature(random_s)
        return self.classifier(s_f), s_f, random_s

class FinetuneCNNModel(nn.Module):
    def __init__(self, backbone, pretrained):
        super(FinetuneCNNModel, self).__init__()
        self.backbone = backbone
        self.fcn_head = ProgressiveFCN()
        self.pretrained = pretrained

    def forward(self, x, random_s):
        features = self.encoder(x)
        pooled_features = torch.mean(features, dim=1)
        logits = self.classifier(pooled_features)

        return pooled_features, logits
