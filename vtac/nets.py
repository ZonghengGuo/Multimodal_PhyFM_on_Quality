import torch
import torch.nn as nn


class FinetuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, num_classes):
        super(FinetuneModel, self).__init__()
        self.encoder = pre_trained_encoder
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.encoder(x)  # 特征 shape: (B, 64)
        logits = self.classifier(features)

        return features, logits

