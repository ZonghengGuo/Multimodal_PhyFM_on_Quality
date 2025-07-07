import torch
import torch.nn as nn


class FinetuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, num_classes):
        super(FinetuneModel, self).__init__()
        self.encoder = pre_trained_encoder
        self.classifier = nn.Linear(400, num_classes)

    def forward(self, x):
        features = self.encoder(x)

        pooled_features = torch.mean(features, dim=1)

        logits = self.classifier(pooled_features)

        return pooled_features, logits

