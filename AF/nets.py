import torch
import torch.nn as nn
import math

class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()
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


class MLP(nn.Module):
    def __init__(self, num_classes=1):
        super(MLP, self).__init__()

        input_features = 2 * 9000

        self.layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=input_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class LSTMModel(nn.Module):
    def __init__(self, num_classes=1):
        super(LSTMModel, self).__init__()

        self.input_size = 2
        self.hidden_size = 128
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if self.num_layers > 1 else 0
        )
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_classes=1,
                 d_model=128, nhead=8, num_encoder_layers=6,
                 dim_feedforward=512, dropout=0.1,
                 patch_size=90):
        super(TransformerModel, self).__init__()

        self.patch_size = patch_size
        num_patches = 9000 // patch_size
        patch_dim = 2 * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        self.d_model = d_model

    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self.patch_to_embedding(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]
        out = self.classifier_head(cls_output)

        return out

class FinetuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, num_classes):
        super(FinetuneModel, self).__init__()
        self.encoder = pre_trained_encoder
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.encoder(x)

        pooled_features = torch.mean(features, dim=1)

        logits = self.classifier(pooled_features)

        return logits


class CnnLstmModel(nn.Module):
    def __init__(self, num_classes=1,
                 cnn_out_channels=128,
                 lstm_hidden_size=128,
                 lstm_num_layers=2):
        super(CnnLstmModel, self).__init__()

        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # Length: 9000 -> 3000

            nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)  # Length: 3000 -> 1000
        )
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier_head = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        x_cnn = self.cnn_extractor(x)
        x_lstm_in = x_cnn.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm_in)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.classifier_head(last_time_step_out)

        return out