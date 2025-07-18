import torch.optim as optim
from torch.utils.data import DataLoader
from BP_estimation.nets import VNet, SpectroResNet, MLPBP, finetune, PPGIABP, ResNet, UNet
from BP.tools import *
import torch.nn as nn
from models.PWSA import MultiModalLongformerQuality
from models.Transformer import MultiModalTransformerQuality
from models.ResNet import MultiModalResNet101Quality


class BPTrain:
    def __init__(self, args):
        self.DATA_DIR = args.raw_data_path
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.LEARNING_RATE = 0.0001
        self.backbone = args.backbone

    def training(self):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")

        X_train, y_train, X_val, y_val, train_mean, train_std = load_and_prepare_data(self.DATA_DIR)

        train_dataset = BPDataset(X_train, y_train, mean=train_mean, std=train_std)
        val_dataset = BPDataset(X_val, y_val, mean=train_mean, std=train_std)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        if self.backbone == "resnet":
            model = ResNet.ResNet18_1D()
        elif self.backbone == "spectro_resnet":
            model = SpectroResNet.RawSignalsDeepResNetPyTorch()
        elif self.backbone == "mlpbp":
            model = MLPBP.MLPMixer()
        elif self.backbone == "unet":
            model = UNet.UNet1D_for_Regression()
        elif self.backbone == "ppgiabp":
            model = PPGIABP.Seq2ValueRegressionModel()
        elif self.backbone == "vnet":
            model = VNet.VNet1D_for_Regression()

        elif self.backbone == "transformer_quality":
            backbone = MultiModalTransformerQuality(2, 512, 4, 2, 256)
            checkpoint = torch.load(f"model_saved/transformer_teacher.pth")
            backbone.load_state_dict(checkpoint["model_state_dict"])
            encoder = backbone.encoder

            for param in encoder.parameters():
                param.requires_grad = True

            model = finetune.FinetuneModel(pre_trained_encoder=encoder, num_classes=2)

        elif self.backbone == "resnet_quality":
            backbone = MultiModalResNet101Quality(2, 200, 18)
            encoder = backbone.encoder
            model = finetune.FinetuneModel(pre_trained_encoder=encoder, num_classes=2)

        elif self.backbone == "pwsa":
            backbone = MultiModalLongformerQuality(2, 512, 4, 2, 256, 8)
            checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
            backbone.load_state_dict(checkpoint["model_state_dict"])
            encoder = backbone.encoder

            for param in encoder.parameters():
                param.requires_grad = True

            model = finetune.FinetuneModel(pre_trained_encoder=encoder, num_classes=2)
        elif self.backbone == "pwsa_large":
            backbone = MultiModalLongformerQuality(2, 512, 4, 21, 512, 8)
            checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
            backbone.load_state_dict(checkpoint["model_state_dict"])
            encoder = backbone.encoder

            for param in encoder.parameters():
                param.requires_grad = True

            model = finetune.FinetuneModel(pre_trained_encoder=encoder, num_classes=2)
        elif self.backbone == "pwsa_huge":
            backbone = MultiModalLongformerQuality(2, 512, 8, 50, 2048, 8)
            checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
            backbone.load_state_dict(checkpoint["model_state_dict"])
            encoder = backbone.encoder

            for param in encoder.parameters():
                param.requires_grad = True

            model = finetune.FinetuneModel(pre_trained_encoder=encoder, num_classes=2)

        else:
            raise ValueError(f"未知的模型类型: {self.backbone}")


        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)

        for epoch in range(self.EPOCHS):
            model.train()
            running_loss = 0.0
            for i, (signals, labels) in enumerate(train_loader):
                signals = signals.float().to(DEVICE)
                labels = labels.to(DEVICE)

                if epoch == 0 and i == 0:
                    print(f"Shape of a training batch: {signals.shape}")  # 应为 (64, 2, 9000)

                outputs = model(signals)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.float().to(DEVICE)
                outputs = model(signals)
                all_predictions.append(outputs.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())

        predictions_np = np.concatenate(all_predictions, axis=0)
        true_labels_np = np.concatenate(all_true_labels, axis=0)

        calculate_metrics(true_labels_np, predictions_np, y_train)