import os
import glob
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from AF.tools import AfDataset
from AF.nets import *
from models.PWSA import MultiModalLongformerQuality
from models.Transformer import MultiModalTransformerQuality
from models.Mamba import MultiModalMambaQuality
from models.ResNet import MultiModalResNet101Quality


class AfTrainer:
    def __init__(self, args):
        self.DATA_DIR = os.path.join(args.raw_data_path, "segment")
        self.LEARNING_RATE = 0.0001
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.NUM_FOLDS = 10
        self.backbone = args.backbone

    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_files = []
        all_labels = []

        for label_str in ["0", "1"]:
            class_dir = os.path.join(self.DATA_DIR, label_str)
            files_in_class = glob.glob(os.path.join(class_dir, "*.npy"))
            all_files.extend(files_in_class)
            all_labels.extend([int(label_str)] * len(files_in_class))

        print(f"扫描完成。共找到 {len(all_files)} 个文件。")
        print(f"总类别分布: Class 0: {all_labels.count(0)}, Class 1: {all_labels.count(1)}")

        X_placeholder = np.zeros(len(all_files))
        skf = StratifiedKFold(n_splits=self.NUM_FOLDS, shuffle=True, random_state=42)

        fold_metrics_list = []

        pos_weight_value = 912 / 488
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print(f"使用的固定pos_weight (类别0数/类别1数): {pos_weight_value:.2f}")

        for fold, (train_indices, val_indices) in enumerate(skf.split(X_placeholder, all_labels)):
            print("-" * 50)
            print(f"交叉验证: 第 {fold + 1} / {self.NUM_FOLDS} 折")

            train_files = [all_files[i] for i in train_indices]
            train_labels = [all_labels[i] for i in train_indices]
            val_files = [all_files[i] for i in val_indices]
            val_labels = [all_labels[i] for i in val_indices]

            print(f"原始训练集大小: {len(train_files)}")
            train_files_augmented = train_files * 10
            train_labels_augmented = train_labels * 10
            print(f"扩充后训练集大小: {len(train_files_augmented)}")

            train_dataset = AfDataset(file_paths=train_files_augmented, labels=train_labels_augmented, augment=True)
            val_dataset = AfDataset(file_paths=val_files, labels=val_labels, augment=False)
            print(f"创建训练集: {len(train_dataset)} 个样本")
            print(
                f"创建验证集: {len(val_dataset)} 个样本 | 标签分布: Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")

            train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2,
                                      pin_memory=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2,
                                    pin_memory=True)

            if self.backbone == "mlp":
                model = MLP()
            elif self.backbone == "cnn":
                model = CNN()
            elif self.backbone == "lstm":
                model = LSTMModel()
            elif self.backbone == "transformer":
                model = TransformerModel()
            elif self.backbone == "cnnlstm":
                model = CnnLstmModel()

            elif self.backbone == "resnet":
                backbone = MultiModalResNet101Quality(2, 200, 18)
                encoder = backbone.encoder
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            elif self.backbone == "transformer_quality":
                backbone = MultiModalTransformerQuality(2, 512, 4, 2, 256)
                checkpoint = torch.load(f"model_saved/transformer_teacher.pth")
                backbone.load_state_dict(checkpoint["model_state_dict"])
                encoder = backbone.encoder
                for param in encoder.parameters():
                    param.requires_grad = True
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            elif self.backbone == "mamba":
                backbone = MultiModalMambaQuality(2, 400, 2, 256)
                checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
                backbone.load_state_dict(checkpoint["model_state_dict"])
                encoder = backbone.encoder
                for param in encoder.parameters():
                    param.requires_grad = True
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            elif self.backbone == "pwsa":
                backbone = MultiModalLongformerQuality(2, 512, 4, 2, 256, 8)
                checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
                backbone.load_state_dict(checkpoint["model_state_dict"])
                encoder = backbone.encoder
                for param in encoder.parameters():
                    param.requires_grad = True
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            elif self.backbone == "pwsa_large":
                backbone = MultiModalLongformerQuality(2, 512, 4, 21, 512, 8)
                checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
                backbone.load_state_dict(checkpoint["model_state_dict"])
                encoder = backbone.encoder
                for param in encoder.parameters():
                    param.requires_grad = True
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            elif self.backbone == "pwsa_huge":
                backbone = MultiModalLongformerQuality(2, 512, 8, 50, 2048, 8)
                checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
                backbone.load_state_dict(checkpoint["model_state_dict"])
                encoder = backbone.encoder
                for param in encoder.parameters():
                    param.requires_grad = True
                model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

            else:
                raise ValueError(f"未知的模型类型: {self.backbone}")


            model = model.to(device)
            print(f"模型: {self.backbone}, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.6f}M")

            # 优化器与损失函数
            optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)
            # 修改点: 直接使用预先计算好的pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # --- 训练与验证循环 (此部分逻辑正确，无需修改) ---
            for epoch in range(self.EPOCHS):
                model.train()
                for inputs, labels in train_loader:
                    # ... (训练代码不变)
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                model.eval()
                epoch_labels, epoch_preds_probs = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        # ... (验证代码不变)
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        outputs = model(inputs)
                        # 注意处理squeeze()可能带来的维度问题，尤其是在batch_size=1时
                        if outputs.ndim > 1:
                            epoch_preds_probs.extend(torch.sigmoid(outputs).cpu().squeeze().tolist())
                        else:  # 处理batch_size=1的情况
                            epoch_preds_probs.append(torch.sigmoid(outputs).cpu().item())
                        epoch_labels.extend(labels.cpu().tolist())

                # ... (指标计算代码不变)
                epoch_preds_binary = [1 if p > 0.5 else 0 for p in epoch_preds_probs]
                tn, fp, fn, tp = confusion_matrix(epoch_labels, epoch_preds_binary, labels=[0, 1]).ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                tpr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                ppv_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * (ppv_precision * tpr_recall) / (ppv_precision + tpr_recall) if (
                                                                                                    ppv_precision + tpr_recall) > 0 else 0.0
                auc = roc_auc_score(epoch_labels, epoch_preds_probs)

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{self.EPOCHS} | AUC: {auc:.4f} | F1: {f1:.4f}")
                scheduler.step(auc)

            final_metrics = {'acc': accuracy, 'tpr': tpr_recall, 'tnr': tnr_specificity, 'ppv': ppv_precision, 'f1': f1,
                             'auc': auc}
            fold_metrics_list.append(final_metrics)
            print(f"第 {fold + 1} 折完成。最终AUC: {auc:.4f}, F1-Score: {f1:.4f}")

        print("-" * 50)
        print(f"{self.NUM_FOLDS}折交叉验证完成。")
        for key in fold_metrics_list[0].keys():
            values = [m[key] for m in fold_metrics_list]
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"平均 {key.upper()}: {mean_val:.4f} ± {std_val:.4f}")