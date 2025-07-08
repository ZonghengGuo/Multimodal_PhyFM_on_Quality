import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from AF.tools import *


class AfTrainer:
    def __init__(self, args):
        self.DATA_DIR = os.path.join(args.raw_data_path, "segments")
        self.LEARNING_RATE = 0.0001
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.NUM_FOLDS = 5


# --- 4. 辅助函数 (不变) ---
def get_subject_id_from_path(filepath):
    match = re.search(r'((?:non_)?af_\d+)', os.path.basename(filepath))
    return match.group(1) if match else None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")


    subject_to_files = defaultdict(list)
    all_files = glob.glob(os.path.join(DATA_DIR, "*", "*.npy"))
    for f_path in all_files:
        subject_id = get_subject_id_from_path(f_path)
        if subject_id: subject_to_files[subject_id].append(f_path)
    unique_subjects = np.array(sorted(list(subject_to_files.keys())))
    print(f"扫描完成。共找到 {len(all_files)} 个文件，来自 {len(unique_subjects)} 个独立的 Subject。")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_metrics_list = []

    for fold, (train_subject_indices, val_subject_indices) in enumerate(kf.split(unique_subjects)):
        print("-" * 50)
        print(f"交叉验证: 第 {fold + 1} / {NUM_FOLDS} 折")

        train_subjects = unique_subjects[train_subject_indices]
        val_subjects = unique_subjects[val_subject_indices]

        train_files, train_labels = [], []
        for sub in train_subjects:
            files = subject_to_files[sub]
            train_files.extend(files)
            train_labels.extend([int(os.path.basename(os.path.dirname(f))) for f in files])

        val_files, val_labels = [], []
        for sub in val_subjects:
            files = subject_to_files[sub]
            val_files.extend(files)
            val_labels.extend([int(os.path.basename(os.path.dirname(f))) for f in files])

        train_dataset = NpyDataset(file_paths=train_files, labels=train_labels, augment=True)
        val_dataset = NpyDataset(file_paths=val_files, labels=val_labels, augment=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                  pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                pin_memory=True)

        model = Simple1DCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)

        neg_count = train_labels.count(0)
        pos_count = train_labels.count(1)
        pos_weight = torch.tensor([neg_count / pos_count if pos_count > 0 else 1.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

            model.eval()
            epoch_labels = []
            epoch_preds_probs = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu()
                    epoch_preds_probs.extend(probs.squeeze().tolist())
                    epoch_labels.extend(labels.cpu().tolist())

            epoch_preds_binary = [1 if p > 0.5 else 0 for p in epoch_preds_probs]


            tn, fp, fn, tp = confusion_matrix(epoch_labels, epoch_preds_binary, labels=[0, 1]).ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            tpr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR, Recall, Sensitivity
            tnr_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR, Specificity
            ppv_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # PPV, Precision
            f1 = 2 * (ppv_precision * tpr_recall) / (ppv_precision + tpr_recall) if (
                                                                                                ppv_precision + tpr_recall) > 0 else 0.0

            try:
                auc = roc_auc_score(epoch_labels, epoch_preds_probs)
            except ValueError:
                auc = 0.5

            print(
                f"  Epoch {epoch + 1}/{EPOCHS} | AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f} | TPR: {tpr_recall:.4f} | TNR: {tnr_specificity:.4f} | PPV: {ppv_precision:.4f}")

            scheduler.step(auc)

        final_metrics = {'acc': accuracy, 'tpr': tpr_recall, 'tnr': tnr_specificity, 'ppv': ppv_precision, 'f1': f1,
                         'auc': auc}
        fold_metrics_list.append(final_metrics)
        print(f"第 {fold + 1} 折完成。最终AUC: {auc:.4f}, F1-Score: {f1:.4f}")

    # --- 总结所有折的交叉验证结果 ---
    print("-" * 50)
    print(f"{NUM_FOLDS}折交叉验证完成。")

    # 计算每个指标的平均值和标准差
    for key in fold_metrics_list[0].keys():
        values = [m[key] for m in fold_metrics_list]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"平均 {key.upper()}: {mean_val:.4f} ± {std_val:.4f}")