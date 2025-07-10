import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split


# --- 1. 定义模型架构 (Simple 1D CNN) ---
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=21, stride=1, padding=10),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=10, stride=10),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=10, stride=10),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=10, stride=10)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 9, 256),  # 9000 -> 900 -> 90 -> 9
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 最终输出2个值 (SBP, DBP)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# --- 2. 自定义数据集 ---
class BPDataset(Dataset):
    def __init__(self, signals, labels, mean=None, std=None):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.mean = mean
        self.std = std

        if self.mean is not None and self.std is not None:
            self.signals = (self.signals - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return torch.from_numpy(signal), torch.from_numpy(label)


# --- 3. 数据加载与预处理函数 ---
def load_and_prepare_data(data_dir, test_size=0.2):
    print("Loading and concatenating data files...")
    phy_files = sorted(glob.glob(os.path.join(data_dir, '*_phy.npy')))

    all_phy_list, all_sbp_list, all_dbp_list, subject_id_list = [], [], [], []

    for phy_file in phy_files:
        base_name = os.path.basename(phy_file).replace('_phy.npy', '')
        identifier = base_name.split('_')[1]

        sbp_path = os.path.join(data_dir, f"{base_name}_sbp.npy")
        dbp_path = os.path.join(data_dir, f"{base_name}_dbp.npy")

        if os.path.exists(sbp_path) and os.path.exists(dbp_path):
            phy_data = np.load(phy_file)
            sbp_data = np.load(sbp_path)
            dbp_data = np.load(dbp_path)

            all_phy_list.append(phy_data)
            all_sbp_list.append(sbp_data)
            all_dbp_list.append(dbp_data)
            subject_id_list.extend([identifier] * phy_data.shape[0])

    # 关键步骤：将所有文件的数据合并成一个大的Numpy数组
    X = np.concatenate(all_phy_list, axis=0)
    y_sbp = np.concatenate(all_sbp_list, axis=0)
    y_dbp = np.concatenate(all_dbp_list, axis=0)

    # 将SBP和DBP合并成一个标签数组，形状为 (N, 2)
    y = np.stack([y_sbp, y_dbp], axis=1)

    # 按受试者ID分割数据集，避免数据泄露
    unique_subjects = sorted(list(set(subject_id_list)), key=int)
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=42)

    train_indices = [i for i, sub_id in enumerate(subject_id_list) if sub_id in train_subjects]
    val_indices = [i for i, sub_id in enumerate(subject_id_list) if sub_id in val_subjects]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    print(f"Data loaded. Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 计算训练集的均值和标准差用于归一化
    # 计算每个通道在所有时间和样本上的均值/标准差
    # X_train shape: (N, C, L), C=2, L=9000
    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2), keepdims=True)

    print("Calculated normalization stats from training data.")

    return X_train, y_train, X_val, y_val, train_mean, train_std


# --- 4. 评估指标计算函数 ---
def calculate_metrics(y_true, y_pred, y_train_for_mase):
    # y_true, y_pred, y_train are numpy arrays of shape (N, 2)
    # SBP is column 0, DBP is column 1

    # MAE
    mae = np.mean(np.abs(y_true - y_pred), axis=0)

    # ME, SD
    error = y_true - y_pred
    me = np.mean(error, axis=0)
    sd = np.std(error, axis=0)

    # MASE
    mae_naive = np.mean(np.abs(np.diff(y_train_for_mase, axis=0)), axis=0)
    mase = mae / (mae_naive + 1e-8)

    print("\n--- Evaluation Metrics ---")
    print(f"SBP: MAE = {mae[0]:.2f}, ME±SD = {me[0]:.2f} ± {sd[0]:.2f}, MASE = {mase[0] * 100:.2f}%")
    print(f"DBP: MAE = {mae[1]:.2f}, ME±SD = {me[1]:.2f} ± {sd[1]:.2f}, MASE = {mase[1] * 100:.2f}%")
    print("--------------------------\n")


# --- 5. 主训练流程 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    DATA_DIR = r"D:\database\BP\save"  # 使用 r"..." 避免转义问题
    BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 数据准备 ---
    X_train, y_train, X_val, y_val, train_mean, train_std = load_and_prepare_data(DATA_DIR)

    train_dataset = BPDataset(X_train, y_train, mean=train_mean, std=train_std)
    val_dataset = BPDataset(X_val, y_val, mean=train_mean, std=train_std)

    # 正确创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型、损失函数、优化器 ---
    model = Simple1DCNN().to(DEVICE)
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            # 检查输入形状 (只在第一次迭代时打印)
            if epoch == 0 and i == 0:
                print(f"Shape of a training batch: {signals.shape}")  # 应为 (64, 2, 9000)

            # 前向传播
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

    # --- 评估模型 ---
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(DEVICE)
            outputs = model(signals)
            all_predictions.append(outputs.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

    # 将所有批次的预测和真实值合并
    predictions_np = np.concatenate(all_predictions, axis=0)
    true_labels_np = np.concatenate(all_true_labels, axis=0)

    # --- 计算并打印最终评估指标 ---
    calculate_metrics(true_labels_np, predictions_np, y_train)