from torch.utils.data import Dataset
import torch
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split


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

    X = np.concatenate(all_phy_list, axis=0)
    y_sbp = np.concatenate(all_sbp_list, axis=0)
    y_dbp = np.concatenate(all_dbp_list, axis=0)

    y = np.stack([y_sbp, y_dbp], axis=1)

    unique_subjects = sorted(list(set(subject_id_list)), key=int)
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=42)

    train_indices = [i for i, sub_id in enumerate(subject_id_list) if sub_id in train_subjects]
    val_indices = [i for i, sub_id in enumerate(subject_id_list) if sub_id in val_subjects]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    print(f"Data loaded. Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2), keepdims=True)

    print("Calculated normalization stats from training data.")

    return X_train, y_train, X_val, y_val, train_mean, train_std


def calculate_metrics(y_true, y_pred, y_train_for_mase):
    """
    Calculates and prints evaluation metrics.
    y_true: (N, 2) numpy array of true labels from the validation set.
    y_pred: (N, 2) numpy array of predictions from the validation set.
    y_train_for_mase: (M, 2) numpy array of the original training labels.
    """
    mae = np.mean(np.abs(y_true - y_pred), axis=0)

    error = y_true - y_pred
    me = np.mean(error, axis=0)
    sd = np.std(error, axis=0)

    naive_prediction = np.mean(y_train_for_mase, axis=0)

    mae_naive = np.mean(np.abs(y_train_for_mase - naive_prediction), axis=0)

    mase = mae / (mae_naive + 1e-8)

    print("\n--- Evaluation Metrics ---")
    print(f"SBP: MAE = {mae[0]:.2f}, ME±SD = {me[0]:.2f} ± {sd[0]:.2f}, MASE = {mase[0] * 100:.2f}%")
    print(f"DBP: MAE = {mae[1]:.2f}, ME±SD = {me[1]:.2f} ± {sd[1]:.2f}, MASE = {mase[1] * 100:.2f}%")
    print("--------------------------\n")