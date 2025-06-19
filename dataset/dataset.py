import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SiamDataset(Dataset):
    def __init__(self, data_paths, augment=True):
        self.augment = augment
        self.data_list = []

        for path in data_paths:
            print(f"Scanning path: '{path}'...")
            if not os.path.isdir(path):
                print(f"Warning: path '{path}' does not exist, skipping.")
                continue

            for file in os.listdir(path):
                if file.endswith('.npy'):
                    npy_path = os.path.join(path, file)
                    self.data_list.append(npy_path)

        print(f"Total .npy files found: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def augment_signal(self, signal):
        if np.random.rand() < 0.5:
            noise = 0.01 * np.random.randn(*signal.shape)
            signal = signal + noise

        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale

        if np.random.rand() < 0.3:
            shift = np.random.uniform(-0.1, 0.1, size=(signal.shape[0], 1))
            signal = signal + shift

        if np.random.rand() < 0.3:
            T = signal.shape[1]
            mask_len = np.random.randint(T // 20, T // 10)
            start = np.random.randint(0, T - mask_len)
            signal[:, start:start + mask_len] = 0

        return signal

    def __getitem__(self, idx):
        npy_path = self.data_list[idx]

        try:
            data = np.load(npy_path, allow_pickle=True)
            self.data_list.append(data)
        except Exception as e:
            print(f"Error: Loading {npy_path}: {e}")

        x1, x2 = data[0], data[1]

        if x1.shape != (2, 9000) or x2.shape != (2, 9000):
            print(f"Warning: Skipping {npy_path} due to incorrect shape: {x1.shape}. Expected (2, 9000).")
            return None

        if self.augment:
            x1 = self.augment_signal(x1)
            x2 = self.augment_signal(x2)

        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)
