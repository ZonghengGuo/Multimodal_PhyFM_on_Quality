import os
import numpy as np
import torch


class SiamDataset:
    def __init__(self, data_paths, augment=True):
        self.augment = augment
        self.data_list = []

        # 遍历你提供的每一个路径
        for path in data_paths:
            print(f"Loading data from '{path}'...")
            if not os.path.isdir(path):
                print(f"Warning：path '{path}' does not exist，skipping")
                continue

            # 遍历该路径下的所有文件
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    npy_path = os.path.join(path, file)
                    try:
                        data = np.load(npy_path, allow_pickle=True)
                        self.data_list.append(data)
                    except Exception as e:
                        print(f"Error: Loading {npy_path}: {e}")

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
        sample = self.data_list[idx]
        x1 = sample[0]
        x2 = sample[1]

        if self.augment:
            x1 = self.augment_signal(x1)
            x2 = self.augment_signal(x2)

        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)