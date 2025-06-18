import os
import numpy as np
import torch
import scipy.signal

class SiamDataset:
    def __init__(self, pairs_save_path, augment=True):
        self.pairs_save_path = pairs_save_path
        self.augment = augment
        self.data_list = []

        for file in os.listdir(self.pairs_save_path):
            npy_path = os.path.join(self.pairs_save_path, file)
            data = np.load(npy_path, allow_pickle=True)
            self.data_list.append(data)


    def __len__(self):
        return len(self.data_list)

    def augment_signal(self, signal):
        # signal: (C, T)
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
        x1 = sample[0]  # shape: (3, 3750)
        x2 = sample[1]

        if self.augment:
            x1 = self.augment_signal(x1)
            x2 = self.augment_signal(x2)

        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)


# Demo
if __name__ == '__main__':
    dataset = SiamDataset("pre_train/pre_train_setting.json")
    print("数据集大小:", len(dataset))
    x1, x2 = dataset[0]
    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)