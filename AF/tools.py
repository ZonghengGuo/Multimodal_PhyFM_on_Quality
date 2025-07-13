from torch.utils.data import Dataset
import numpy as np
import torch
import random


class AfDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths, self.labels, self.augment = file_paths, labels, augment
        self.AUGMENT_PROB = 0.5
        self.NOISE_STD = 0.01
        self.MAX_SHIFT = 100

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx]).astype(np.float32)
        data_tensor = torch.from_numpy(data)
        if self.augment:
            if random.random() < self.AUGMENT_PROB: data_tensor += torch.randn_like(data_tensor) * self.NOISE_STD
            if random.random() < self.AUGMENT_PROB: data_tensor = torch.roll(data_tensor,
                                                                        shifts=random.randint(-self.MAX_SHIFT, self.MAX_SHIFT),
                                                                        dims=-1)
        return data_tensor, torch.tensor(self.labels[idx]).float()
