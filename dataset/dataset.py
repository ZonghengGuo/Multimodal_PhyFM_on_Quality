import os
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class SiamDataset(Dataset):
    def __init__(self, chunk_dir, augment=True):
        self.augment = augment
        self.chunk_dir = chunk_dir

        print(f"Scanning for data chunks in: '{chunk_dir}'...")
        all_files = os.listdir(self.chunk_dir)

        chunk_files = [f for f in all_files if f.startswith('all_paired_data_chunk_') and f.endswith('.npy')]

        chunk_files.sort(key=lambda f: int(re.search(r'_(\d+)\.npy$', f).group(1)))

        self.chunk_paths = [os.path.join(self.chunk_dir, f) for f in chunk_files]

        if not self.chunk_paths:
            raise FileNotFoundError(f"No data chunks found in directory: {self.chunk_dir}")

        self.chunk_lengths = []
        self.cumulative_lengths = []
        total_length = 0
        print("Pre-calculating dataset size...")

        for path in self.chunk_paths:
            length = len(np.load(path, mmap_mode='r'))
            self.chunk_lengths.append(length)
            total_length += length
            self.cumulative_lengths.append(total_length)

        self._total_length = total_length
        print(f"Found {len(self.chunk_paths)} chunks with a total of {self._total_length} samples.")

        self.cached_chunk_index = -1
        self.cached_chunk_data = None

    def __len__(self):
        return self._total_length

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
        chunk_index = np.searchsorted(self.cumulative_lengths, idx, side='right')

        if chunk_index != self.cached_chunk_index:
            self.cached_chunk_data = np.load(self.chunk_paths[chunk_index])
            self.cached_chunk_index = chunk_index

        if chunk_index == 0:
            local_index = idx
        else:
            local_index = idx - self.cumulative_lengths[chunk_index - 1]

        data_pair = self.cached_chunk_data[local_index]
        x1, x2 = data_pair[0], data_pair[1]

        if self.augment:
            x1 = self.augment_signal(x1.copy())
            x2 = self.augment_signal(x2.copy())

        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)
