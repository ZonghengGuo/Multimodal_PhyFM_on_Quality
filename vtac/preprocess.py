import wfdb
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


class PreprocessVtac:
    def __init__(self, vtac_args):
        self.SAMPLING_FREQ = vtac_args.sampling_rate
        self.POWERLINE_FREQ = vtac_args.powerline_frequency
        self.dataset_path = vtac_args.raw_data_path


    def interpolate_nan_multichannel(self, sig):
        df = pd.DataFrame(sig)
        df.interpolate(method='linear', limit_direction='both', axis=0, inplace=True)
        return df.to_numpy()


    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a


    def butter_highpass(self, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a


    def butter_lowpass(self, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a


    def notch_filter(self, freq, Q, fs):
        b, a = iirnotch(freq, Q, fs)
        return b, a


    def filter_ecg_channel(self, data):
        b, a = self.butter_highpass(1.0, self.SAMPLING_FREQ)
        b2, a2 = self.butter_lowpass(30.0, self.SAMPLING_FREQ)
        tempfilt = filtfilt(b, a, data)
        tempfilt = filtfilt(b2, a2, tempfilt)
        b_notch, a_notch = self.notch_filter(self.POWERLINE_FREQ, 30, self.SAMPLING_FREQ)
        tempfilt = filtfilt(b_notch, a_notch, tempfilt)
        return tempfilt


    def filter_ppg_channel(self, data):
        b_notch, a_notch = self.notch_filter(self.POWERLINE_FREQ, 30, self.SAMPLING_FREQ)
        tempfilt = filtfilt(b_notch, a_notch, data)
        N_bp, Wn_bp = butter(1, [0.5, 5], btype="band", analog=False, fs=self.SAMPLING_FREQ)
        tempfilt = filtfilt(N_bp, Wn_bp, tempfilt)
        return tempfilt


    def filter_abp_channel(self, data):
        b_notch, a_notch = self.notch_filter(self.POWERLINE_FREQ, 30, self.SAMPLING_FREQ)
        tempfilt = filtfilt(b_notch, a_notch, data)
        b2, a2 = self.butter_lowpass(16.0, self.SAMPLING_FREQ)
        tempfilt = filtfilt(b2, a2, tempfilt)
        return tempfilt


    def min_max_norm(self, data, feature_range=(0, 1)):
        min_val = np.min(data)
        max_val = np.max(data)

        if max_val == min_val:  # Avoid division by zero
            return np.zeros_like(data) if feature_range[0] == 0 else np.full_like(data, feature_range[0])

        scale = feature_range[1] - feature_range[0]
        return feature_range[0] + (data - min_val) * scale / (max_val - min_val)

    def preprocess_save(self):
        dataset_path = self.dataset_path
        save_path = dataset_path + "/out/raw"

        os.makedirs(save_path, exist_ok=True)

        # get waveform and label
        waveform_path = os.path.join(dataset_path, "waveforms")
        csv_path = os.path.join(dataset_path, "event_labels.csv")
        event_label_df = pd.read_csv(csv_path)

        for record in tqdm(os.listdir(waveform_path)):
            record_path = os.path.join(waveform_path, record)

            event_id_set = set()
            for event in os.listdir(record_path):
                event_id_set.add(os.path.splitext(event)[0])

            for event_id in event_id_set:
                required_samples = []

                event_path = os.path.join(record_path, event_id)
                record = wfdb.rdrecord(event_path)

                sample_record = record.p_signal
                sample_name = record.record_name
                sample_length = record.sig_len
                sig_names = record.sig_name

                sample_record = self.interpolate_nan_multichannel(sample_record)

                if "PLETH" not in sig_names or "II" not in sig_names :
                    continue

                index_ppg = sig_names.index("PLETH")
                index_ii = sig_names.index("II")

                wave_ppg = sample_record[:, index_ppg]
                wave_ii = sample_record[:, index_ii]

                required_samples.append(self.min_max_norm(self.filter_ppg_channel(wave_ppg)))
                required_samples.append(self.min_max_norm(self.filter_ecg_channel(wave_ii)))

                required_samples = np.array(required_samples)

                # get label
                decision_value = event_label_df.loc[event_label_df['event'] == sample_name, 'decision'].values[0]

                if not decision_value:
                    decision_value = 0
                elif decision_value:
                    decision_value = 1

                # Save sample_record and decision_value as .npy files
                np.save(f"{save_path}/{sample_name}_record.npy", required_samples)
                np.save(f"{save_path}/{sample_name}_label.npy", decision_value)

    def splitting(self):
        save_path = os.path.join(self.dataset_path, "out/raw")
        split_path = os.path.join(self.dataset_path, "benchmark_data_split.csv")
        # Read the benchmark data split CSV
        split_df = pd.read_csv(split_path)

        # Initialize lists to store data
        train_samples = []
        val_samples = []
        test_samples = []

        train_labels = []
        val_labels = []
        test_labels = []

        # Iterate over the split dataframe
        for _, row in tqdm(split_df.iterrows()):
            event = row['event']
            split = row['split']

            record_file = os.path.join(save_path, f"{event}_record.npy")
            label_file = os.path.join(save_path, f"{event}_label.npy")

            if os.path.exists(record_file) and os.path.exists(label_file):
                record = np.load(record_file)
                label = np.load(label_file)

                if split == 'train':
                    train_samples.append(record)
                    train_labels.append(label)
                elif split == 'val':
                    val_samples.append(record)
                    val_labels.append(label)
                elif split == 'test':
                    test_samples.append(record)
                    test_labels.append(label)

        print(len(train_samples), len(val_samples), len(test_samples))

        # Convert lists to tensors
        train_samples = torch.tensor(np.array(train_samples))
        val_samples = torch.tensor(np.array(val_samples))
        test_samples = torch.tensor(np.array(test_samples))

        train_labels = torch.tensor(np.array(train_labels))
        val_labels = torch.tensor(np.array(val_labels))
        test_labels = torch.tensor(np.array(test_labels))

        # Save the datasets
        output_dir = os.path.join(self.dataset_path, "out/lead_selected")
        os.makedirs(output_dir, exist_ok=True)

        torch.save((train_samples, train_labels), os.path.join(output_dir, "train.pt"))
        torch.save((val_samples, val_labels), os.path.join(output_dir, "val.pt"))
        torch.save((test_samples, test_labels), os.path.join(output_dir, "test.pt"))

        print("Datasets saved successfully!")
