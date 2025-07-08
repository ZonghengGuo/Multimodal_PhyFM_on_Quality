import numpy as np
import wfdb
from tqdm import tqdm
import os
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample


class AFProcessor:
    def __init__(self, args):
        self.dataset_path = args.raw_data_path
        self.original_fs = 125
        self.slide_segment_time = 30
        self.seg_save_path = os.path.join(args.raw_data_path, "segments")

    def interpolate_nan_multichannel(self, sig):
        # sig: shape (channels, time)
        interpolated = []
        for channel in sig:
            interpolated_channel = pd.Series(channel).interpolate(method='linear',
                                                                  limit_direction='both').to_numpy()
            interpolated.append(interpolated_channel)
        return np.array(interpolated)

    def resample_waveform(self, signal: np.ndarray, target_length: int = 9000) -> np.ndarray:
        upsampled_signal = np.zeros((2, target_length))

        for i in range(signal.shape[0]):
            upsampled_signal[i, :] = resample(signal[i, :], target_length)

        return upsampled_signal

    def normalize_to_minus_one_to_one(self, data):
        if data.size == 0 or np.all(data == data[0]):
            return data

        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            return np.zeros_like(data)

        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized_data

    def _butter_bandpass_filter(self, data: np.ndarray, fs: int, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        can_lowpass = self.highcut > 0 and self.highcut < nyq
        can_highpass = self.lowcut > 0 and self.lowcut < nyq

        if can_bandpass := (can_lowpass and can_highpass and self.lowcut < self.highcut):
            low = self.lowcut / nyq
            high = self.highcut / nyq
            b, a = butter(order, [low, high], btype='band', analog=False)
        elif can_highpass:
            low = self.lowcut / nyq
            b, a = butter(order, low, btype='high', analog=False)
        elif can_lowpass:
            high = self.highcut / nyq
            b, a = butter(order, high, btype='low', analog=False)
        else:
            return data

        if len(data) <= order * 3:
            print(f"警告: 数据长度 {len(data)} 过短，无法进行阶数为 {order} 的滤波。跳过滤波。")
            return data
        y = filtfilt(b, a, data)
        return y

    def _notch_filter(self, data: np.ndarray, fs: int, quality_factor: float = 30.0) -> np.ndarray:
        if self.powerline_freq <= 0:
            return data
        nyq = 0.5 * fs
        freq = self.powerline_freq / nyq
        if not (0 < freq < 1):
            return data

        if len(data) <= 8:
            print(f"警告: 数据长度 {len(data)} 过短，无法进行陷波滤波。跳过。")
            return data
        b, a = iirnotch(freq, quality_factor)
        y = filtfilt(b, a, data)
        return y

    def filter_ppg_channel(self, data: np.ndarray, fs: int) -> np.ndarray:
        b_notch, a_notch = iirnotch(60, 30, fs)
        tempfilt = filtfilt(b_notch, a_notch, data)
        N_bp, Wn_bp = butter(1, [0.5, 5], btype="band", analog=False, fs=fs)
        tempfilt = filtfilt(N_bp, Wn_bp, tempfilt)
        return tempfilt

    def process_save(self):
        for class_name in tqdm(os.listdir(self.dataset_path)):
            record_path = os.path.join(self.dataset_path, class_name)

            event_id_set = set()
            for event in os.listdir(record_path):
                event_id_set.add(os.path.splitext(event)[0])

            for event_id in event_id_set:
                required_samples = []

                event_path = os.path.join(record_path, event_id)
                record = wfdb.rdrecord(event_path)

                record_ppg = record.p_signal[:, 0]
                record_ecg = record.p_signal[:, 1]

                multi_channel_records = np.stack((record_ppg, record_ecg), axis=0)

                slide_segment_length = self.slide_segment_time * self.original_fs
                slide_segments = []

                for i, start in enumerate(range(0, len(record_ppg) - slide_segment_length + 1, slide_segment_length)):
                    end = start + slide_segment_length
                    slide_segment = multi_channel_records[:, start:end]

                    slide_segment = self.interpolate_nan_multichannel(slide_segment)

                    # filter
                    record.p_signal[:, 1] = self._notch_filter(record.p_signal[:, 1], fs=self.original_fs)
                    record.p_signal[:, 1] = self._butter_bandpass_filter(record.p_signal[:, 1], fs=self.original_fs)

                    record.p_signal[:, 0] = self.filter_ppg_channel(record.p_signal[:, 0], fs=self.original_fs)

                    resampled_slide_segment = self.resample_waveform(slide_segment, 9000)
                    print("set nan value to zero and normalize signal")

                    resampled_slide_segment = self.normalize_to_minus_one_to_one(resampled_slide_segment)

                    segment_save_path = os.path.join(self.seg_save_path, class_name, event_id)

                    segment_directory = os.path.dirname(segment_save_path)

                    os.makedirs(segment_directory, exist_ok=True)

                    np.save(f"{segment_save_path}_{i}", resampled_slide_segment)

                    print(f"save segments into: {segment_save_path}_{i}.npy")
