
# HOO
import os
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample


class PreprocessBP:
    def __init__(self, args):
        self.SAMPLING_FREQ = args.sampling_rate
        self.POWERLINE_FREQ = args.powerline_frequency
        self.dataset_path = args.raw_data_path
        self.segment_length = 30 # seconds
        self.original_fs = 125
        self.temp_size = self.segment_length * self.original_fs

    def interpolate_nan_multichannel(self, sig):
        interpolated = []
        for channel in sig:
            interpolated_channel = pd.Series(channel).interpolate(method='linear',
                                                                  limit_direction='both').to_numpy()
            interpolated.append(interpolated_channel)
        return np.array(interpolated)

    def resample_waveform(self, signal: np.ndarray, target_length: int = 9000) -> np.ndarray:
        upsampled_signal = np.zeros((3, target_length))
        for i in range(signal.shape[0]):
            upsampled_signal[i, :] = resample(signal[i, :], target_length)
        return upsampled_signal

    def process_save(self):
        for part_name in tqdm(os.listdir(self.dataset_path)):
            if "part" in part_name:
                multi_channel_records = []
                bp = []
                sbp = []
                dbp = []

                part_path = os.path.join(self.dataset_path, part_name)
                sample_file = scipy.io.loadmat(part_path)['p']
                sample_length = len(sample_file[0])
                print(f"processing {part_name}... total Samples: {sample_length}")

                for i in range(sample_length):
                    temp_mat = sample_file[0, i]
                    temp_length = temp_mat.shape[1]

                    for i, start in enumerate(
                            range(0, temp_length - self.temp_size + 1, self.temp_size)):
                        end = start + self.temp_size
                        temp_mat_segment = temp_mat[:, start:end]
                        temp_mat_segment = self.interpolate_nan_multichannel(temp_mat_segment)
                        temp_mat_segment = self.resample_waveform(temp_mat_segment, 9000)

                        temp_ppg_seg = temp_mat_segment[0, :]
                        temp_bp_seg = temp_mat_segment[1, :]
                        temp_ecg_seg = temp_mat_segment[2, :]

                        max_value = max(temp_bp_seg)
                        min_value = min(temp_bp_seg)

                        multi_channel_record = np.stack((temp_ppg_seg, temp_ecg_seg), axis=0)

                        multi_channel_records.append(multi_channel_record)
                        sbp.append(max_value)
                        dbp.append(min_value)
                        bp.append(temp_bp_seg)

                multi_channel_records = np.array(multi_channel_records)
                bp = np.array(bp)
                sbp = np.array(sbp)
                dbp = np.array(dbp)

                save_path = os.path.join(self.dataset_path, "save")
                os.makedirs(save_path, exist_ok=True)

                part_name_pure = part_name.rsplit('.', 1)[0]

                np.save(os.path.join(save_path, f"{part_name_pure}_phy.npy"), multi_channel_records)
                np.save(os.path.join(save_path, f"{part_name_pure}_bp.npy"), bp)
                np.save(os.path.join(save_path, f"{part_name_pure}_sbp.npy"), sbp)
                np.save(os.path.join(save_path, f"{part_name_pure}_dbp.npy"), dbp)

            else:
                continue


