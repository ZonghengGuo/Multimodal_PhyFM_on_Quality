import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from scipy.signal import butter, filtfilt, iirnotch, resample, detrend
import vitaldb
from biosppy.signals import ecg
from scipy import signal, stats
from tqdm import tqdm


class BaseProcessor:
    def __init__(self, args: argparse.Namespace):
        self.raw_data_path = args.raw_data_path
        self.seg_save_path = args.seg_save_path
        self.qua_save_path = args.qua_save_path

        self.target_sfreq = args.rsfreq
        self.lowcut = args.l_freq
        self.highcut = args.h_freq
        self.powerline_freq = getattr(args, 'powerline_freq', 50.0)

        self.ecg_segments_path = args.seg_save_path
        self.qualities_path = args.qua_save_path
        self.pairs_save_path = args.pair_save_path

        self.quality_rank = {
            "Excellent": 0,
            "Good": 1,
            "Acceptable": 2,
            "Poor": 3,
            "Bad": 4
        }

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

    def resample_waveform(self, original_sfreq, signal):
        num_original_samples = len(signal)
        num_target_samples = int(num_original_samples * (self.target_sfreq / original_sfreq))
        if num_target_samples == 0 and num_original_samples > 0:
            num_target_samples = 1

        if num_original_samples > 0 and num_target_samples > 0:
             resampled_data = resample(signal, num_target_samples)
        elif num_original_samples > 0 and num_target_samples == 0:
             resampled_data = np.array([])
             print(f"警告: 重采样目标长度为0，原始长度{num_original_samples}。信号变为空。")
        else:
             resampled_data = np.array([])
        return resampled_data

    def normalize_to_minus_one_to_one(self, data):
        if data.size == 0 or np.all(data == data[0]):
            return data

        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            return np.zeros_like(data)

        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized_data

    def interpolate_nan_multichannel(self, sig):
        # sig: shape (channels, time)
        interpolated = []
        for channel in sig:
            interpolated_channel = pd.Series(channel).interpolate(method='linear', limit_direction='both').to_numpy()
            interpolated.append(interpolated_channel)
        return np.array(interpolated)

    def is_any_constant_signal(self, slide_segment):
        return np.any(np.all(slide_segment == slide_segment[:, [0]], axis=1))

    def is_nan_ratio_exceed_any(self, sig, threshold, fs, segment_time):
        nan_ratios = np.isnan(sig).sum(axis=1) / (fs * segment_time)
        return np.any(nan_ratios > threshold)

    def peak_detection(self, sig, fs, band_freq=45):
        r = ecg.ecg(signal=sig, sampling_rate=fs, show=False, band_frequency=band_freq)
        return r['rpeaks']

    def sample_entropy(self, signal, m, r, scale):
        return np.random.random()

    def rrSQI(self, ECG, qrs, freq):
        if len(qrs) < 20 or len(ECG) < 200:
            return np.array([]), np.array([]), 0.0

        fs = freq
        timeECG = np.arange(len(ECG)) / fs
        RR = np.diff(timeECG[qrs])

        # Thresholds
        rangeHR = [40, 120]  # bpm
        dHR = 0.30
        dPeriod = 0.5  # 0.5 seconds
        noiseEN = 2

        # Beat quality
        HR = 60.0 / RR
        badHR = np.where((HR < rangeHR[0]) | (HR > rangeHR[1]))[0]

        jerkPeriod = 1 + np.where(np.abs(np.diff(RR)) > dPeriod)[0]
        jerkHR = 1 + np.where(np.abs(np.diff(HR)) / HR[:-1] > dHR)[0]

        # ECG quality
        w = int(fs * 1)  # 1-second window
        E = []
        sampen = []
        ecg = detrend(ECG) / np.std(ECG) + 10

        for i in range(0, len(ECG) - w, w):
            e = ecg[i:i + w]
            E.append(np.sum(e))
            sampen.append(self.sample_entropy(e, 1, 0.1, 0))

        E = np.array(E)
        sampen = np.array(sampen)

        B = np.ceil(qrs / w).astype(int)
        B = B[:-1]  # Match RR length
        B[B >= len(E)] = len(E) - 1  # Ensure indices are within bounds

        noise = np.column_stack((E[B], sampen[B]))

        M = np.percentile(E, 95)
        j = np.where(noise[:, 0] > M)[0]
        jj = np.where(noise[:, 1] > noiseEN)[0]

        # Initialize beat quality matrix
        bq = np.zeros((len(qrs) - 1, 6), dtype=int)
        bq[badHR, 1] = 1
        bq[jerkPeriod, 2] = 1
        bq[jerkHR, 3] = 1
        bq[j, 4] = 1
        bq[jj, 5] = 1

        # Combine conditions for column 1
        bq[:, 0] = bq[:, 1] | bq[:, 2] | bq[:, 3]

        # Make all "...101..." into "...111..."
        y = bq[:, 0]
        y[np.where(np.diff(y, 2) == 2)[0] + 1] = 1
        bq[:, 0] = y

        BeatQ = bq.astype(bool)

        # Fraction of good beats overall
        r = len(np.where(bq[:, 0] == 0)[0]) / len(qrs)

        # BeatN (noisy beats)
        bn = bq[:, 4] | bq[:, 5]
        BeatN = bn.astype(bool)

        return BeatQ, BeatN, r

    def ppg_SQI(self, ppg, fs=125):

        # 预处理：去趋势和归一化
        ppg = signal.detrend(ppg)
        ppg_normalized = (ppg - np.mean(ppg)) / np.std(ppg)

        try:
            # 1. 信号功率质量指标
            f, Pxx = signal.welch(ppg_normalized, fs=fs, nperseg=min(len(ppg), 256))
            signal_power = np.sum(Pxx[(f > 0.5) & (f < 5)])  # PPG主要频率范围
            noise_power = np.sum(Pxx[(f > 5) & (f < 25)])  # 高频噪声范围
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_score = 1 / (1 + np.exp(-0.5 * (snr - 5)))

            peaks, _ = signal.find_peaks(ppg_normalized, distance=fs // 2, prominence=0.5)  # 增加prominence阈值
            if len(peaks) < 2:
                return 0.0

            # 2. 灌注指数
            ac_component = np.max(ppg) - np.min(ppg)
            dc_component = np.mean(ppg)
            perfusion_index = ac_component / dc_component
            perfusion_score = np.clip(perfusion_index * 10, 0, 1)  # 归一化到0-1

            # 3. 信号偏度
            skewness = stats.skew(ppg_normalized)
            skewness_score = 1 / (1 + np.exp(-5 * (abs(skewness) - 1)))  # 理想PPG偏度接近0

            # 4. 相对功率
            total_power = np.sum(Pxx[(f > 0.1) & (f < 50)])
            rel_power = signal_power / (total_power + 1e-10)
            rel_power_score = np.clip(rel_power * 2, 0, 1)  # 归一化

            # 5. 节律性
            rr_intervals = np.diff(peaks) / fs * 1000
            rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
            rhythm_score = 1 / (1 + rr_cv)

            # 6. 幅度变化
            peak_amplitudes = ppg_normalized[peaks]
            amp_cv = np.std(peak_amplitudes) / np.mean(peak_amplitudes)
            amp_score = 1 / (1 + amp_cv)

            # 7. 自相关
            autocorr = np.correlate(ppg_normalized, ppg_normalized, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr_score = np.mean(autocorr[:fs] / autocorr[0])

            # 8. 熵
            hist, _ = np.histogram(ppg_normalized, bins=20)
            hist = hist / np.sum(hist)
            entropy = stats.entropy(hist)
            entropy_score = 1 - (entropy / np.log(20))

            # 综合所有指标
            quality_score = (
                    0.2 * snr_score +
                    0.2 * rhythm_score +
                    0.1 * amp_score +
                    0.1 * autocorr_score +
                    0.1 * entropy_score +
                    0.1 * perfusion_score +
                    0.1 * skewness_score +
                    0.1 * rel_power_score
            )

            return np.clip(quality_score, 0, 1)

        except Exception as e:
            print(f"Error in quality assessment: {e}")
            return 0.0

    def scale_ppg_score(self, qua_ppg):
        self.qua_ppg_scaled = (qua_ppg - 0.5) / 0.3
        self.qua_ppg_scaled = max(0, min(self.qua_ppg_scaled, 1))
        return self.qua_ppg_scaled


class MimicProcessor(BaseProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.required_sigs = ['II', 'Pleth']
        self.req_seg_duration = 300
        self.slide_segment_time = 30
        self.nan_limit = 0.2

    def process_record(self):
        dataset_path = self.raw_data_path
        for subject_title_name in os.listdir(dataset_path):
            subject_title_path = os.path.join(dataset_path, subject_title_name)
            for subject_name in os.listdir(subject_title_path):
                subject_path = os.path.join(subject_title_path, subject_name)
                if os.path.isdir(subject_path):

                    for filename in os.listdir(subject_path):
                        file_path = os.path.join(subject_path, filename)

                        if os.path.isdir(file_path):
                            hea_files = [f for f in os.listdir(file_path) if f.endswith('.hea')]

                            for hea_file in hea_files:
                                wave_name = os.path.splitext(hea_file)[0]
                                wave_path = os.path.join(file_path, wave_name)
                                try:
                                    segment_metadata = wfdb.rdrecord(wave_path)
                                except Exception as e:
                                    print(f"读取记录 {wave_path} 时发生错误: {e}")
                                    continue

                                original_fs = int(segment_metadata.fs)
                                print(original_fs)
                                # Check if the segments include required lead
                                sigs_leads = segment_metadata.sig_name

                                if len(sigs_leads) < 2:
                                    print("Not enough channels, skip..")
                                    continue

                                if not all(x in sigs_leads for x in self.required_sigs):
                                    print(f'{sigs_leads} is missing signal of II, PLETH')
                                    continue

                                # check if the segments is longer than f{shortest_minutes}
                                seg_length = segment_metadata.sig_len / original_fs
                                if seg_length < self.req_seg_duration:
                                    print(f' (too short at {seg_length / 60:.1f} mins)')
                                    continue

                                print(f"Have the {sigs_leads}..........")

                                # segment every signal into 30s slides
                                sig_ppg_index = segment_metadata.sig_name.index('Pleth')
                                sig_ii_index = segment_metadata.sig_name.index('II')

                                sig_ppg = segment_metadata.p_signal[:, sig_ppg_index]
                                sig_ii = segment_metadata.p_signal[:, sig_ii_index]

                                multi_channel_signal = np.stack((sig_ppg, sig_ii), axis=0)

                                # setting
                                slide_segment_length = self.slide_segment_time * original_fs
                                slide_segments = []
                                qua_labels = []

                                # divide into 30 sec, and discard the last slide (<30s)
                                for i, start in enumerate(range(0, len(sig_ppg) - slide_segment_length + 1, slide_segment_length)):
                                    end = start + slide_segment_length
                                    slide_segment = multi_channel_signal[:, start:end]

                                    # check if too much nan value
                                    if self.is_nan_ratio_exceed_any(slide_segment, self.nan_limit, original_fs, self.slide_segment_time):
                                        print(f"too much missing value, nan ratio is {((np.isnan(slide_segment).sum() / 3750) * 100):.2f}%")
                                        continue

                                    # check if the signal is stable
                                    if self.is_any_constant_signal(slide_segment):
                                        print(f"the sequence is stable, not a signal")
                                        continue

                                    # interpolate
                                    slide_segment = self.interpolate_nan_multichannel(slide_segment)

                                    print("set nan value to zero and normalize signal")

                                    lead_ppg_segments = slide_segment[0, :]
                                    lead_ii_segments = slide_segment[1, :]


                                    # ECG quality assessment
                                    try:
                                        peaks = self.peak_detection(lead_ii_segments, original_fs, band_freq=30)
                                        print("find peaks")
                                    except ValueError as e:
                                        print(f"Warning: {e}, skipping this segment.")
                                        continue

                                    _, _, qua_ii = self.rrSQI(lead_ii_segments, peaks, original_fs)

                                    # PPG quality assessment
                                    qua_ppg = self.ppg_SQI(lead_ppg_segments, self.target_sfreq)
                                    qua_ppg = self.scale_ppg_score(qua_ppg)

                                    qua = (qua_ii + qua_ppg) / 2

                                    if qua >= 0.9:
                                        label = "Excellent"
                                    elif 0.9 < qua <= 0.7:
                                        label = "Good"
                                    elif 0.7 < qua <= 0.5:
                                        label = "Acceptable"
                                    elif 0.5 < qua <= 0.3:
                                        label = "Poor"
                                    else:
                                        label = "Bad"

                                    qua_labels.append(label)
                                    print(f"The quality in {wave_name}.npy_{i} is: {qua}")

                                    resampled_slide_segment = self.resample_waveform(original_fs, slide_segment)
                                    resampled_slide_segment = self.normalize_to_minus_one_to_one(resampled_slide_segment)

                                    slide_segments.append(resampled_slide_segment)

                                # save the segments and qualities list
                                segment_save_path = os.path.join(self.seg_save_path, subject_name, filename,
                                                                 wave_name)
                                quality_save_path = os.path.join(self.qua_save_path, subject_name, filename,
                                                                 wave_name)

                                segment_directory = os.path.dirname(segment_save_path)
                                quality_directory = os.path.dirname(quality_save_path)

                                os.makedirs(segment_directory, exist_ok=True)
                                os.makedirs(quality_directory, exist_ok=True)

                                np.save(segment_save_path, slide_segments)
                                try:
                                    np.save(quality_save_path, qua_labels)
                                except ValueError as e:
                                    print(f"Skip wrong dimension of qua_labels in {file_path}/{wave_name}")
                                    continue

                                print(f"save segments into: {segment_save_path}.npy and qualities into {quality_save_path}.npy")

    def compare_quality(self, quality_rank, q1, q2):
        return quality_rank[q1] < quality_rank[q2]

    def get_data_pair(self):
        for ecg_subject_title in tqdm(os.listdir(self.ecg_segments_path)):  # p19994379
            ecg_subject_title_path = os.path.join(self.ecg_segments_path, ecg_subject_title)
            for ecg_subject_name in os.listdir(ecg_subject_title_path):  # 87407093
                ecg_subject_path = os.path.join(ecg_subject_title_path, ecg_subject_name)

                # display process bar
                for ecg_segments_name in os.listdir(ecg_subject_path):  # 87407093.npy

                    # read segments.npy
                    ecg_slide_segments_path = os.path.join(ecg_subject_path, ecg_segments_name)
                    segments = np.load(ecg_slide_segments_path)

                    # read corresponding quality label
                    quality_path = os.path.join(self.qualities_path, ecg_subject_title, ecg_subject_name,
                                                ecg_segments_name)
                    qualities = np.load(quality_path)

                    # Find surrounding 5min segments
                    n = len(segments)
                    for i in range(n):
                        for j in range(i + 1, min(i + 10, n)):
                            # if two samples qualities are the same, skip this pair
                            if qualities[i] == qualities[j]:
                                print("The quality scales are the same")
                                continue

                            if segments[i].size == 0 or segments[j].size == 0:
                                print("There is no value, skip....")
                                continue

                            # # save pairs in dict value, and key is according to diff

                            # if ith-quality is better than j-th quality, then save [segments[i], segments[j]]
                            # Reversly, if j-th is better, save [segments[j], segments[i]]
                            # the first segment is relatively good signal, and second is bad.
                            if self.compare_quality(self.quality_rank, qualities[i], qualities[j]):
                                pair = [segments[i], segments[j]]
                            else:
                                pair = [segments[j], segments[i]]

                            file_name = f"{ecg_segments_name}_pair_{i}_{j}.npy"
                            file_path = os.path.join(self.pairs_save_path, file_name)
                            np.save(file_path, pair)




class VitaldbProcessor(BaseProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # self.required_sigs = ['II', 'PLETH']
        # self.req_seg_duration = 300
        self.slide_segment_time = 30
        self.original_fs = 300
        self.nan_limit = 0.2

    def process_record(self):
        dataset_path = self.raw_data_path
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            vf = vitaldb.VitalFile(file_path)
            ECG_ii = vf.to_numpy(['SNUADC/ECG_II'], 1 / self.original_fs)
            ppg = vf.to_numpy(['SNUADC/PLETH'], 1 / self.original_fs)

            if len(ECG_ii) ==  len(ppg):
                time = len(ECG_ii) // self.original_fs
            else:
                time = min(len(ECG_ii) // self.original_fs, len(ppg) // self.original_fs)

            multi_channel_signal = np.stack((ppg, ECG_ii), axis=0)

            # setting
            slide_segment_length = self.slide_segment_time * self.original_fs
            slide_segments = []
            qua_labels = []

            # divide into 30 sec, and discard the last slide (<30s)
            for i, start in enumerate(range(0, len(ppg) - slide_segment_length + 1, slide_segment_length)):
                end = start + slide_segment_length
                slide_segment = np.squeeze(multi_channel_signal[:, start:end])

                # check if too much nan value
                if self.is_nan_ratio_exceed_any(slide_segment, self.nan_limit, self.original_fs,
                                                self.slide_segment_time):
                    print(f"too much missing value, nan ratio is {((np.isnan(slide_segment).sum() / 3750) * 100):.2f}%")
                    continue

                # check if the signal is stable
                if self.is_any_constant_signal(slide_segment):
                    print(f"the sequence is stable, not a signal")
                    continue

                # interpolate
                slide_segment = self.interpolate_nan_multichannel(slide_segment)
                print("set nan value to zero and normalize signal")

                lead_ppg_segments = slide_segment[0, :]
                lead_ii_segments = slide_segment[1, :]

                # ECG quality assessment
                try:
                    peaks = self.peak_detection(lead_ii_segments, self.original_fs)
                    print("find peaks")
                except ValueError as e:
                    print(f"Warning: {e}, skipping this segment.")
                    continue

                _, _, qua_ii = self.rrSQI(lead_ii_segments, peaks, self.original_fs)

                # PPG quality assessment
                qua_ppg = self.ppg_SQI(lead_ppg_segments, self.original_fs)
                qua_ppg = self.scale_ppg_score(qua_ppg)

                qua = (qua_ii + qua_ppg) / 2

                if qua >= 0.9:
                    label = "Excellent"
                elif 0.9 < qua <= 0.7:
                    label = "Good"
                elif 0.7 < qua <= 0.5:
                    label = "Acceptable"
                elif 0.5 < qua <= 0.3:
                    label = "Poor"
                else:
                    label = "Bad"

                # Todo: give classification to qua
                qua_labels.append(label)
                print(f"The quality in {filename}.npy_{i} is: {qua}")

                slide_segment = self.normalize_to_minus_one_to_one(slide_segment)

                slide_segments.append(slide_segment)

            # save the segments and qualities list
            segment_save_path = self.seg_save_path + '/' + filename
            os.makedirs(os.path.dirname(segment_save_path), exist_ok=True)

            quality_save_path = self.qua_save_path + '/' + filename
            os.makedirs(os.path.dirname(quality_save_path), exist_ok=True)

            np.save(segment_save_path, slide_segments)
            try:
                np.save(quality_save_path, qua_labels)
            except ValueError as e:
                print(f"Skip wrong dimension of qua_labels in {filename}")
                continue

            print(f"save segments into: {segment_save_path}.npy and qualities into {quality_save_path}.npy")

    def compare_quality(self, quality_rank, q1, q2):
        return quality_rank[q1] < quality_rank[q2]

    def get_data_pair(self):
        # display process bar
        for ecg_segments_name in tqdm(os.listdir(self.ecg_segments_path)):
            # read segments.npy
            ecg_slide_segments_path = os.path.join(self.ecg_segments_path, ecg_segments_name)
            segments = np.load(ecg_slide_segments_path)

            # read corresponding quality label
            quality_path = os.path.join(self.qualities_path, ecg_segments_name)
            qualities = np.load(quality_path)

            # Find surrounding 5min segments
            n = len(segments)
            for i in range(n):
                for j in range(i + 1, min(i + 10, n)):
                    # if two samples qualities are the same, skip this pair
                    if qualities[i] == qualities[j]:
                        continue

                    if segments[i].size == 0 or segments[j].size == 0:
                        print("There is no value, skip....")
                        continue

                    if qualities[i] == 0 or qualities[j] == 0:
                        continue

                    # # save pairs in dict value, and key is according to diff

                    # if ith-quality is better than j-th quality, then save [segments[i], segments[j]]
                    # Reversly, if j-th is better, save [segments[j], segments[i]]
                    # the first segment is relatively good signal, and second is bad.
                    if self.compare_quality(self.quality_rank, qualities[i], qualities[j]):
                        pair = [segments[i], segments[j]]
                    else:
                        pair = [segments[j], segments[i]]

                    file_name = f"{ecg_segments_name}_pair_{i}_{j}.npy"
                    file_path = os.path.join(self.pairs_save_path, file_name)
                    np.save(file_path, pair)