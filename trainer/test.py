import torch
import torch.nn as nn


class FourierSpectrumProcessor(nn.Module):
    def __init__(self, target_sequence_length=1000, downsample_method='slice'):
        """
        初始化傅里叶频谱处理器。

        Args:
            target_sequence_length (int): 目标序列长度，默认为 1000。
            downsample_method (str): 下采样方法，'slice'（切片）或 'pool'（池化）。
                                     切片通常用于选择低频部分。
        """
        super(FourierSpectrumProcessor, self).__init__()
        self.target_sequence_length = target_sequence_length
        self.downsample_method = downsample_method

        if self.downsample_method == 'pool':
            # 如果选择池化，计算合适的 kernel_size 和 stride
            # 假设原始长度是 9000，目标是 1000，所以池化因子是 9000 / 1000 = 9
            pool_factor = 9000 // self.target_sequence_length
            if 9000 % self.target_sequence_length != 0:
                raise ValueError("对于 'pool' 方法，原始序列长度必须是目标序列长度的整数倍。")
            self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)
            # 也可以使用 nn.AvgPool1d

    def std_norm(self, x):
        """
        对输入张量进行均值-标准差标准化。
        """
        # 计算均值和标准差时，只在最后一个维度（序列长度维度）上进行，
        # 并保持维度，以便进行广播操作。
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 避免除以零，添加一个小的 epsilon
        normalized_x = (x - mean) / (std + 1e-6)
        return normalized_x

    def forward(self, x):
        """
        计算信号的傅里叶频谱，进行下采样，并对幅度和相位进行标准化。

        Args:
            x (torch.Tensor): 输入信号，形状为 (batch, channels, sequence_length)。
                              期望形状为 (batch, 2, 9000)。

        Returns:
            tuple: 包含两个标准化后的张量 (normalized_amplitude, normalized_phase)，
                   它们的形状都将是 (batch, 2, target_sequence_length)。
        """
        # 输入形状检查，确保符合预期
        expected_seq_len = 9000
        if x.dim() != 3 or x.shape[1] != 2 or x.shape[2] != expected_seq_len:
            raise ValueError(
                f"输入张量形状不正确。期望 (batch, 2, {expected_seq_len})，但收到 {x.shape}"
            )

        # --- 1. 傅里叶频谱计算 (全长) ---
        # 对最后一个维度（序列长度）进行FFT
        x_fft = torch.fft.fft(x, dim=-1)

        # 计算幅度和相位
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # --- 2. 频谱下采样 ---
        if self.downsample_method == 'slice':
            # 切片：直接截取前 target_sequence_length 个频率分量
            # 这通常对应于信号的低频部分，因为FFT的输出频率是按顺序排列的。
            amplitude = amplitude[:, :, :self.target_sequence_length]
            phase = phase[:, :, :self.target_sequence_length]
        elif self.downsample_method == 'pool':
            # 池化：在应用池化层之前，将维度从 (B, L, C) 转换为 (B, C, L)，
            # 因为nn.MaxPool1d期望通道维度在第二个位置。
            # 这里输入给ConvTranspose1d/Conv1d的都是(B, C, L)，所以这里也保持(B,C,L)

            # 确保在池化前张量的形状是 (batch, channels, sequence_length)
            # amplitude 和 phase 已经是 (batch, channels, sequence_length) 格式
            amplitude = self.pool(amplitude)
            phase = self.pool(phase)
        else:
            raise ValueError("不支持的下采样方法。请选择 'slice' 或 'pool'。")

        # --- 3. 标准化幅度和相位 ---
        normalized_amplitude = self.std_norm(amplitude)
        normalized_phase = self.std_norm(phase)

        return normalized_amplitude, normalized_phase


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建一个随机的输入信号，模拟 batch_size 为 4
    batch_size = 4
    channels = 2
    sequence_length = 9000
    dummy_input = torch.randn(batch_size, channels, sequence_length, dtype=torch.float32)

    print(f"原始输入信号形状: {dummy_input.shape}")

    # 实例化傅里叶频谱处理器，目标长度为 1000，使用切片下采样
    processor_slice = FourierSpectrumProcessor(target_sequence_length=1000, downsample_method='slice')
    normalized_amp_slice, normalized_phase_slice = processor_slice(dummy_input)

    print(f"\n使用 'slice' 方法：")
    print(f"标准化幅度谱形状: {normalized_amp_slice.shape}")
    print(f"标准化相位谱形状: {normalized_phase_slice.shape}")

    # 实例化傅里叶频谱处理器，目标长度为 1000，使用池化下采样
    processor_pool = FourierSpectrumProcessor(target_sequence_length=1000, downsample_method='pool')
    normalized_amp_pool, normalized_phase_pool = processor_pool(dummy_input)

    print(f"\n使用 'pool' 方法：")
    print(f"标准化幅度谱形状: {normalized_amp_pool.shape}")
    print(f"标准化相位谱形状: {normalized_phase_pool.shape}")

    # 尝试错误形状的输入
    try:
        wrong_input = torch.randn(2, 1, 1000)
        processor_slice(wrong_input)
    except ValueError as e:
        print(f"\n捕获到错误 (输入形状不正确): {e}")