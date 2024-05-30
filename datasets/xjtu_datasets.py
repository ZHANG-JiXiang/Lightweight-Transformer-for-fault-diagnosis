# import torch
# from torch.utils.data import Dataset
# from datasets.preprocess import max_min_function, load_xjtu_data
# import numpy as np
#
#
# class XJTU_Datasets(Dataset):
#     def __init__(self, work_dir, size, step, length):
#         self.data_info = load_xjtu_data(work_dir, size, step, length, preprocess_function=max_min_function)
#
#     def __len__(self):
#         return len(self.data_info)
#
#
#     def __getitem__(self, item):
#         vibration, label = self.data_info[item]
#         vibration = vibration.astype(float)
#         vibration = torch.tensor(vibration, dtype=torch.float)
#         label = torch.tensor(label, dtype=torch.long)
#         return vibration, label
#
#
import torch
from torch.utils.data import Dataset
from datasets.preprocess import max_min_function, load_xjtu_data
import numpy as np


class XJTU_Datasets(Dataset):
    def __init__(self, work_dir, size, step, length, snr_db=-10):
        self.data_info = load_xjtu_data(work_dir, size, step, length, preprocess_function=max_min_function)
        self.snr_db = snr_db

    def __len__(self):
        return len(self.data_info)

    def add_noise(self, signal):
        # 计算信号功率
        signal_power = np.mean(np.abs(signal) ** 2)

        # 计算噪声功率
        snr = 10 ** (self.snr_db / 10)  # 将信噪比（单位dB）转换为线性信噪比
        noise_power = signal_power / snr

        # 生成高斯噪声
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

        # 添加噪声后的信号
        noisy_signal = signal + noise
        return noisy_signal

    def __getitem__(self, item):
        vibration, label = self.data_info[item]
        vibration = vibration.astype(float)

        # 给vibration添加高斯噪声
        noisy_vibration = self.add_noise(vibration)

        vibration = torch.tensor(noisy_vibration, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return vibration, label

