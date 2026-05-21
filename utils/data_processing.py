import random
import numpy as np
import torch

class NoiseTransformation(object):
    """
    向数据中添加高斯噪声。
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
        return X + noise

class SubAnomaly(object):
    """
    一个复杂的异常注入器，可以生成多样化、多强度的异常。
    """
    def __init__(self, portion_len):
        self.portion_len = portion_len

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int = None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None):
        """
        底层的通用异常注入函数，保持不变。
        """
        window = window.copy()
        if subsequence_length is None:
            min_len, max_len = int(window.shape[0] * 0.2), int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max(min_len + 1, max_len)) if min_len < max_len else min_len
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0)
        if start_index is None:
            start_index = np.random.randint(0, max(1, len(window) - subsequence_length))

        end_index = min(start_index + subsequence_length, window.shape[0])
        if trend_end:
            end_index = window.shape[0]

        anomalous_subsequence = window[start_index:end_index]
        if len(anomalous_subsequence) == 0:
            return window

        anomalous_subsequence = np.repeat(anomalous_subsequence, compression_factor, axis=0)[::compression_factor]
        anomalous_subsequence = anomalous_subsequence * scale_factor

        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1 if np.random.uniform() < 0.5 else -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence), window.shape[1]) * 0.1)


        replace_len = min(len(anomalous_subsequence), window.shape[0] - start_index)
        window[start_index : start_index + replace_len] = anomalous_subsequence[:replace_len]

        return window

    def __call__(self, X):
        """
        通过随机组合多种异常类型和参数，并引入“困难度”选择，
        生成一个从“困难(微妙)”到“简单(明显)”的异常谱系。
        """
        window = X.copy()

        is_1d = False
        if window.ndim == 1:
            is_1d = True
            window = window.reshape(-1, 1)

        anomalous_window = window.copy()
        num_features = window.shape[1]
        window_len = window.shape[0]


        anomaly_types = ['seasonal', 'trend', 'global', 'contextual', 'shapelet', 'drift', 'variance']
        num_anomalies_to_inject = random.choices([1, 2, 3], weights=[0.5, 0.4, 0.1], k=1)[0]
        chosen_types = random.choices(anomaly_types, k=num_anomalies_to_inject)


        for anomaly_type in chosen_types:
            num_dims = np.random.randint(1, max(2, int(num_features / 4) + 1))
            dims_to_inject = np.random.choice(num_features, num_dims, replace=False)


            difficulty = 'hard' if random.random() < 0.5 else 'easy'


            if difficulty == 'hard':
                min_len, max_len = int(window_len * 0.05), int(window_len * 0.2)
                global_scale_range = (1.5, 3.0)
                contextual_scale_range = (1.2, 2.5)
                trend_factor_range = (0.1, 0.5)
                drift_magnitude = np.random.uniform(0.05, 0.2) * (np.max(window) - np.min(window))
                variance_scale = np.random.uniform(1.5, 3.0)
            else:
                min_len, max_len = int(window_len * 0.2), int(window_len * 0.7)
                global_scale_range = (3.0, 10.0)
                contextual_scale_range = (2.0, 5.0)
                trend_factor_range = (0.5, 2.0)
                drift_magnitude = np.random.uniform(0.2, 0.5) * (np.max(window) - np.min(window))
                variance_scale = np.random.uniform(3.0, 10.0)

            subsequence_length = np.random.randint(min_len, max(min_len + 1, max_len)) if min_len < max_len else min_len
            if subsequence_length == 0: continue
            start_index = np.random.randint(0, max(1, window_len - subsequence_length))


            params = {}
            if anomaly_type == 'seasonal':
                params = {'scale_factor': 1, 'trend_factor': 0, 'compression_factor': np.random.randint(2, 4)}
            elif anomaly_type == 'trend':
                params = {'compression_factor': 1, 'scale_factor': 1, 'trend_end': True, 'trend_factor': np.random.uniform(*trend_factor_range)}
            elif anomaly_type == 'global':
                subsequence_length = np.random.randint(2, 6)
                start_index = np.random.randint(0, max(1, window_len - subsequence_length))
                params = {'subsequence_length': subsequence_length, 'compression_factor': 1, 'scale_factor': np.random.uniform(*global_scale_range), 'trend_factor': 0}
            elif anomaly_type == 'contextual':
                min_sub_len, max_sub_len = (3, 8) if difficulty == 'hard' else (5, 15)
                subsequence_length = np.random.randint(min_sub_len, max(min_sub_len + 1, max_sub_len))
                start_index = np.random.randint(0, max(1, window_len - subsequence_length))
                params = {'subsequence_length': subsequence_length, 'compression_factor': 1, 'scale_factor': np.random.uniform(*contextual_scale_range), 'trend_factor': 0}
            elif anomaly_type == 'shapelet':
                params = {'compression_factor': 1, 'scale_factor': 1, 'trend_factor': 0, 'shapelet_factor': True}


            elif anomaly_type == 'drift':
                end_index = min(start_index + subsequence_length, window_len)
                coef = 1 if np.random.uniform() < 0.5 else -1
                anomalous_window[start_index:end_index, dims_to_inject] += coef * drift_magnitude
                continue

            elif anomaly_type == 'variance':
                end_index = min(start_index + subsequence_length, window_len)
                sub_window = anomalous_window[start_index:end_index, dims_to_inject]
                if sub_window.size > 0:
                    noise = np.random.normal(0, np.std(sub_window, axis=0) * variance_scale, sub_window.shape)
                    anomalous_window[start_index:end_index, dims_to_inject] += noise
                continue


            for dim_idx in dims_to_inject:
                temp_win_dim = anomalous_window[:, dim_idx].reshape(-1, 1)

                final_params = params.copy()
                final_params['window'] = temp_win_dim
                final_params['subsequence_length'] = params.get('subsequence_length', subsequence_length)
                final_params['start_index'] = start_index

                modified_dim = self.inject_frequency_anomaly(**final_params)
                anomalous_window[:, dim_idx] = modified_dim.flatten()

        if is_1d:
            return anomalous_window.flatten()

        return anomalous_window


class DataAugmentation:
    """
    数据增强主类，调用SubAnomaly来生成异常样本。
    """
    def __init__(self, config):
        self.num_anomalies = config['anomaly']['num_anomalies_per_sample']
        self.noise_sigma = config['anomaly']['noise_sigma']
        self.noise_transform = NoiseTransformation(self.noise_sigma)


        window_size = 100
        if 'data' in config and 'window_size' in config['data']:
            window_size = config['data']['window_size']

        self.anomaly_transform = SubAnomaly(portion_len=window_size)

    def inject_anomalies(self, window):
        """
        为单个窗口注入指定数量的异常。
        """
        anomalous_samples = []
        for _ in range(self.num_anomalies):
            anomalous_window = self.anomaly_transform(window)
            anomalous_samples.append(anomalous_window)
        return anomalous_samples

    def __call__(self, X_batch):
        """
        处理一个批次的窗口，为每个窗口生成异常样本。
        Args:
            X_batch (np.ndarray): 一批窗口数据，形状 [batch_size, window_size, num_features].
        """
        all_anomalous_samples = []
        for window in X_batch:

            anomalous_for_window = self.inject_anomalies(window)
            all_anomalous_samples.extend(anomalous_for_window)


        return None, all_anomalous_samples
