import os
import json
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


ECG_SERIES_NAMES = [
    'chfdb_chf01_275.pkl', 'chfdb_chf13_45590.pkl', 'chfdbchf15.pkl',
    'ltstdb_20221_43.pkl', 'ltstdb_20321_240.pkl', 'mitdb__100_180.pkl',
    'qtdbsel102.pkl', 'stdb_308_0.pkl', 'xmitdb_x108_0.pkl'
]

UCR_SERIES_NAMES = ['135_', '136_', '137_', '138_']

MSL_SERIES_NAMES = [
    'D-16', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 'P-15', 'C-1', 'T-12',
    'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14', 'T-8', 'P-11', 'D-15',
    'M-7', 'F-8', 'M-5', 'M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'C-2'
]

SMAP_SERIES_NAMES = [
    'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
    'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2',
    'P-3', 'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2',
    'D-5', 'D-6', 'D-7', 'F-1', 'P-4', 'T-1', 'T-2', 'D-8',
    'D-9', 'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7',
    'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9', 'F-3'
]

SMD_SERIES_NAMES = [
    'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4',
    'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-1-8',
    'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4',
    'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8',
    'machine-2-9', 'machine-3-1', 'machine-3-2', 'machine-3-3',
    'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7',
    'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11'
]


def get_dataset_series_names(dataset, data_path=None):
    del data_path
    dataset_norm = str(dataset).lower()

    if dataset_norm in {'swat', 'wadi', 'psm', 'msds', 'pd', 'gesture', 'nab'}:
        return [str(dataset)]
    if dataset_norm == 'ecg':
        return list(ECG_SERIES_NAMES)
    if dataset_norm == 'ucr':
        return list(UCR_SERIES_NAMES)
    if dataset_norm == 'kpi':
        return ['kpi_0']
    if dataset_norm == 'smd':
        return list(SMD_SERIES_NAMES)
    if dataset_norm == 'msl':
        return list(MSL_SERIES_NAMES)
    if dataset_norm == 'smap':
        return list(SMAP_SERIES_NAMES)
    if dataset_norm == 'credit':
        raise NotImplementedError("credit dataset loader is not implemented in the original project.")
    raise ValueError('this dataset is not supported')


discover_dataset_series = get_dataset_series_names


class SlidingWindowTrainDataset(Dataset):
    def __init__(self, data, window_size, step_size=1, starts=None, return_indices=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.return_indices = bool(return_indices)

        if starts is None:
            self.starts = _build_window_starts(self.data.shape[1], self.window_size, self.step_size)
        else:
            self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return int(self.starts.shape[0])

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = np.ascontiguousarray(self.data[:, s:s + self.window_size], dtype=np.float32)
        x = torch.from_numpy(x)
        if self.return_indices:
            return x, x, int(idx)
        return x, x


class SlidingWindowEvalDataset(Dataset):
    def __init__(self, data, label, window_size, step_size=1, starts=None, return_indices=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.label = np.asarray(label).reshape(-1)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.return_indices = bool(return_indices)

        if starts is None:
            self.starts = _build_window_starts(self.data.shape[1], self.window_size, self.step_size)
        else:
            self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return int(self.starts.shape[0])

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = np.ascontiguousarray(self.data[:, s:s + self.window_size], dtype=np.float32)
        y = np.ascontiguousarray(self.label[s:s + self.window_size].reshape(1, -1), dtype=np.float32)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.return_indices:
            return x, y, int(idx)
        return x, y


def _build_window_starts(total_length, window_size, step_size):
    total_length = int(total_length)
    window_size = int(window_size)
    step_size = int(step_size)
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if total_length < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, total_length - window_size + 1, step_size, dtype=np.int64)


def _build_window_pos_mask(test_label, window_size, step_size):
    y = np.asarray(test_label).reshape(-1)
    y = (y > 0).astype(np.int32)

    starts = _build_window_starts(len(y), window_size, step_size)
    if starts.size == 0:
        return starts, np.empty((0,), dtype=bool)

    prefix = np.zeros(len(y) + 1, dtype=np.int64)
    prefix[1:] = np.cumsum(y, dtype=np.int64)
    pos_mask = (prefix[starts + window_size] - prefix[starts]) > 0
    return starts, pos_mask


def _find_true_segments(bool_mask: np.ndarray):
    segs = []
    N = int(bool_mask.shape[0])
    i = 0
    while i < N:
        if bool_mask[i]:
            j = i + 1
            while j < N and bool_mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def _split_val_test_indices(pos_mask, val_ratio, seed=42):
    N = int(len(pos_mask))
    if N == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    split_count = int(N * float(val_ratio))
    split_count = max(1, split_count) if (N > 1 and val_ratio > 0) else split_count

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]

    rng = np.random.default_rng(seed)
    val_pos_n = min(int(len(pos_idx) * float(val_ratio)), len(pos_idx))
    val_neg_n = min(max(0, split_count - val_pos_n), len(neg_idx))

    val_pos = rng.choice(pos_idx, size=val_pos_n, replace=False) if val_pos_n > 0 else np.array([], dtype=np.int64)
    val_neg = rng.choice(neg_idx, size=val_neg_n, replace=False) if val_neg_n > 0 else np.array([], dtype=np.int64)

    val_idx = np.sort(np.concatenate([val_pos, val_neg])).astype(np.int64)
    test_idx = np.setdiff1d(np.arange(N, dtype=np.int64), val_idx)
    return val_idx, test_idx


def _split_val_test_wadi_indices(pos_mask, val_ratio, preserve_segments=True, seed=42):
    N = int(len(pos_mask))
    if N == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    pos_mask = np.asarray(pos_mask, dtype=bool)
    pos_idx_all = np.where(pos_mask)[0]
    neg_idx_all = np.where(~pos_mask)[0]
    pos_count = int(pos_idx_all.size)

    n_val = int(max(1, round(N * float(val_ratio))))
    n_pos_target = int(round(n_val * (pos_count / max(1, N))))
    n_pos_target = max(0, min(n_pos_target, min(pos_count, n_val)))
    n_neg_target = n_val - n_pos_target

    rng = np.random.default_rng(seed)
    val_pos_idx = []

    if preserve_segments and pos_count > 0:
        segs = _find_true_segments(pos_mask)
        acc = 0
        for a, b in segs:
            if acc >= n_pos_target:
                break
            need = n_pos_target - acc
            seg_len = b - a
            if seg_len <= need:
                val_pos_idx.extend(range(a, b))
                acc += seg_len
            else:
                val_pos_idx.extend(range(a, a + need))
                acc += need
                break
    elif n_pos_target > 0:
        val_pos_idx = rng.choice(pos_idx_all, size=n_pos_target, replace=False).tolist()

    val_pos_idx = np.array(sorted(set(val_pos_idx)), dtype=np.int64)
    pool_neg = np.setdiff1d(neg_idx_all, val_pos_idx, assume_unique=False)
    if n_neg_target > 0:
        if pool_neg.size >= n_neg_target:
            val_neg_idx = rng.choice(pool_neg, size=n_neg_target, replace=False)
        else:
            val_neg_idx = pool_neg
            remain = n_neg_target - pool_neg.size
            pool_any = np.setdiff1d(np.arange(N, dtype=np.int64), val_pos_idx, assume_unique=False)
            extra = rng.choice(pool_any, size=remain, replace=False) if remain > 0 else np.array([], dtype=np.int64)
            val_neg_idx = np.concatenate([val_neg_idx, extra])
    else:
        val_neg_idx = np.array([], dtype=np.int64)

    val_idx = np.array(sorted(set(val_pos_idx.tolist() + val_neg_idx.tolist())), dtype=np.int64)
    if val_idx.size > n_val:
        val_idx = val_idx[:n_val]
    elif val_idx.size < n_val:
        remain = n_val - val_idx.size
        pool_rest = np.setdiff1d(np.arange(N, dtype=np.int64), val_idx, assume_unique=False)
        add_idx = rng.choice(pool_rest, size=remain, replace=False) if remain > 0 else np.array([], dtype=np.int64)
        val_idx = np.sort(np.concatenate([val_idx, add_idx]))

    test_idx = np.setdiff1d(np.arange(N, dtype=np.int64), val_idx, assume_unique=False)
    return val_idx, test_idx


def _split_val_test_indices_legacy(pos_mask, val_ratio, seed=42):
    total_samples = int(len(pos_mask))
    if total_samples == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    split_count = int(total_samples * float(val_ratio))
    pos_idx = np.where(np.asarray(pos_mask, dtype=bool))[0]
    neg_idx = np.where(~np.asarray(pos_mask, dtype=bool))[0]

    rs = np.random.RandomState(seed)
    val_pos_n = min(int(len(pos_idx) * float(val_ratio)), len(pos_idx))
    val_neg_n = min(max(0, split_count - val_pos_n), len(neg_idx))

    val_pos = rs.choice(pos_idx, size=val_pos_n, replace=False) if val_pos_n > 0 else np.array([], dtype=np.int64)
    val_neg = rs.choice(neg_idx, size=val_neg_n, replace=False) if val_neg_n > 0 else np.array([], dtype=np.int64)

    val_idx = np.sort(np.concatenate([val_pos, val_neg])).astype(np.int64)
    test_idx = np.setdiff1d(np.arange(total_samples, dtype=np.int64), val_idx, assume_unique=False)
    return val_idx, test_idx


def _safe_series_name(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "series"


def _get_split_cache_dir(data_path: str) -> str:
    split_dir = os.path.join(data_path, "_split_cache")
    os.makedirs(split_dir, exist_ok=True)
    return split_dir


def _build_split_cache_path(data_path, dataset_key, series_name, window_size, step_size, val_ratio,
                            downsample_factor=1, downsample_method="stride",
                            split_policy="legacy_stratified_window_all_datasets", split_seed=42):
    split_dir = _get_split_cache_dir(data_path)
    fname = (
        f"{_safe_series_name(dataset_key)}"
        f"__{_safe_series_name(series_name)}"
        f"__w{int(window_size)}"
        f"__s{int(step_size)}"
        f"__vr{float(val_ratio):.6f}.npz"
    )
    fname = fname[:-4] + f"__policy_{_safe_series_name(split_policy)}.npz"
    fname = fname[:-4] + f"__splitseed{int(split_seed)}.npz"
    if int(downsample_factor) > 1:
        fname = fname[:-4] + f"__ds{int(downsample_factor)}_{_safe_series_name(downsample_method)}.npz"
    return os.path.join(split_dir, fname)


def _save_split_cache(path, *, dataset_key, series_name, window_size, step_size, val_ratio,
                      all_test_starts, val_idx, test_idx, policy,
                      downsample_factor=1, downsample_method="stride", split_seed=42):
    meta = {
        "dataset_key": str(dataset_key),
        "series_name": str(series_name),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "val_ratio": float(val_ratio),
        "downsample_factor": int(downsample_factor),
        "downsample_method": str(downsample_method),
        "policy": str(policy),
        "split_seed": int(split_seed),
        "num_all_test_windows": int(len(all_test_starts)),
        "num_val_windows": int(len(val_idx)),
        "num_test_windows": int(len(test_idx)),
    }
    all_test_starts = np.asarray(all_test_starts, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        all_test_starts=all_test_starts,
        val_idx=val_idx,
        test_idx=test_idx,
        val_window_starts=all_test_starts[val_idx],
        test_window_starts=all_test_starts[test_idx],
    )


def _load_split_cache(path, all_test_starts):
    obj = np.load(path, allow_pickle=True)
    all_test_starts = np.asarray(all_test_starts, dtype=np.int64)

    if "val_idx" in obj and "test_idx" in obj:
        val_idx = np.asarray(obj["val_idx"], dtype=np.int64)
        test_idx = np.asarray(obj["test_idx"], dtype=np.int64)
        if "all_test_starts" in obj:
            old_all = np.asarray(obj["all_test_starts"], dtype=np.int64)
            if old_all.shape[0] != all_test_starts.shape[0]:
                raise RuntimeError(
                    f"split cache total windows mismatch: cache={old_all.shape[0]} current={all_test_starts.shape[0]}"
                )
        return val_idx, test_idx

    if "val_window_starts" in obj and "test_window_starts" in obj:
        val_starts = np.asarray(obj["val_window_starts"], dtype=np.int64)
        test_starts = np.asarray(obj["test_window_starts"], dtype=np.int64)
        start_to_idx = {int(s): i for i, s in enumerate(all_test_starts.tolist())}
        val_idx = np.asarray([start_to_idx[int(s)] for s in val_starts], dtype=np.int64)
        test_idx = np.asarray([start_to_idx[int(s)] for s in test_starts], dtype=np.int64)
        return val_idx, test_idx

    raise RuntimeError(f"invalid split cache file: {path}")


def _load_split_cache_policy(path, default="reuse"):
    obj = np.load(path, allow_pickle=True)
    if "meta_json" not in obj:
        return default
    try:
        meta = json.loads(str(obj["meta_json"]))
        return str(meta.get("policy", default))
    except Exception:
        return default


def _get_or_create_legacy_compatible_split(*, dataset_name_is_wadi, data_path, dataset_key, series_name,
                                           window_size, step_size, val_ratio, all_test_starts, pos_mask,
                                           downsample_factor=1, downsample_method="stride", split_seed=42):
    del dataset_name_is_wadi
    policy = "legacy_stratified_window_all_datasets"
    split_cache_path = _build_split_cache_path(
        data_path=data_path,
        dataset_key=dataset_key,
        series_name=series_name,
        window_size=window_size,
        step_size=step_size,
        val_ratio=val_ratio,
        downsample_factor=downsample_factor,
        downsample_method=downsample_method,
        split_policy=policy,
        split_seed=split_seed,
    )

    if os.path.isfile(split_cache_path):
        try:
            val_idx, test_idx = _load_split_cache(split_cache_path, all_test_starts)
            cached_policy = _load_split_cache_policy(split_cache_path, default=policy)
            print(f"[Split] reuse cached split: {split_cache_path}")
            return val_idx, test_idx, split_cache_path, cached_policy
        except Exception as e:
            print(f"[Split] cache load failed, regenerate legacy-compatible split: {e}")

    val_idx, test_idx = _split_val_test_indices_legacy(pos_mask=pos_mask, val_ratio=val_ratio, seed=split_seed)
    _save_split_cache(
        split_cache_path,
        dataset_key=dataset_key,
        series_name=series_name,
        window_size=window_size,
        step_size=step_size,
        val_ratio=val_ratio,
        all_test_starts=all_test_starts,
        val_idx=val_idx,
        test_idx=test_idx,
        policy=policy,
        downsample_factor=downsample_factor,
        downsample_method=downsample_method,
        split_seed=split_seed,
    )
    print(f"[Split] create and save split: {split_cache_path}  policy={policy}")
    return val_idx, test_idx, split_cache_path, policy


class Dataset_Loader():
    def __init__(self, dataset, data_path, window_size, ts_num, step_size, val_ratio=0.2,
                 downsample_factor=1, downsample_method="stride", split_seed=42):
        dataset_norm = str(dataset).lower()
        supported = {
            'ecg', 'ucr', 'pd', 'gesture', 'credit', 'nab', 'kpi', 'psm',
            'smd', 'msds', 'msl', 'smap', 'swat', 'wadi'
        }
        if dataset_norm not in supported:
            raise ValueError('this dataset is not supported')

        if dataset_norm == 'msl':
            self.dataset_key = 'MSL'
        elif dataset_norm == 'smap':
            self.dataset_key = 'SMAP'
        else:
            self.dataset_key = dataset_norm

        self.dataset_name_str = str(dataset)
        self.data_path = data_path
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.val_ratio = float(val_ratio)
        self.ts_num = int(ts_num)
        self.downsample_factor = int(max(1, downsample_factor))
        self.downsample_method = str(downsample_method).lower()
        self.split_seed = int(split_seed)
        if self.downsample_method not in {"stride", "mean"}:
            raise ValueError("downsample_method must be 'stride' or 'mean'")

        self.series_name = str(dataset)

        single_series_datasets = {'swat', 'wadi', 'psm', 'msds', 'pd', 'gesture', 'nab'}
        if self.dataset_key in single_series_datasets and self.ts_num != 0:
            raise IndexError(f"{self.dataset_key} is a single-series dataset; only ts_num=0 is valid.")

        if self.dataset_key == 'ecg':
            self.ecg_dataset_name = discover_dataset_series('ecg', self.data_path)
            if self.ts_num >= len(self.ecg_dataset_name):
                raise IndexError("ecg ts_num out of range")
            self.series_name = self.ecg_dataset_name[self.ts_num]

        if self.dataset_key == 'ucr':
            self.ucr_dataset_name = discover_dataset_series('ucr', self.data_path)
            if self.ts_num >= len(self.ucr_dataset_name):
                raise IndexError("ucr ts_num out of range")
            self.series_name = self.ucr_dataset_name[self.ts_num]

        if self.dataset_key == 'MSL':
            self.msl_dataset_name = discover_dataset_series('MSL', self.data_path)
            if self.ts_num >= len(self.msl_dataset_name):
                raise IndexError("MSL ts_num out of range")
            self.series_name = self.msl_dataset_name[self.ts_num]

        if self.dataset_key == 'SMAP':
            self.smap_dataset_name = discover_dataset_series('SMAP', self.data_path)
            if self.ts_num >= len(self.smap_dataset_name):
                raise IndexError("SMAP ts_num out of range")
            self.series_name = self.smap_dataset_name[self.ts_num]

        if self.dataset_key == 'smd':
            self.smd_dataset_name = discover_dataset_series('smd', self.data_path)
            if self.ts_num >= len(self.smd_dataset_name):
                raise IndexError("smd ts_num out of range")
            self.series_name = self.smd_dataset_name[self.ts_num]

        if self.dataset_key == 'kpi':
            self.series_name = f'kpi_{self.ts_num}'

        self.__get_dataset()
        self.__apply_downsampling()
        self.__build_window_indices()

    def __apply_downsampling(self):
        factor = int(self.downsample_factor)
        if factor <= 1:
            return

        def downsample_data(x):
            x = np.asarray(x, dtype=np.float32)
            if self.downsample_method == "stride":
                return x[:, ::factor].astype(np.float32, copy=False)
            usable = (x.shape[1] // factor) * factor
            if usable <= 0:
                raise ValueError(f"downsample_factor={factor} is larger than sequence length={x.shape[1]}")
            return x[:, :usable].reshape(x.shape[0], -1, factor).mean(axis=2).astype(np.float32, copy=False)

        def downsample_label(y):
            y = np.asarray(y).reshape(-1)
            if self.downsample_method == "stride":
                return y[::factor].astype(np.int64, copy=False)
            usable = (y.shape[0] // factor) * factor
            if usable <= 0:
                raise ValueError(f"downsample_factor={factor} is larger than label length={y.shape[0]}")
            return y[:usable].reshape(-1, factor).max(axis=1).astype(np.int64, copy=False)

        old_train_len = int(self.dataset["train_data"].shape[1])
        old_test_len = int(self.dataset["test_data"].shape[1])
        self.dataset["train_data"] = downsample_data(self.dataset["train_data"])
        self.dataset["test_data"] = downsample_data(self.dataset["test_data"])
        self.dataset["test_label"] = downsample_label(self.dataset["test_label"])
        if self.dataset["test_data"].shape[1] != self.dataset["test_label"].shape[0]:
            n = min(self.dataset["test_data"].shape[1], self.dataset["test_label"].shape[0])
            self.dataset["test_data"] = self.dataset["test_data"][:, :n]
            self.dataset["test_label"] = self.dataset["test_label"][:n]
        print(
            f"[Downsample] method={self.downsample_method} factor={factor} "
            f"train_len {old_train_len}->{self.dataset['train_data'].shape[1]} "
            f"test_len {old_test_len}->{self.dataset['test_data'].shape[1]}"
        )

    def __get_dataset(self):
        if self.dataset_key == 'ecg':
            self.dataset = get_ECG_dataset(self.data_path, self.ecg_dataset_name, self.ts_num, normalized=True)
        elif self.dataset_key == 'pd':
            self.dataset = get_Power_Demand_dataset(data_path=self.data_path)
        elif self.dataset_key == 'gesture':
            self.dataset = get_Gesture_dataset(data_path=self.data_path)
        elif self.dataset_key == 'nab':
            self.dataset = get_NAB_dataset(data_path=self.data_path)
        elif self.dataset_key == 'ucr':
            self.dataset = get_UCR_dataset(self.data_path, self.ucr_dataset_name, self.ts_num, normalized=True)
        elif self.dataset_key == 'kpi':
            self.dataset = get_KPI_dataset(self.data_path, self.ts_num, normalized=True)
        elif self.dataset_key == 'psm':
            self.dataset = get_PSM_dataset(data_path=self.data_path, normalized=True)
        elif self.dataset_key == 'swat':
            self.dataset = get_SWaT_dataset(data_path=self.data_path, normalized=True)
        elif self.dataset_key == 'wadi':
            self.dataset = get_WADI_dataset(data_path=self.data_path, normalized=True)
        elif self.dataset_key == 'msds':
            self.dataset = get_MSDS_dataset(data_path=self.data_path, normalized=True)
        elif self.dataset_key == 'smd':
            self.dataset = get_SMD_dataset(self.data_path, self.ts_num, self.smd_dataset_name, normalized=True)
        elif self.dataset_key == 'MSL':
            self.dataset = get_SMAP_MSL_dataset(self.data_path, self.ts_num, self.msl_dataset_name, normalized=True)
        elif self.dataset_key == 'SMAP':
            self.dataset = get_SMAP_MSL_dataset(self.data_path, self.ts_num, self.smap_dataset_name, normalized=True)
        elif self.dataset_key == 'credit':
            raise NotImplementedError("credit dataset loader is not implemented in the original project.")
        else:
            raise ValueError(f"unsupported dataset key: {self.dataset_key}")

    def __build_window_indices(self):
        train_data = self.dataset['train_data']
        test_label = self.dataset['test_label']

        self.train_window_starts = _build_window_starts(train_data.shape[1], self.window_size, self.step_size)
        all_test_starts, pos_mask = _build_window_pos_mask(test_label, self.window_size, self.step_size)

        val_idx, test_idx, split_cache_path, split_policy = _get_or_create_legacy_compatible_split(
            dataset_name_is_wadi=self.dataset_name_is_wadi(),
            data_path=self.data_path,
            dataset_key=self.dataset_key,
            series_name=self.series_name,
            window_size=self.window_size,
            step_size=self.step_size,
            val_ratio=self.val_ratio,
            all_test_starts=all_test_starts,
            pos_mask=pos_mask,
            downsample_factor=self.downsample_factor,
            downsample_method=self.downsample_method,
            split_seed=self.split_seed,
        )

        self.split_cache_path = split_cache_path
        self.split_policy = split_policy
        self.val_window_starts = all_test_starts[val_idx]
        self.test_window_starts = all_test_starts[test_idx]

        self.num_train_windows = int(self.train_window_starts.shape[0])
        self.num_val_windows = int(self.val_window_starts.shape[0])
        self.num_test_windows = int(self.test_window_starts.shape[0])

        self.train_set = {
            'samples': np.zeros((self.num_train_windows, 1, 1), dtype=np.uint8),
            'labels': np.zeros((self.num_train_windows, 1, 1), dtype=np.uint8),
            'indices': self.train_window_starts,
            'num_windows': self.num_train_windows
        }

        val_pos_flags = pos_mask[val_idx].astype(np.uint8) if val_idx.size > 0 else np.zeros((0,), dtype=np.uint8)
        test_pos_flags = pos_mask[test_idx].astype(np.uint8) if test_idx.size > 0 else np.zeros((0,), dtype=np.uint8)

        self.val_set = {
            'samples': np.zeros((self.num_val_windows, 1, 1), dtype=np.uint8),
            'labels': val_pos_flags.reshape(-1, 1, 1),
            'indices': self.val_window_starts,
            'num_windows': self.num_val_windows
        }
        self.test_set = {
            'samples': np.zeros((self.num_test_windows, 1, 1), dtype=np.uint8),
            'labels': test_pos_flags.reshape(-1, 1, 1),
            'indices': self.test_window_starts,
            'num_windows': self.num_test_windows
        }

        val_pos = int(val_pos_flags.sum()) if val_pos_flags.size > 0 else 0
        test_pos = int(test_pos_flags.sum()) if test_pos_flags.size > 0 else 0
        print(f"[Split check] val_windows={self.num_val_windows} (pos={val_pos}), "
              f"test_windows={self.num_test_windows} (pos={test_pos})")
        print(f"[Split info] policy={self.split_policy} cache={self.split_cache_path}")

    def dataset_name_is_wadi(self):
        return str(self.dataset_name_str).lower() == 'wadi' or self.dataset_key == 'wadi'

    def train_loader_generation(self, batch_size, shuffle=True, seed=None, num_workers=0,
                                pin_memory=True, persistent_workers=False, drop_last=False,
                                return_indices=False):
        ds = SlidingWindowTrainDataset(
            data=self.dataset['train_data'],
            window_size=self.window_size,
            step_size=self.step_size,
            starts=self.train_window_starts,
            return_indices=return_indices
        )

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            generator=generator
        )

    def val_loader_generation(self, batch_size, shuffle=False, num_workers=0,
                              pin_memory=True, persistent_workers=False, drop_last=False,
                              return_indices=False):
        ds = SlidingWindowEvalDataset(
            data=self.dataset['test_data'],
            label=self.dataset['test_label'],
            window_size=self.window_size,
            step_size=self.step_size,
            starts=self.val_window_starts,
            return_indices=return_indices
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

    def test_loader_generation(self, batch_size, shuffle=False, num_workers=0,
                               pin_memory=True, persistent_workers=False, drop_last=False,
                               return_indices=False):
        ds = SlidingWindowEvalDataset(
            data=self.dataset['test_data'],
            label=self.dataset['test_label'],
            window_size=self.window_size,
            step_size=self.step_size,
            starts=self.test_window_starts,
            return_indices=return_indices
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

    def val_test_loader_generation(self, batch_size, shuffle=False, num_workers=0,
                                   pin_memory=True, persistent_workers=False, drop_last=False,
                                   return_indices=False):
        return self.test_loader_generation(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            return_indices=return_indices
        )

    def plot(self, data_type='train', dataset_name=None):
        data = None
        labels = None
        if data_type == 'train':
            data = self.dataset['train_data']
        elif data_type == 'val':
            data = self.dataset.get('validate_data', self.dataset['test_data'])
            labels = self.dataset.get('validate_label', self.dataset['test_label'])
        elif data_type == 'test':
            data = self.dataset['test_data']
            labels = self.dataset['test_label']
        else:
            raise ValueError("data_type must be 'train', 'val', or 'test'.")

        for i in range(data.shape[0]):
            plt.plot(data[i])

        if data_type in {'val', 'test'} and labels is not None:
            start = 0
            end = 0
            for k in range(len(labels) - 1):
                if labels[k] == 0. and labels[k + 1] == 1:
                    start = k + 1
                if labels[k] == 1. and labels[k + 1] == 0:
                    end = k
            plt.axvspan(start, end, 0., 1., alpha=0.5, color='lightgreen')

        plt.title(str(data_type) + ' data of dataset ' + str(dataset_name))
        plt.xlabel('time point')
        plt.ylabel('values')
        plt.show()

        if data_type in {'val', 'test'} and labels is not None:
            plt.plot(labels)
            plt.xlabel('time point')
            plt.ylabel('anomaly labels')
            plt.title(str(data_type) + ' label')
            plt.show()

    def statistics(self):
        label = np.asarray(self.dataset['test_label']).reshape(-1)
        n_anomalies = int((label > 0).sum())
        perent_anomalies = n_anomalies / max(1, len(label))
        return self.dataset['train_data'].shape, self.dataset['test_data'].shape, perent_anomalies


def _fill_invalid_2d_adjacent(x_ct: np.ndarray) -> np.ndarray:
    assert x_ct.ndim == 2, "expected [C, T]"
    x = x_ct.copy()
    if np.isfinite(x).all():
        return x

    x[~np.isfinite(x)] = np.nan
    C, T = x.shape
    for c in range(C):
        col = x[c]
        last = np.nan
        for t in range(T):
            if np.isnan(col[t]):
                col[t] = last
            else:
                last = col[t]
        last = np.nan
        for t in range(T - 1, -1, -1):
            if np.isnan(col[t]):
                col[t] = last
            else:
                last = col[t]
        col[np.isnan(col)] = 0.0
        x[c] = col
    return x


def get_WADI_dataset(data_path: str, adjacent_fill: bool = True, normalized=True):
    train_path = os.path.join(data_path, "train_data.npy")
    test_path = os.path.join(data_path, "test_data.npy")
    label_path = os.path.join(data_path, "test_label.npy")

    Xtr_mm = np.load(train_path, mmap_mode="r", allow_pickle=False)
    Xte_mm = np.load(test_path, mmap_mode="r", allow_pickle=False)
    y_mm = np.load(label_path, mmap_mode="r", allow_pickle=False)
    print(f"loaded: train {Xtr_mm.shape}, test {Xte_mm.shape}, label {y_mm.shape}")

    train_ct = np.asarray(Xtr_mm.T, dtype=np.float32)
    test_ct = np.asarray(Xte_mm.T, dtype=np.float32)

    if adjacent_fill:
        train_ct = _fill_invalid_2d_adjacent(train_ct)
        test_ct = _fill_invalid_2d_adjacent(test_ct)

    y = np.asarray(y_mm).reshape(-1)
    if y.dtype.kind != 'i':
        y = y.astype(np.int64, copy=False)
    y = np.clip(y, 0, 1).astype(np.int64, copy=False)

    meta = {}
    feat_path = os.path.join(data_path, "feature_names.npy")
    mean_path = os.path.join(data_path, "train_mean.npy")
    std_path = os.path.join(data_path, "train_std.npy")
    if os.path.exists(feat_path):
        meta["feature_names"] = np.load(feat_path, allow_pickle=True)
    if os.path.exists(mean_path):
        meta["train_mean"] = np.load(mean_path)
    if os.path.exists(std_path):
        meta["train_std"] = np.load(std_path)

    if normalized:
        tr_mean = meta["train_mean"] if "train_mean" in meta else train_ct.mean(axis=1, keepdims=True)
        tr_std = meta["train_std"] if "train_std" in meta else (train_ct.std(axis=1, keepdims=True) + 1e-6)
        train_ct = ((train_ct - tr_mean) / tr_std).astype(np.float32, copy=False)
        test_ct = ((test_ct - tr_mean) / tr_std).astype(np.float32, copy=False)

    ds = {"train_data": train_ct, "test_data": test_ct, "test_label": y}
    if meta:
        ds["meta"] = meta
    print("WADI loaded OK.")
    print(f"Train [C,T]: {train_ct.shape}, Test [C,T]: {test_ct.shape}, Labels: {y.shape} (pos_ratio={y.mean():.4f})")
    return ds


def get_SWaT_dataset(data_path, normalized=True):
    train_path = None
    test_path = None
    label_path = None
    for file_name in os.listdir(data_path):
        if file_name.endswith('train.npy'):
            train_path = os.path.join(data_path, file_name)
        elif file_name.endswith('test.npy'):
            test_path = os.path.join(data_path, file_name)
        elif file_name.endswith('labels.npy'):
            label_path = os.path.join(data_path, file_name)

    if train_path is None or test_path is None or label_path is None:
        raise FileNotFoundError(f"SWaT dataset files not complete under: {data_path}")

    Xtr_mm = np.load(train_path, mmap_mode="r", allow_pickle=False)
    Xte_mm = np.load(test_path, mmap_mode="r", allow_pickle=False)
    y_mm = np.load(label_path, mmap_mode="r", allow_pickle=False)

    train_data = np.asarray(Xtr_mm.T, dtype=np.float32)
    test_data = np.asarray(Xte_mm.T, dtype=np.float32)
    label = np.asarray(y_mm).reshape(-1).astype(np.int64, copy=False)

    if normalized:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)
        denom = max(float(max_val - min_val), 1e-6)
        train_data = ((train_data - min_val) / denom).astype(np.float32, copy=False)
        test_data = ((test_data - min_val) / denom).astype(np.float32, copy=False)

    return {'train_data': train_data, 'test_data': test_data, 'test_label': label}


def get_SMD_dataset(data_path, ts_num, dataset_name, normalized=True):
    dataset = {}
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    label_dir = os.path.join(data_path, 'test_label')
    target = dataset_name[ts_num]

    for key, folder in [('train_data', train_dir), ('test_data', test_dir)]:
        found = False
        for file_name in os.listdir(folder):
            if file_name.startswith(target):
                dataset[key] = np.loadtxt(os.path.join(folder, file_name), delimiter=',').T.astype(np.float32)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"SMD {key} file not found for {target}")

    found = False
    for file_name in os.listdir(label_dir):
        if file_name.startswith(target):
            label = np.loadtxt(os.path.join(label_dir, file_name), delimiter=',')
            dataset['test_label'] = np.asarray([1 if float(v) == 1.0 else 0 for v in label], dtype=np.int64)
            found = True
            break
    if not found:
        raise FileNotFoundError(f"SMD label file not found for {target}")

    if normalized:
        max_val = np.nanmax(dataset['train_data'])
        min_val = np.nanmin(dataset['train_data'])
        denom = max(float(max_val - min_val), 1e-6)
        dataset['train_data'] = ((dataset['train_data'] - min_val) / denom).astype(np.float32, copy=False)
        dataset['test_data'] = ((dataset['test_data'] - min_val) / denom).astype(np.float32, copy=False)

    return dataset


def get_SMAP_MSL_dataset(data_path, ts_num, dataset_name, normalized=True):
    dataset = {}
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    target_name = dataset_name[ts_num]

    train_file = os.path.join(train_path, f"{target_name}.npy")
    if os.path.exists(train_file):
        dataset['train_data'] = np.load(train_file).T.astype(np.float32)
    else:
        raise FileNotFoundError(f"Training file for {target_name} not found.")

    test_file = os.path.join(test_path, f"{target_name}.npy")
    if os.path.exists(test_file):
        dataset['test_data'] = np.load(test_file).T.astype(np.float32)
    else:
        raise FileNotFoundError(f"Testing file for {target_name} not found.")

    label_file = os.path.join(data_path, 'labeled_anomalies.csv')
    if os.path.exists(label_file):
        label_df = pd.read_csv(label_file)
        if 'chan_id' not in label_df.columns:
            raise KeyError(
                f"Expected column 'chan_id' not found in labeled_anomalies.csv. "
                f"Available columns: {label_df.columns}"
            )

        anomaly_data = label_df[label_df['chan_id'] == target_name]['anomaly_sequences'].values
        if len(anomaly_data) > 0:
            labels = np.zeros(dataset['test_data'].shape[1], dtype=np.int64)
            for seq_str in anomaly_data:
                anomaly_ranges = ast.literal_eval(seq_str)
                for start, end in anomaly_ranges:
                    labels[start:end + 1] = 1
            dataset['test_label'] = labels
        else:
            raise ValueError(f"No anomaly data found for {target_name}.")
    else:
        raise FileNotFoundError(f"Labels file not found at {label_file}.")

    if normalized:
        max_val = np.nanmax(dataset['train_data'])
        min_val = np.nanmin(dataset['train_data'])
        denom = max(float(max_val - min_val), 1e-6)
        dataset['train_data'] = ((dataset['train_data'] - min_val) / denom).astype(np.float32, copy=False)
        dataset['test_data'] = ((dataset['test_data'] - min_val) / denom).astype(np.float32, copy=False)

    return dataset


def get_PSM_dataset(data_path, normalized=True):
    tr_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    te_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    te_label = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

    train_data = tr_data.iloc[:, 1:].to_numpy(dtype=np.float32).T
    test_data = te_data.iloc[:, 1:].to_numpy(dtype=np.float32).T
    test_label = te_label.iloc[:, -1].to_numpy().reshape(-1).astype(np.int64)

    if np.isnan(train_data).any():
        train_data = np.nan_to_num(train_data, nan=0.0)
    if np.isnan(test_data).any():
        test_data = np.nan_to_num(test_data, nan=0.0)
    if np.isnan(test_label).any():
        test_label = np.nan_to_num(test_label, nan=0).astype(np.int64)

    if normalized:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)
        denom = max(float(max_val - min_val), 1e-6)
        train_data = ((train_data - min_val) / denom).astype(np.float32, copy=False)
        test_data = ((test_data - min_val) / denom).astype(np.float32, copy=False)

    return {'train_data': train_data, 'test_data': test_data, 'test_label': test_label}


def normalize3(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def get_MSDS_dataset(data_path, normalized=True):
    folder = data_path
    dataset_folder = data_path

    df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
    df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
    df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]

    _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
    train, _, _ = normalize3(df_train, min_a, max_a)
    test, _, _ = normalize3(df_test, min_a, max_a)
    labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv')).values[::1, 1:]

    np.save(os.path.join(folder, 'train.npy'), train.astype('float32'))
    np.save(os.path.join(folder, 'test.npy'), test.astype('float32'))
    np.save(os.path.join(folder, 'labels.npy'), labels.astype('float32'))

    dataset = {
        'train_data': train.T.astype(np.float32),
        'test_data': test.T.astype(np.float32),
        'test_label': np.asarray([1 if 1. in row else 0 for row in labels], dtype=np.int64)
    }

    if normalized:
        max_val = np.nanmax(dataset['train_data'])
        min_val = np.nanmin(dataset['train_data'])
        denom = max(float(max_val - min_val), 1e-6)
        dataset['train_data'] = ((dataset['train_data'] - min_val) / denom).astype(np.float32, copy=False)
        dataset['test_data'] = ((dataset['test_data'] - min_val) / denom).astype(np.float32, copy=False)

    return dataset


def get_UCR_dataset(data_path, dataset_name, ts_num, normalized=True):
    dataset = {}
    file_list = os.listdir(data_path)
    found_any = False

    for file_name in file_list:
        if file_name.startswith(dataset_name[ts_num]):
            found_any = True
            path = os.path.join(data_path, file_name)
            if file_name.endswith('train.npy'):
                dataset['train_data'] = np.load(path).T.astype(np.float32)
            elif file_name.endswith('test.npy'):
                dataset['test_data'] = np.load(path).T.astype(np.float32)
            elif file_name.endswith('labels.npy'):
                dataset['test_label'] = np.load(path).flatten().astype(np.int64)

    if not found_any:
        raise FileNotFoundError(f"UCR files not found for prefix {dataset_name[ts_num]}")

    if normalized:
        max_value = np.max(dataset['train_data'][0])
        min_value = np.min(dataset['train_data'][0])
        denom = max(float(max_value - min_value), 1e-6)
        dataset['test_data'][0][:] = ((dataset['test_data'][0][:] - min_value) / denom)
        dataset['train_data'][0][:] = ((dataset['train_data'][0][:] - min_value) / denom)

    dataset['train_data'] = dataset['train_data'].astype(np.float32, copy=False)
    dataset['test_data'] = dataset['test_data'].astype(np.float32, copy=False)
    return dataset


def get_KPI_dataset(data_path, ts_num, normalized=True):
    tr = pd.read_csv(os.path.join(data_path, 'phase2_train', 'phase2_train.csv'))
    ids1 = tr['KPI ID'].unique()
    if ts_num >= len(ids1):
        raise IndexError("kpi ts_num out of range")
    tr = tr.loc[tr['KPI ID'] == ids1[ts_num]]['value'].values.reshape(1, -1).astype(np.float32)

    gt2 = os.path.join(data_path, 'phase2_ground_truth', 'phase2_ground_truth.hdf')
    te = pd.read_hdf(gt2)
    ids2 = te['KPI ID'].unique()
    te = te.loc[te['KPI ID'] == ids2[ts_num]]
    te_labels = te['label'].values.astype(np.int64)
    te_data = te['value'].values.reshape(1, -1).astype(np.float32)

    if normalized:
        max_value = np.max(tr)
        min_value = np.min(tr)
        denom = max(float(max_value - min_value), 1e-6)
        te_data[:] = ((te_data[:] - min_value) / denom)
        tr_data = ((tr[:] - min_value) / denom).astype(np.float32)
    else:
        tr_data = tr.astype(np.float32)

    return {'train_data': tr_data, 'test_data': te_data.astype(np.float32), 'test_label': te_labels}


def get_Gesture_dataset(data_path, normalized=True, validation_ratio=0.2):
    with open(os.path.join(data_path, 'labeled/train/ann_gun_CentroidA.pkl'), 'rb') as trainfile:
        tr_data = pd.DataFrame(pd.read_pickle(trainfile))
    with open(os.path.join(data_path, 'labeled/test/ann_gun_CentroidA.pkl'), 'rb') as testfile:
        te_data = pd.DataFrame(pd.read_pickle(testfile))

    train_data = tr_data[[0, 1]].to_numpy(dtype=np.float32).T
    test_data = te_data[[0, 1]].to_numpy(dtype=np.float32).T
    test_label = te_data[[2]].to_numpy().reshape(-1).astype(np.int64)

    if normalized:
        f_max = max(np.max(train_data[0, :]), np.max(train_data[1, :]))
        f_min = min(np.min(train_data[0, :]), np.min(train_data[1, :]))
        denom = max(float(f_max - f_min), 1e-6)
        train_data = ((train_data - f_min) / denom).astype(np.float32, copy=False)
        test_data = ((test_data - f_min) / denom).astype(np.float32, copy=False)

    n_validate = int(test_data.shape[1] * validation_ratio)
    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_label': test_label,
        'validate_data': test_data[:, 0:n_validate],
        'validate_label': test_label[0:n_validate]
    }


def get_Power_Demand_dataset(data_path, normalized=True, validation_ratio=0.2):
    with open(os.path.join(data_path, 'labeled/train/power_data.pkl'), 'rb') as trainfile:
        tr_data = pd.DataFrame(pd.read_pickle(trainfile))
    with open(os.path.join(data_path, 'labeled/test/power_data.pkl'), 'rb') as testfile:
        te_data = pd.DataFrame(pd.read_pickle(testfile))

    train_data = tr_data[[0]].to_numpy(dtype=np.float32).T
    test_data = te_data[[0]].to_numpy(dtype=np.float32).T
    test_label = te_data[[1]].to_numpy().reshape(-1).astype(np.int64)

    if normalized:
        max_value = np.max(train_data)
        min_value = np.min(train_data)
        denom = max(float(max_value - min_value), 1e-6)
        test_data[:] = ((test_data[:] - min_value) / denom)
        train_data[:] = ((train_data[:] - min_value) / denom)

    n_validate = int(test_data.shape[1] * validation_ratio)
    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_label': test_label,
        'validate_data': test_data[:, 0:n_validate],
        'validate_label': test_label[0:n_validate]
    }


def get_ECG_dataset(data_path, dataset_name, ts_num, normalized=True, validation_ratio=0.2):
    print(dataset_name[ts_num])
    with open(os.path.join(data_path, 'labeled/train', dataset_name[ts_num]), 'rb') as trainfile:
        tr_data = pd.DataFrame(pd.read_pickle(trainfile))
    with open(os.path.join(data_path, 'labeled/test', dataset_name[ts_num]), 'rb') as testfile:
        te_data = pd.DataFrame(pd.read_pickle(testfile))

    train_data = tr_data[[0, 1]].to_numpy(dtype=np.float32).T
    test_data = te_data[[0, 1]].to_numpy(dtype=np.float32).T
    test_label = te_data[[2]].to_numpy().reshape(-1).astype(np.int64)

    if normalized:
        f_max = max(np.max(train_data[0, :]), np.max(train_data[1, :]))
        f_min = min(np.min(train_data[0, :]), np.min(train_data[1, :]))
        denom = max(float(f_max - f_min), 1e-6)
        train_data = ((train_data - f_min) / denom).astype(np.float32, copy=False)
        test_data = ((test_data - f_min) / denom).astype(np.float32, copy=False)

    n_validate = int(test_data.shape[1] * validation_ratio)
    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_label': test_label,
        'validate_data': test_data[:, 0:n_validate],
        'validate_label': test_label[0:n_validate]
    }


def get_NAB_dataset(data_path, normalized=True, validation_ratio=0.2):
    dataset = {}
    for filename in os.listdir(data_path):
        if filename.startswith('ec2'):
            path = os.path.join(data_path, filename)
            if filename.endswith('train.npy'):
                dataset['train_data'] = np.load(path)[0:2001, :].T.astype(np.float32)
            elif filename.endswith('test.npy'):
                dataset['test_data'] = np.load(path)[2001:, :].T.astype(np.float32)
            elif filename.endswith('labels.npy'):
                labels = np.load(path).flatten()
                dataset['test_label'] = labels[2001:].astype(np.int64)

    if normalized:
        max_value = np.max(dataset['train_data'][0])
        min_value = np.min(dataset['train_data'][0])
        denom = max(float(max_value - min_value), 1e-6)
        dataset['test_data'][0][:] = ((dataset['test_data'][0][:] - min_value) / denom)
        dataset['train_data'][0][:] = ((dataset['train_data'][0][:] - min_value) / denom)

    n_validate = int(dataset['test_data'].shape[1] * validation_ratio)
    dataset['validate_data'] = dataset['test_data'][:, 0:n_validate]
    dataset['validate_label'] = dataset['test_label'][0:n_validate]
    dataset['train_data'] = dataset['train_data'].astype(np.float32, copy=False)
    dataset['test_data'] = dataset['test_data'].astype(np.float32, copy=False)
    return dataset


def sliding_window_generation(dataset, window_size, step_size=1):
    train_samples, reconstruction_label = training_samples_generation(
        train_data=dataset['train_data'], window_size=window_size, step_size=step_size
    )
    trainset = {'samples': train_samples, 'labels': reconstruction_label}

    test_samples, anomaly_labels = testing_samples_generation(
        test_data=dataset['test_data'], test_label=dataset['test_label'],
        window_size=window_size, step_size=step_size
    )
    testset = {'samples': test_samples, 'labels': anomaly_labels}
    return trainset, testset


def training_samples_generation(train_data, window_size, step_size=1):
    dimension = train_data.shape[0]
    total_length = train_data.shape[1]
    starts = _build_window_starts(total_length, window_size, step_size)
    num_samples = int(starts.shape[0])

    samples = np.zeros((num_samples, dimension, window_size), dtype=np.float32)
    reconstruction_label = np.zeros((num_samples, dimension, window_size), dtype=np.float32)
    for idx, i in enumerate(starts):
        win = np.asarray(train_data[:, i:i + window_size], dtype=np.float32)
        samples[idx] = win
        reconstruction_label[idx] = win
    return samples, reconstruction_label


def testing_samples_generation(test_data, test_label, window_size, step_size=1):
    dimension = test_data.shape[0]
    total_length = test_data.shape[1]
    starts = _build_window_starts(total_length, window_size, step_size)
    num_samples = int(starts.shape[0])

    samples = np.zeros((num_samples, dimension, window_size), dtype=np.float32)
    test_labels = np.zeros((num_samples, 1, window_size), dtype=np.float32)
    test_label = np.asarray(test_label).reshape(-1)
    for idx, i in enumerate(starts):
        samples[idx] = np.asarray(test_data[:, i:i + window_size], dtype=np.float32)
        test_labels[idx] = np.asarray(test_label[i:i + window_size].reshape(1, -1), dtype=np.float32)
    return samples, test_labels


def _split_val_test_wadi_by_rate(full_test_set: dict, val_ratio: float,
                                 preserve_segments: bool = True, seed: int = 42):
    samples = full_test_set['samples']
    labels = full_test_set['labels']
    pos_mask = (labels.max(axis=(1, 2)) > 0.0).astype(bool)
    val_idx, test_idx = _split_val_test_wadi_indices(
        pos_mask=pos_mask,
        val_ratio=val_ratio,
        preserve_segments=preserve_segments,
        seed=seed
    )

    val_set = {'samples': samples[val_idx], 'labels': labels[val_idx]}
    test_set = {'samples': samples[test_idx], 'labels': labels[test_idx]}
    val_pos = int((val_set['labels'].max(axis=(1, 2)) > 0.0).sum())
    test_pos = int((test_set['labels'].max(axis=(1, 2)) > 0.0).sum())
    print(f"[WADI split@rate] val windows={val_idx.size} (pos={val_pos}), "
          f"test windows={test_idx.size} (pos={test_pos})")
    return val_set, test_set
