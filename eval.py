

import os
import re
import glob
import csv
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

from utils.config import read_config
from utils.ds_reader import Dataset_Loader, get_dataset_series_names
from model.models import TransformerEncoderModel, WeakDecoder
from model.loss import ReconstructionLoss

warnings.filterwarnings("ignore")


def _safe_mkdir(p: str):
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)


def _parse_epoch_from_ckpt(path: str, ts_name: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(rf"^{re.escape(ts_name)}_epoch_(\d+)\.pth$", base)
    return int(m.group(1)) if m else None


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    try:
        win = np.lib.stride_tricks.sliding_window_view(xpad, k)
        return np.median(win, axis=-1).astype(x.dtype)
    except Exception:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.median(xpad[i:i + k])
        return out


def _ema_filter_1d(x: np.ndarray, alpha: float) -> np.ndarray:

    alpha = float(alpha)
    if alpha <= 0:
        return x
    if alpha > 1:
        alpha = 1.0
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = (1.0 - alpha) * y[i - 1] + alpha * x[i]
    return y


def _max_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    try:
        win = np.lib.stride_tricks.sliding_window_view(xpad, k)
        return np.max(win, axis=-1).astype(x.dtype)
    except Exception:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.max(xpad[i:i + k])
        return out


def _min_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    return -_max_filter_1d(-x, k)


def score_morphology(x: np.ndarray, mode: str, kernel: int) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode in ("none", "off", "false", ""):
        return x
    k = int(kernel)
    if k <= 1:
        return x
    if mode in ("closing", "close"):
        return _min_filter_1d(_max_filter_1d(x, k), k)
    if mode in ("opening", "open"):
        return _max_filter_1d(_min_filter_1d(x, k), k)
    if mode in ("close_open", "closing_opening"):
        closed = _min_filter_1d(_max_filter_1d(x, k), k)
        return _max_filter_1d(_min_filter_1d(closed, k), k)
    if mode in ("open_close", "opening_closing"):
        opened = _max_filter_1d(_min_filter_1d(x, k), k)
        return _min_filter_1d(_max_filter_1d(opened, k), k)
    raise ValueError(f"Unknown score morphology mode: {mode}")


def smooth_scores(x: np.ndarray, method: str, kernel: int = 5, ema_alpha: float = 0.2) -> np.ndarray:
    method = (method or "none").lower()
    if method in ("none", "off", "false", ""):
        return x
    if method in ("median", "med"):
        return _median_filter_1d(x, int(kernel))
    if method in ("max", "maxpool", "maximum"):
        return _max_filter_1d(x, int(kernel))
    if method.startswith("max") and method[3:].isdigit():
        return _max_filter_1d(x, int(method[3:]))
    if method in ("ema", "lowpass", "iir"):
        return _ema_filter_1d(x, float(ema_alpha))
    raise ValueError(f"Unknown smoothing method: {method}")


def _bridge_gaps(pred: np.ndarray, gap: int) -> np.ndarray:

    if gap <= 0:
        return pred
    y = pred.copy().astype(np.uint8)
    n = y.size
    i = 0
    while i < n:
        if y[i] == 1:
            i += 1
            continue
        j = i
        while j < n and y[j] == 0:
            j += 1

        left_is_1 = (i - 1 >= 0 and y[i - 1] == 1)
        right_is_1 = (j < n and y[j] == 1)
        if left_is_1 and right_is_1 and (j - i) <= gap:
            y[i:j] = 1
        i = j
    return y


def _remove_short_positive_runs(pred: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return pred
    y = pred.copy().astype(np.uint8)
    n = y.size
    i = 0
    while i < n:
        if y[i] == 0:
            i += 1
            continue
        j = i
        while j < n and y[j] == 1:
            j += 1
        if (j - i) < min_len:
            y[i:j] = 0
        i = j
    return y


def postprocess_pred(pred: np.ndarray, min_len: int = 1, bridge_gap: int = 0) -> np.ndarray:
    y = pred.astype(np.uint8)
    if bridge_gap > 0:
        y = _bridge_gaps(y, int(bridge_gap))
    if min_len > 1:
        y = _remove_short_positive_runs(y, int(min_len))
    return y.astype(np.uint8)


def reconstruct_point_labels_from_window_labels(win_labels: np.ndarray, step_size: int) -> np.ndarray:
    """
    win_labels: [num_windows, 1, W] where each window stores the original point labels slice.
    We assume windows are generated sequentially with stride=step_size and shuffle=False.
    """
    win_labels = (win_labels > 0).astype(np.uint8)
    num_w, _, W = win_labels.shape
    if num_w == 0:
        return np.zeros((0,), dtype=np.uint8)
    if step_size <= 0:
        step_size = 1

    seq = win_labels[0, 0, :].astype(np.uint8).tolist()
    if num_w > 1:
        tail = win_labels[1:, 0, -step_size:].reshape(-1).astype(np.uint8).tolist()
        seq.extend(tail)
    return np.asarray(seq, dtype=np.uint8)


def window_scores_to_point_series(win_scores: np.ndarray, window_size: int, step_size: int,
                                 mode: str = "interp_center") -> np.ndarray:
    """
    Map per-window scores (length = num_windows) to a point-wise score series (length = T),
    where T = (num_windows-1)*step_size + window_size.
    mode:
      - interp_center: assign each window score to its center point, then linear interpolate.
      - hold: piecewise-constant between centers (nearest left).
    """
    s = np.asarray(win_scores, dtype=np.float64)
    n = s.size
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if step_size <= 0:
        step_size = 1
    W = int(window_size)
    T = int((n - 1) * step_size + W)
    starts = np.arange(n, dtype=np.int64) * int(step_size)
    centers = starts + (W // 2)
    xq = np.arange(T, dtype=np.int64)

    if mode == "hold":
        out = np.empty((T,), dtype=np.float64)
        out[:] = s[0]
        j = 0
        for t in range(T):
            while (j + 1 < n) and (centers[j + 1] <= t):
                j += 1
            out[t] = s[j]
        return out.astype(np.float32)

    out = np.interp(xq.astype(np.float64), centers.astype(np.float64), s,
                    left=float(s[0]), right=float(s[-1]))
    return out.astype(np.float32)


def collect_recon_and_emb(projection_layer: nn.Module,
                          encoder: nn.Module,
                          decoder: nn.Module,
                          recon_loss_fn: nn.Module,
                          data_loader,
                          device: torch.device,
                          need_window_labels: bool,
                          use_amp: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      recon_win: [num_windows]
      emb_win:   [num_windows, hidden_dim]
      win_labels (if need_window_labels): [num_windows, 1, W]
    """
    projection_layer.eval()
    encoder.eval()
    decoder.eval()

    all_recon = []
    all_emb = []
    all_lbl = [] if need_window_labels else None

    amp_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    with torch.inference_mode():
        for pack in tqdm(data_loader, desc="Infer", leave=False):
            x = pack[0]
            y = pack[1] if (need_window_labels and len(pack) >= 2) else None


            x = x.permute(0, 2, 1).contiguous().to(device, non_blocking=True)

            with amp_ctx:
                z = encoder(projection_layer(x))
                xhat = decoder(z)
                recon = torch.mean(recon_loss_fn(xhat, x), dim=(1, 2))
                emb = torch.mean(z, dim=1)

            all_recon.append(recon.detach().float().cpu().numpy())
            all_emb.append(emb.detach().float().cpu().numpy())
            if need_window_labels and (y is not None):
                all_lbl.append(y.detach().cpu().numpy())

    recon_win = np.concatenate(all_recon, axis=0).astype(np.float32) if len(all_recon) else np.empty((0,), dtype=np.float32)
    emb_win = np.concatenate(all_emb, axis=0).astype(np.float32) if len(all_emb) else np.empty((0, 1), dtype=np.float32)
    win_labels = np.concatenate(all_lbl, axis=0).astype(np.float32) if (need_window_labels and all_lbl) else None
    return recon_win, emb_win, win_labels


def find_best_gmm_components(embeddings: np.ndarray, max_components: int = 30, covariance_type: str = "full") -> int:
    lowest_bic = np.inf
    best_k = 1
    X = embeddings.astype(np.float64)
    for k in tqdm(range(1, max_components + 1), desc="BIC(GMM)", leave=False):
        try:
            gmm, _ = fit_gmm_robust(X, k, covariance_type=covariance_type, verbose=False)
            bic = gmm.bic(X)
        except Exception:
            continue
        if bic < lowest_bic:
            lowest_bic = bic
            best_k = k
    return best_k


def fit_gmm_robust(
    embeddings: np.ndarray,
    requested_components: int,
    covariance_type: str = "full",
    verbose: bool = True,
) -> Tuple[GaussianMixture, int]:
    X = embeddings.astype(np.float64)
    requested_components = int(max(1, min(int(requested_components), X.shape[0])))
    component_candidates = []
    for k in [requested_components, 32, 24, 16, 12, 8, 4, 2, 1]:
        k = int(max(1, min(k, X.shape[0])))
        if k not in component_candidates:
            component_candidates.append(k)

    last_error = None
    for k in component_candidates:
        for reg in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    random_state=42,
                    reg_covar=reg,
                    max_iter=200,
                )
                gmm.fit(X)
                if verbose and k != requested_components:
                    print(f"[GMM] fallback components {requested_components}->{k} reg={reg:g}")
                return gmm, k
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"GMM fit failed after fallback: {last_error}")


def _pr_best_threshold(labels01: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thr = precision_recall_curve(labels01.astype(int), score.astype(float))
    if thr.size == 0:
        return 1.0
    p = precision[:-1]
    r = recall[:-1]
    f1 = (2 * p * r) / (p + r + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx])


def _f1_pr_from_pred(labels01: np.ndarray, pred01: np.ndarray) -> Tuple[float, float, float]:
    y = labels01.astype(np.uint8)
    p = pred01.astype(np.uint8)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(f1), float(prec), float(rec)


def refine_threshold_with_postproc(score: np.ndarray,
                                  labels01: np.ndarray,
                                  t0: float,
                                  delta: float,
                                  steps: int,
                                  min_len: int,
                                  bridge_gap: int) -> Tuple[float, float, float, float]:
    """
    Returns: best_f1, best_p, best_r, best_t
    """
    if steps <= 1:
        pred = (score >= t0).astype(np.uint8)
        pred = postprocess_pred(pred, min_len=min_len, bridge_gap=bridge_gap)
        f1, p, r = _f1_pr_from_pred(labels01, pred)
        return f1, p, r, float(t0)

    cands = np.linspace(t0 - delta, t0 + delta, steps)
    cands = np.clip(cands, 0.0, 1.0)

    best = (-1.0, 0.0, 0.0, float(t0))
    for t in cands:
        pred = (score >= t).astype(np.uint8)
        pred = postprocess_pred(pred, min_len=min_len, bridge_gap=bridge_gap)
        f1, p, r = _f1_pr_from_pred(labels01, pred)
        if f1 > best[0]:
            best = (f1, p, r, float(t))
    return best


def evaluate_fixed_alpha(
    gmm_norm_point: np.ndarray,
    recon_norm_point: np.ndarray,
    labels_point: np.ndarray,
    alpha: float,
    smooth_method: str,
    smooth_kernel: int,
    ema_alpha: float,
    min_len: int,
    bridge_gap: int,
    thr_refine_delta: float,
    thr_refine_steps: int,
    score_morph_mode: str = "none",
    score_morph_kernel: int = 1,
) -> Dict[str, float]:
    labels01 = labels_point.astype(np.uint8)
    s = alpha * gmm_norm_point + (1.0 - alpha) * recon_norm_point
    s = np.clip(s, 0.0, 1.0)
    s = smooth_scores(s, method=smooth_method, kernel=smooth_kernel, ema_alpha=ema_alpha)
    s = np.clip(s, 0.0, 1.0)
    s = score_morphology(s, mode=score_morph_mode, kernel=score_morph_kernel)
    s = np.clip(s, 0.0, 1.0)

    t0 = _pr_best_threshold(labels01, s)
    f1, p, r, t = refine_threshold_with_postproc(
        s, labels01, t0=t0,
        delta=thr_refine_delta,
        steps=thr_refine_steps,
        min_len=min_len,
        bridge_gap=bridge_gap
    )
    try:
        auc = float(roc_auc_score(labels01.astype(int), s.astype(float)))
    except Exception:
        auc = 0.0
    return {
        "alpha": float(alpha),
        "best_f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "auc": float(auc),
        "best_threshold": float(t),
    }


@dataclass
class EvalRow:
    exp_name: str
    exp_dir: str
    ts_name: str
    ckpt_path: str
    epoch: int
    best_f1: float
    precision: float
    recall: float
    auc: float
    aupr: float
    alpha: float
    threshold: float
    gmm_n_components: int
    recon_margin: float
    gmm_margin: float
    avg_recon_normal: float
    avg_recon_anomaly: float
    avg_gmm_normal: float
    avg_gmm_anomaly: float


def write_csv(path: str, rows: List[EvalRow]):
    header = [
        "exp_name", "exp_dir",
        "ts_name", "epoch", "ckpt_path",
        "best_f1", "precision", "recall", "aupr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.exp_name, r.exp_dir,
                r.ts_name, r.epoch, r.ckpt_path,
                f"{r.best_f1:.6f}", f"{r.precision:.6f}", f"{r.recall:.6f}",
                f"{r.aupr:.6f}",
            ])


def print_macro_summary(rows: List[EvalRow], title: str):
    f1_values = [float(r.best_f1) for r in rows if np.isfinite(float(r.best_f1))]
    aupr_values = [float(r.aupr) for r in rows if np.isfinite(float(r.aupr))]
    macro_f1 = float(np.mean(f1_values)) if f1_values else float("nan")
    macro_aupr = float(np.mean(aupr_values)) if aupr_values else float("nan")
    print(f"[Macro] {title}: F1={macro_f1:.6f}, AUPR={macro_aupr:.6f}, N={len(rows)}")


def read_best_ckpt_csv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_name = str(row.get("ts_name", "")).strip()
            ckpt_path = str(row.get("ckpt_path", "")).strip()
            if ts_name and ckpt_path:
                out[ts_name] = ckpt_path
    return out


def evaluate_one_checkpoint(
    ckpt_path: str,
    ts_name: str,
    exp_name: str,
    exp_dir: str,
    config: Dict[str, Any],
    device: torch.device,
    train_loader,
    test_loader,
    step_size: int,
    window_size: int,
    cached_n_components: Optional[int] = None,
    compute_curve_metrics: bool = False,
    return_fused_labels: bool = False,
) -> Tuple[EvalRow, int, Optional[Dict[str, np.ndarray]]]:

    model_cfg = config["model"]["encoder"]
    input_dim = int(model_cfg["input_dim"])
    proj_dim = int(model_cfg["proj_dim"])
    hidden_dim = int(model_cfg["hidden_dim"])
    nhead = int(model_cfg["nhead"])
    num_layers = int(model_cfg["num_layers"])
    dim_feedforward = int(model_cfg["dim_feedforward"])
    dropout = float(model_cfg["dropout"])

    projection_layer = nn.Linear(input_dim, proj_dim).to(device)
    encoder = TransformerEncoderModel(proj_dim, hidden_dim, nhead, num_layers, dim_feedforward, dropout).to(device)
    decoder = WeakDecoder(hidden_dim=hidden_dim, output_dim=input_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "projection_layer_state_dict" in ckpt:
        projection_layer.load_state_dict(ckpt["projection_layer_state_dict"])
    else:
        raise KeyError(f"checkpoint missing projection_layer_state_dict: {ckpt_path}")
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    eval_cfg = config.get("evaluation", {})
    score_level = str(eval_cfg.get("score_level", "point")).lower()
    if score_level not in {"point", "window"}:
        raise ValueError("evaluation.score_level must be 'point' or 'window'")


    use_amp = bool(eval_cfg.get("use_amp", False)) and (device.type == "cuda")


    gmm_components = int(eval_cfg.get("gmm_components", 0))
    max_components = int(eval_cfg.get("max_gmm_components", 36))
    reuse_gmm_components = bool(eval_cfg.get("reuse_gmm_components", True))
    cov_type = str(eval_cfg.get("gmm_covariance_type", "full"))


    smooth_cfg = eval_cfg.get("smoothing", {}) or {}
    smooth_method = str(smooth_cfg.get("method", "median"))
    smooth_kernel = int(smooth_cfg.get("kernel", 5))
    ema_alpha = float(smooth_cfg.get("ema_alpha", 0.2))

    score_pp_cfg = eval_cfg.get("score_postprocess", {}) or {}
    score_morph_mode = str(score_pp_cfg.get("mode", "none"))
    score_morph_kernel = int(score_pp_cfg.get("kernel", 1))
    fixed_alpha = score_pp_cfg.get("fixed_alpha", None)
    fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)

    pp_cfg = eval_cfg.get("postprocess", {}) or {}
    min_len = int(pp_cfg.get("min_len", 0))
    bridge_gap = int(pp_cfg.get("bridge_gap", 0))

    thr_cfg = eval_cfg.get("threshold_refine", {}) or {}
    thr_refine_delta = float(thr_cfg.get("delta", 0.02))
    thr_refine_steps = int(thr_cfg.get("steps", 11))

    recon_loss_fn = ReconstructionLoss().to(device)


    train_recon_win, train_emb, _ = collect_recon_and_emb(
        projection_layer, encoder, decoder, recon_loss_fn,
        train_loader, device, need_window_labels=False, use_amp=use_amp
    )

    if train_emb.size == 0 or (not np.isfinite(train_emb).all()):
        raise ValueError("Train embeddings contain NaN/Inf or are empty.")


    if gmm_components > 0:
        n_components = gmm_components
    else:
        if (cached_n_components is not None) and reuse_gmm_components:
            n_components = int(cached_n_components)
        else:
            n_components = find_best_gmm_components(train_emb, max_components=max_components, covariance_type=cov_type)

    gmm, n_components = fit_gmm_robust(
        train_emb,
        requested_components=int(n_components),
        covariance_type=cov_type,
        verbose=True,
    )


    train_ll = gmm.score_samples(train_emb.astype(np.float64))
    train_gmm_win = (-train_ll).astype(np.float32)


    test_recon_win, test_emb, test_win_labels = collect_recon_and_emb(
        projection_layer, encoder, decoder, recon_loss_fn,
        test_loader, device, need_window_labels=True, use_amp=use_amp
    )
    if test_emb.size == 0 or test_win_labels is None:
        raise ValueError("Test outputs are empty.")


    test_ll = gmm.score_samples(test_emb.astype(np.float64))
    test_gmm_win = (-test_ll).astype(np.float32)

    if score_level == "window":
        labels_point = (test_win_labels.max(axis=(1, 2)) > 0).astype(np.uint8)
        recon_point = test_recon_win.astype(np.float32)
        gmm_point = test_gmm_win.astype(np.float32)
        train_recon_ref = train_recon_win.astype(np.float32)
        train_gmm_ref = train_gmm_win.astype(np.float32)
    else:

        labels_point = reconstruct_point_labels_from_window_labels(test_win_labels, step_size=step_size)
        labels_point = (labels_point > 0).astype(np.uint8)


        recon_point = window_scores_to_point_series(test_recon_win, window_size=window_size, step_size=step_size, mode="interp_center")
        gmm_point = window_scores_to_point_series(test_gmm_win, window_size=window_size, step_size=step_size, mode="interp_center")
        train_recon_ref = window_scores_to_point_series(train_recon_win, window_size=window_size, step_size=step_size, mode="interp_center")
        train_gmm_ref = window_scores_to_point_series(train_gmm_win, window_size=window_size, step_size=step_size, mode="interp_center")


    scaler_fit = str(eval_cfg.get("scaler_fit", "train")).lower()
    if scaler_fit == "test":
        recon_scaler = MinMaxScaler(feature_range=(0, 1)).fit(recon_point.reshape(-1, 1))
        gmm_scaler = MinMaxScaler(feature_range=(0, 1)).fit(gmm_point.reshape(-1, 1))
    else:
        recon_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_recon_ref.reshape(-1, 1))
        gmm_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_gmm_ref.reshape(-1, 1))

    recon_norm_point = recon_scaler.transform(recon_point.reshape(-1, 1)).ravel().astype(np.float32)
    gmm_norm_point = gmm_scaler.transform(gmm_point.reshape(-1, 1)).ravel().astype(np.float32)
    recon_norm_point = np.clip(recon_norm_point, 0.0, 1.0)
    gmm_norm_point = np.clip(gmm_norm_point, 0.0, 1.0)

    if fixed_alpha is None:
        raise ValueError("Release evaluation requires evaluation.score_postprocess.fixed_alpha in the config.")
    best = evaluate_fixed_alpha(
        gmm_norm_point, recon_norm_point, labels_point,
        alpha=fixed_alpha,
        smooth_method=smooth_method,
        smooth_kernel=smooth_kernel,
        ema_alpha=ema_alpha,
        min_len=min_len,
        bridge_gap=bridge_gap,
        thr_refine_delta=thr_refine_delta,
        thr_refine_steps=thr_refine_steps,
        score_morph_mode=score_morph_mode,
        score_morph_kernel=score_morph_kernel,
    )


    alpha = float(best["alpha"])
    fused = alpha * gmm_norm_point + (1.0 - alpha) * recon_norm_point
    fused = np.clip(fused, 0.0, 1.0)
    fused = smooth_scores(fused, method=smooth_method, kernel=smooth_kernel, ema_alpha=ema_alpha)
    fused = np.clip(fused, 0.0, 1.0)
    fused = score_morphology(fused, mode=score_morph_mode, kernel=score_morph_kernel)
    fused = np.clip(fused, 0.0, 1.0)


    aupr = float("nan")
    if compute_curve_metrics:
        try:
            aupr = float(average_precision_score(labels_point.astype(int), fused.astype(float)))
        except Exception:
            aupr = 0.0


    lb = labels_point.astype(bool)
    avg_recon_normal = float(np.mean(recon_point[~lb])) if np.any(~lb) else 0.0
    avg_recon_anomaly = float(np.mean(recon_point[lb])) if np.any(lb) else 0.0
    avg_gmm_normal = float(np.mean(gmm_point[~lb])) if np.any(~lb) else 0.0
    avg_gmm_anomaly = float(np.mean(gmm_point[lb])) if np.any(lb) else 0.0
    recon_margin = (avg_recon_anomaly - avg_recon_normal) / (avg_recon_normal + 1e-6)
    gmm_margin = (avg_gmm_anomaly - avg_gmm_normal) / (abs(avg_gmm_normal) + 1e-6)

    epoch = _parse_epoch_from_ckpt(ckpt_path, ts_name)
    epoch = int(epoch) if epoch is not None else -1

    row = EvalRow(
        exp_name=str(exp_name),
        exp_dir=str(exp_dir),
        ts_name=ts_name,
        ckpt_path=ckpt_path,
        epoch=epoch,
        best_f1=float(best["best_f1"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        auc=float(best["auc"]),
        aupr=float(aupr) if np.isfinite(aupr) else float("nan"),
        alpha=float(best["alpha"]),
        threshold=float(best["best_threshold"]),
        gmm_n_components=int(n_components),
        recon_margin=float(recon_margin),
        gmm_margin=float(gmm_margin),
        avg_recon_normal=avg_recon_normal,
        avg_recon_anomaly=avg_recon_anomaly,
        avg_gmm_normal=avg_gmm_normal,
        avg_gmm_anomaly=avg_gmm_anomaly,
    )

    aux = None
    if return_fused_labels:
        aux = {
            "fused": fused.astype(np.float32),
            "labels_point": labels_point.astype(np.uint8),
        }

    return row, int(n_components), aux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")


    parser.add_argument(
        "--exp_dirs",
        type=str,
        default="",
        help="comma-separated experiment checkpoint dirs; overrides config.evaluation.exp_dirs"
    )


    parser.add_argument("--ckpt_dir", type=str, default=None, help="override single checkpoint folder")


    parser.add_argument("--out_csv", type=str, default=None, help="merged output csv path (optional)")

    parser.add_argument("--pattern", type=str, default="{ts}_epoch_*.pth", help="ckpt filename pattern")
    parser.add_argument("--best_csv", type=str, default=None, help="evaluate only ckpts listed in this best summary csv")
    parser.add_argument("--single_output_name", type=str, default="best_ckpts_summary.csv", help="per-exp output csv filename")
    args = parser.parse_args()

    config = read_config(args.config)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device = {device}")

    eval_cfg = config.get("evaluation", {})


    exp_dirs: List[str] = []

    if args.exp_dirs.strip():
        exp_dirs = [s.strip() for s in args.exp_dirs.split(",") if s.strip()]
    elif isinstance(eval_cfg.get("exp_dirs", None), (list, tuple)) and len(eval_cfg["exp_dirs"]) > 0:
        exp_dirs = [str(s) for s in eval_cfg["exp_dirs"] if str(s).strip()]
    else:
        ckpt_dir = args.ckpt_dir or eval_cfg.get("ckpt_dir", None)
        if ckpt_dir is None:
            exp_name = config.get("experiment", {}).get("name", "default_exp")
            ckpt_dir = os.path.join("./experiments", exp_name)
        exp_dirs = [ckpt_dir]

    exp_dirs = [os.path.abspath(d) for d in exp_dirs]
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
    if len(exp_dirs) == 0:
        raise FileNotFoundError("No valid exp_dirs found. Please set --exp_dirs or config.evaluation.exp_dirs")


    num_workers = int(config.get("training", {}).get("num_workers", 0))
    pin_memory = bool(config.get("training", {}).get("pin_memory", True))
    persistent_workers = bool(config.get("training", {}).get("persistent_workers", num_workers > 0))

    ts_name_list = get_dataset_series_names(
        config["data"]["dataset_name"],
        config["data"].get("data_path")
    )
    ts_name_filter = config.get("data", {}).get("ts_name_filter", None)
    if ts_name_filter:
        wanted = {str(x) for x in ts_name_filter}
        ts_name_list = [name for name in ts_name_list if name in wanted]
        print(f"[Eval] ts_name_filter enabled: {len(ts_name_list)} selected")
    print(f"[Eval] using {len(ts_name_list)} series from utils.ds_reader for dataset={config['data']['dataset_name']}")
    all_series_names = get_dataset_series_names(
        config["data"]["dataset_name"],
        config["data"].get("data_path")
    )
    series_to_idx = {name: idx for idx, name in enumerate(all_series_names)}

    window_size = int(config["data"]["window_size"])
    step_size = int(config["data"].get("step_size", 1))


    merged_rows: List[EvalRow] = []

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        print(f"\n==============================")
        print(f"[Eval] EXP: {exp_name}")
        print(f"[Eval] DIR: {exp_dir}")
        print(f"==============================")

        best_rows: List[EvalRow] = []
        best_ckpt_map: Dict[str, str] = {}
        if args.best_csv is not None:
            best_csv_path = args.best_csv
            if not os.path.isabs(best_csv_path):
                best_csv_path = os.path.join(exp_dir, best_csv_path)
            best_ckpt_map = read_best_ckpt_csv(best_csv_path)
            print(f"[Eval] using best csv: {best_csv_path} ({len(best_ckpt_map)} ckpts)")

        for loop_idx, ts_name in enumerate(ts_name_list):
            ts_idx = series_to_idx[ts_name]
            print(f"\n[Eval] TS {loop_idx + 1}/{len(ts_name_list)}: {ts_name}")

            dataset_instance = Dataset_Loader(
                dataset=config["data"]["dataset_name"],
                data_path=config["data"]["data_path"],
                ts_num=ts_idx,
                window_size=window_size,
                step_size=step_size
            )


            try:
                train_loader = dataset_instance.train_loader_generation(
                    batch_size=config["training"]["batch_size"],
                    shuffle=False,
                    return_indices=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers
                )
            except TypeError:
                train_loader = dataset_instance.train_loader_generation(
                    batch_size=config["training"]["batch_size"],
                    shuffle=False
                )


            try:
                test_loader = dataset_instance.val_test_loader_generation(
                    batch_size=config["training"]["batch_size"],
                    shuffle=False,
                    return_indices=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers
                )
            except TypeError:
                test_loader = dataset_instance.val_test_loader_generation(
                    batch_size=config["training"]["batch_size"],
                    shuffle=False
                )

            if best_ckpt_map:
                ckpt_from_csv = best_ckpt_map.get(ts_name)
                if ckpt_from_csv is None:
                    print(f"[Warn] no best csv row for {ts_name}, skip")
                    continue
                if not os.path.isabs(ckpt_from_csv):
                    ckpt_from_csv = os.path.join(exp_dir, ckpt_from_csv)
                ckpts = [ckpt_from_csv]
            else:
                patt = args.pattern.format(ts=ts_name)
                ckpts = sorted(glob.glob(os.path.join(exp_dir, patt)))
            if len(ckpts) == 0:
                print(f"[Warn] no checkpoints found for {ts_name} in {exp_dir}")
                continue

            cached_n_components = None
            best_for_ts: Optional[EvalRow] = None
            best_aux: Optional[Dict[str, np.ndarray]] = None

            for ckpt_path in ckpts:
                try:
                    row, cached_n_components_new, aux = evaluate_one_checkpoint(
                        ckpt_path=ckpt_path,
                        ts_name=ts_name,
                        exp_name=exp_name,
                        exp_dir=exp_dir,
                        config=config,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        step_size=step_size,
                        window_size=window_size,
                        cached_n_components=cached_n_components,
                        compute_curve_metrics=False,
                        return_fused_labels=True
                    )
                except Exception as e:
                    print(f"  - {os.path.basename(ckpt_path)} | [Skip] {type(e).__name__}: {e}")
                    continue

                if cached_n_components is None:
                    cached_n_components = cached_n_components_new

                if (best_for_ts is None) or (row.best_f1 > best_for_ts.best_f1):
                    best_for_ts = row
                    best_aux = aux

                print(
                    f"  - {os.path.basename(ckpt_path)} | epoch={row.epoch:>3d} | "
                    f"F1={row.best_f1:.4f} (P={row.precision:.4f}, R={row.recall:.4f}) | "
                    f"AUPR={row.aupr:.4f}"
                )

            if best_for_ts is not None:

                if best_aux is not None:
                    fused = best_aux["fused"]
                    labels_point = best_aux["labels_point"]

                    try:
                        best_for_ts.aupr = float(average_precision_score(labels_point.astype(int), fused.astype(float)))
                    except Exception:
                        best_for_ts.aupr = 0.0

                print(
                    f"[Best] {ts_name} -> epoch={best_for_ts.epoch}, "
                    f"F1={best_for_ts.best_f1:.4f}, AUPR={best_for_ts.aupr:.4f}, "
                    f"ckpt={os.path.basename(best_for_ts.ckpt_path)}"
                )
                best_rows.append(best_for_ts)

        if len(best_rows) == 0:
            print(f"[Warn] EXP {exp_name}: no best rows produced (no ckpts matched or all failed).")
            continue

        out_csv_single = os.path.join(exp_dir, args.single_output_name)
        _safe_mkdir(os.path.dirname(out_csv_single))
        write_csv(out_csv_single, best_rows)
        print(f"\n[Done] EXP {exp_name} summary saved to: {out_csv_single}")
        print_macro_summary(best_rows, f"EXP {exp_name}")

        merged_rows.extend(best_rows)

    if len(merged_rows) == 0:
        raise RuntimeError("No best rows produced across all exp_dirs (no ckpts matched or all failed).")

    if args.out_csv is not None:
        out_csv_merged = os.path.abspath(args.out_csv)
        _safe_mkdir(os.path.dirname(out_csv_merged))
        write_csv(out_csv_merged, merged_rows)
        print(f"\n[Done] MERGED summary saved to: {out_csv_merged}")

    print_macro_summary(merged_rows, "ALL")


if __name__ == "__main__":
    main()
