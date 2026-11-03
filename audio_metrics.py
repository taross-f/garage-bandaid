"""音声の診断値を計算する共通モジュール（prep.py / mix.py から使う）。"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from scipy import signal

BANDS = {
    "low (20-200Hz)": (20, 200),
    "low-mid (200-400Hz)": (200, 400),
    "mid (400-4kHz)": (400, 4000),
    "high (4k-20kHz)": (4000, 20000),
}
DBFS_EPS = 1e-12


def db(x: float) -> float:
    return float(20 * np.log10(max(x, DBFS_EPS)))


def to_2d(x: np.ndarray) -> np.ndarray:
    """(samples,) または (samples, ch) を (samples, ch) に揃える。"""
    return x[:, None] if x.ndim == 1 else x


def clip_runs(x: np.ndarray, threshold: float, min_run: int = 3) -> dict:
    """|x| >= threshold が min_run サンプル以上続く区間を数える。"""
    x = to_2d(x)
    runs: list[int] = []
    for ch in range(x.shape[1]):
        over = (np.abs(x[:, ch]) >= threshold).astype(np.int8)
        edges = np.diff(np.concatenate(([0], over, [0])))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        runs.extend(lengths[lengths >= min_run].tolist())
    samples = int(sum(runs))
    return {
        "threshold_dbfs": db(threshold),
        "runs": len(runs),
        "clipped_samples": samples,
        "ratio_pct": 100.0 * samples / x.size,
        "longest_run": max(runs, default=0),
    }


def true_peak_dbfs(x: np.ndarray, oversample: int = 4) -> float:
    x = to_2d(x)
    up = signal.resample_poly(x, oversample, 1, axis=0)
    return db(float(np.max(np.abs(up))))


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """再生したときの音量。モノラルは左右両方から鳴るので、2ch に複製して測る。"""
    x = to_2d(x)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return float(pyln.Meter(sr).integrated_loudness(x))


def band_balance(x: np.ndarray, sr: int) -> dict:
    """帯域ごとのエネルギー比（%）。"""
    mono = to_2d(x).mean(axis=1)
    freqs, psd = signal.welch(mono, sr, nperseg=8192)
    in_range = (freqs >= 20) & (freqs < 20000)
    total = float(psd[in_range].sum())
    return {
        name: 100.0 * float(psd[(freqs >= lo) & (freqs < hi)].sum()) / total
        for name, (lo, hi) in BANDS.items()
    }


def diagnose(x: np.ndarray, sr: int, clip_threshold: float = 32767 / 32768) -> dict:
    return {
        "lufs": integrated_lufs(x, sr),
        "true_peak_dbtp": true_peak_dbfs(x),
        "sample_peak_dbfs": db(float(np.max(np.abs(x)))),
        "rms_dbfs": db(float(np.sqrt(np.mean(x**2)))),
        "clip": clip_runs(x, clip_threshold),
        "bands_pct": band_balance(x, sr),
    }
