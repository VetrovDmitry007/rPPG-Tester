"""Common quality metrics for rPPG evaluation."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean Absolute Error (BPM)."""
    return float(np.mean(np.abs(pred - ref)))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Root Mean Squared Error (BPM)."""
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def snr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Signal‑to‑Noise Ratio (dB) between predicted and reference signals."""
    noise = pred - ref
    return float(10 * np.log10(np.mean(ref**2) / np.mean(noise**2)))


def corr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(pearsonr(pred, ref)[0])