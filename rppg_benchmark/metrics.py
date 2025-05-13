"""Common quality metrics for rPPG evaluation."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    """Средняя абсолютная ошибка (BPM)."""
    return float(np.mean(np.abs(pred - ref)))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Средняя квадратная ошибка в квадрате (BPM)."""
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def snr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Отношение сигнал / шум (DB) между прогнозируемыми и эталонными сигналами."""
    noise = pred - ref
    return float(10 * np.log10(np.mean(ref**2) / np.mean(noise**2)))


def corr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона."""
    return float(pearsonr(pred, ref)[0])

def mape(pred: np.ndarray, ref: np.ndarray) -> float:
    """Средний абсолютный процент ошибки (%) между прогнозируемыми и эталонными значениями."""
    mask = ref != 0
    if not np.any(mask):
        return float("nan")  # Нет допустимых значений
    return float(np.mean(np.abs((pred[mask] - ref[mask]) / ref[mask])) * 100.0)

def smape(pred, ref):
    """ Симметричная средняя абсолютная ошибка в процентах """
    denominator = (np.abs(ref) + np.abs(pred)) / 2.0
    mask = denominator != 0
    return np.mean(np.abs(pred[mask] - ref[mask]) / denominator[mask]) * 100