"""Common quality metrics for rPPG evaluation."""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr

from rppg_benchmark.rppg_analyzer import RPPGSignalAnalyzer


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

def ref_mean_hr(ref_ppg, fps: float = 30):
    """ Среднее (ЧСС) HR по пикам (BPM) на основе эталонного сигнала

    :param ref_ppg: Эталонный PPG
    :param fps: Частота кадров
    """
    ref_analyzer = RPPGSignalAnalyzer(ref_ppg, fps=fps)
    return ref_analyzer.mean_hr

def pred_mean_hr(pred_ppg, fps: float = 30):
    """ Среднее (ЧСС) HR по пикам (BPM) на основе прогнозируемого сигнала

    :param pred_ppg: Прогнозируемый PPG
    :param fps: Частота кадров
    """
    pred_analyzer = RPPGSignalAnalyzer(pred_ppg, fps=fps)
    return pred_analyzer.mean_hr

def errors_mean_hr(ref_ppg, pred_ppg, fps: float = 30):
    """ Ошибка в процентах между эталонного и прогнозируемого сигналов среднего (ЧСС)

    :param ref_ppg: Эталонный PPG
    :param pred_ppg: Прогнозируемый PPG
    :param fps: Частота кадров
    """
    val_ref = ref_mean_hr(ref_ppg, fps=fps)
    val_pred = pred_mean_hr(pred_ppg, fps=fps)
    res = round(abs(val_pred - val_ref) / val_ref * 100, 1)
    return res

def ref_median_hr(ref_ppg, fps: float = 30):
    """ Медианный (ЧСС) HR по пикам (BPM) на основе эталонного сигнала

    :param ref_ppg: Эталонный PPG
    :param fps: Частота кадров
    """
    ref_analyzer = RPPGSignalAnalyzer(ref_ppg, fps=fps)
    return ref_analyzer.median_hr

def pred_median_hr(pred_ppg, fps: float = 30):
    """ Медианный (ЧСС) HR по пикам (BPM) на основе прогнозируемого сигнала

    :param pred_ppg: Прогнозируемый PPG
    :param fps: Частота кадров
    """
    pred_analyzer = RPPGSignalAnalyzer(pred_ppg, fps=fps)
    return pred_analyzer.median_hr

def errors_median_hr(ref_ppg, pred_ppg, fps: float = 30):
    """ Ошибка в процентах между эталонного и прогнозируемого сигналов медианный (ЧСС)

    :param ref_ppg: Эталонный PPG
    :param pred_ppg: Прогнозируемый PPG
    :param fps: Частота кадров
    """
    val_ref = ref_median_hr(ref_ppg, fps=fps)
    val_pred = pred_median_hr(pred_ppg, fps=fps)
    res = round(abs(val_pred - val_ref) / val_ref * 100, 1)
    return res

def ref_hrv(ref_ppg, fps: float = 30):
    """ Вариабельность сердечного ритма (разброс между последовательными ударами сердца)

    :param pred_ppg: Эталонный PPG
    :param fps: Частота кадров
    """
    ref_analyzer = RPPGSignalAnalyzer(ref_ppg, fps=fps)
    return ref_analyzer.hrv

def pred_hrv(pred_ppg, fps: float = 30):
    """ Вариабельность сердечного ритма (разброс между последовательными ударами сердца)

    :param pred_ppg: Прогнозируемый PPG
    :param fps: Частота кадров
    """
    pred_analyzer = RPPGSignalAnalyzer(pred_ppg, fps=fps)
    return pred_analyzer.hrv