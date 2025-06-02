import math
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq


class RPPGSignalAnalyzer:
    """
    Класс для анализа rPPG-сигнала.

    Данный класс на основе сырого rPPG-сигнал вычисляет:
    1. 'HR по БПФ (BPM)'
    2. 'Средний HR по пикам (BPM)'
    3. 'Медианный (ЧСС) HR (BPM)'
    4. 'Стандартное отклонение (ЧСС) HR (HRV)
    5. 'BPM по числу пиков'
    6.  'Число пиков

    7. Вычисление комплекса показателей вариабельности сердечного ритма HRV

    BPM – это сокращение от "ударов в минуту" (beats per minute).
    "HR по БПФ" – пульс, вычисленный через частотный анализ (Фурье).

    HRV – это вариабельность сердечного ритма (разброс между последовательными ударами сердца).
    - Чем выше HRV, тем более адаптивна нервная система (обычно показатель хорошего здоровья и стрессоустойчивости).
    - Чем ниже HRV, тем более ригиден (напряжён) организм (может указывать на усталость, стресс или болезни).

    В процессе вычисления производится:
    - фильтрация сигнала,
    - извлечение пиков,
    - расчёт ЧСС и вариабельности,
    - визуализация и подготовка данных для оценки моделей.
    """

    def __init__(self, ppg: np.ndarray, fps: float = 30.0,  threshold: int = 0):
        """
        Инициализация на основе DataFrame с одним столбцом сигнала.
        :param ppg: Масcив 1-D с временным рядом rPPG (сырые значения rPPG)
        :param fps: Частота кадров, с которой был получен сигнал Permissible threshold
        :param threshold: Минимальный допустимый порог размера фрейма в секундах
        """
        if ppg.size <= threshold*fps:
            # raise ValueError("PPG сигнал слишком короткий")
            self.flag = False
            return None
        self.flag = True
        self.signal =  ppg.astype(np.float32)
        self.fps = fps
        self.n = len(self.signal)
        self.duration_sec = self.n / self.fps

        # Фильтрация сигнала в допустимом диапазоне (0.6–4 Гц)
        self.filtered = self._bandpass_filter(self.signal)

        # Ищутся локальные максимумы (пики), соответствующие ударам сердца
        # 0.5 -- Ограничивает макс. ЧСС ≈ 120
        self.peaks, _ = find_peaks(self.filtered, distance=self.fps * 0.5)
        # Переводим индексы пиков (в отсчётах) во время (в секундах).
        self.peak_times = self.peaks / self.fps
        # ibi -- время между соседними ударами (в секундах !!!)
        self.ibi = np.diff(self.peak_times)
        # ЧСС
        self.hr_from_peaks = 60 / self.ibi if len(self.ibi) > 0 else np.array([])

        # rr_ms -- интервалы между ударами сердца в ms.
        self.rr_ms = self.ibi * 1000
        # Вычисление комплекса показателей вариабельности сердечного ритма HRV
        self.index_hrv = self.calc_hrv(self.rr_ms)

    def calc_hrv(self, rr_ms:np.ndarray) -> Dict:
        """ Вычисление комплекса показателей вариабельности сердечного ритма HRV
        (SDNN, RMSSD, pNN20, pNN50, SD1, SD2, SD1/SD2, CSI, CVI)

        SDNN — Общая вариабельность (рекомендуется для окон > 30 сек.)
        RMSSD — Кратковременная вариабельность
        pNN20 — процент(20) отличия соседних RR-интервалов
        pNN50 — процент(50) отличия соседних RR-интервалов
        SD1 — поперечная вариабельность
        SD2 — продольная вариабельность
        SD1/SD2 - индикатор автономного баланса, чем меньше — тем сильнее симпатика
        CSI, CVI — индексы стресса (рекомендуется для окон > 30 сек.)
        """
        # rr_ms = np.array([800, 810, 790, 780, 795, 805, 800, 790])
        mean_rr = np.mean(rr_ms)

        # 1. Временные метрики (Время-доменные индексы HRV) -- SDNN,
        # RMSSD, pNN50 (для окон 20-30–с.)
        sdnn = np.std(rr_ms, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
        pnn20 = self.pnn20(rr_ms)
        pnn50 = self.pnn50(rr_ms)

        # # 2. Спектральные показатели -- LF, HF, LF/HF (требуют ≥ 1 мин)
        # rr_s = rr_ms / 1000  # сек
        # fs = 4  # Гц — целевая частота дискретизации (интерполяции)
        # t = np.cumsum(np.r_[0, rr_s])  # абсолютные временные отметки ударов сердца
        # ts = np.arange(0, t[-1], 1 / fs)  # равномерная временная шкала
        # rri = np.interp(ts, t[:-1], rr_s)  # линейная интерполяция RR на сетку ts
        # # ── Спектральный анализ методом Welch ──────────────────────────
        # # f — массив частот, P — оценённая плотность мощности (PSD)
        # f, P = sp.welch(
        #     rri,  # равномерно дискретизированный RR‑ряд
        #     fs=fs,  # частота дискретизации
        #     nperseg=256  # длина сегмента окна Хэмминга
        # )
        # # Интеграл PSD в полосах LF/HF
        # lf = np.trapz(P[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
        # hf = np.trapz(P[(f >= 0.15) & (f <= 0.40)], f[(f >= 0.15) & (f <= 0.40)])
        # lf_hf = lf / hf

        # 3. Нелинейные (Poincaré-plot) показатели -- SD1, SD2, SD1/SD2
        sd1 = rmssd / np.sqrt(2)
        sd2 = np.sqrt(2 * sdnn ** 2 - 0.5 * rmssd ** 2)
        sd1_sd2 = sd1 / sd2

        # 4. Расчёт индексов стресса -- CSI, CVI на основе Poincaré-диаграммы
        csi = sd2 / sd1  # Сердечный симпатический индекс
        cvi = math.log10(4 * sd1 * sd2)  # Сердечный вагальный индекс

        index_hrv = {'sdnn':sdnn, 'rmssd':rmssd, 'pnn20':pnn20, 'pnn50':pnn50,
                     'sd1':sd1, 'sd2':sd2, 'sd1_sd2':sd1_sd2, 'csi':csi, 'cvi':cvi,
                     'mean_rr': mean_rr, 'median_hr': self.median_hr}
        return index_hrv

    def pnn20(self, rr_ms: np.ndarray) -> float:
        """
        Вычисляет pNN20 для массива RR-интервалов в миллисекундах.
        pNN20 — это процент соседних RR-интервалов, отличающихся
        друг от друга более чем на 20 мс

        :return: pNN20 в процентах.
        """
        # 1. Разности соседних интервалов
        diff = np.diff(rr_ms)
        # 2. Сколько из них по модулю > 20 мс
        nn20 = np.sum(np.abs(diff) > 20)
        # 3. Процент от общего числа разностей
        return nn20 / len(diff) * 100

    def pnn50(self, rr_ms: np.ndarray) -> float:
        """
        Вычисляет pNN50 для массива RR-интервалов в миллисекундах.
        pNN50 — это процент соседних RR-интервалов, отличающихся
        друг от друга более чем на 50 мс

        :param rr_ms: Массив RR-интервалов в миллисекундах
        :return: pNN50 в процентах.
        """
        # 1. Разности соседних интервалов
        diff = np.diff(rr_ms)
        # 2. Сколько из них по модулю > 50 мс
        nn50 = np.sum(np.abs(diff) > 50)
        # 3. Процент от общего числа разностей
        return nn50 / len(diff) * 100

    def _bandpass_filter(self, signal: np.ndarray, lowcut=0.7, highcut=4.0) -> np.ndarray:
        """
        Применяет полосовой фильтр к сигналу.
        :param signal: исходный сигнал
        :param lowcut: нижняя граница в Гц
        :param highcut: верхняя граница в Гц
        :return: фильтрованный сигнал
        """
        # Находим частоту Найквиста
        nyquist = 0.5 * self.fps
        # Цифровой полосовой фильтр Баттерворта 3-го порядка, пропускает частоты [0.7–4.0] Гц
        # возвращает массивы b и a, которые полностью описывают поведение фильтра
        b, a = butter(N=3, Wn=[lowcut / nyquist, highcut / nyquist], btype='band')
        # Применяется двухпроходная фильтрация
        return filtfilt(b, a, signal)

    def compute_fft_hr(self) -> float:
        """
        Вычисляет частоту сердечных сокращений через БПФ.
        :return: ЧСС в ударах в минуту
        """
        centered = self.filtered - np.mean(self.filtered)
        spec = np.abs(rfft(centered))
        freqs = rfftfreq(self.n, d=1 / self.fps)
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        if not np.any(mask):
            return float("nan")
        peak_freq = freqs[mask][np.argmax(spec[mask])]
        return float(peak_freq * 60)

    @property
    def median_hr(self):
        """ Медианный (ЧСС) HR (BPM) """
        return round(np.median(self.hr_from_peaks), 1) if self.hr_from_peaks.size else float("nan")

    @property
    def mean_hr(self):
        """ Средний (ЧСС) HR по пикам (BPM) """
        return  round(np.mean(self.hr_from_peaks), 1) if self.hr_from_peaks.size else float(
                "nan")

    @property
    def hrv(self):
        """ HRV – вариабельность сердечного ритма """
        return round(np.std(self.hr_from_peaks), 2) if self.hr_from_peaks.size else float(
                "nan")

    def summary(self) -> dict:
        """
        Возвращает сводную информацию о сигнале.
        """
        return {
            "HR по БПФ (BPM)": round(self.compute_fft_hr(), 1),
            "Средний HR по пикам (BPM)": self.mean_hr,
            "Медианный HR (BPM)": self.median_hr,
            "Ст. отклонение HR (HRV)": round(np.std(self.hr_from_peaks), 2) if self.hr_from_peaks.size else float(
                "nan"),
            "BPM по числу пиков": round(len(self.peaks) / self.duration_sec * 60, 1),
            "Число пиков": len(self.peaks),
            "Длительность сигнала (сек)": round(self.duration_sec, 1),
        }

    def plot_signal_with_peaks(self):
        """
        Отображает фильтрованный сигнал с обозначением пиков.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(self.filtered, label="Фильтрованный сигнал")
        plt.plot(self.peaks, self.filtered[self.peaks], 'ro', label="Пики")
        plt.title("Фильтрованный rPPG-сигнал с обнаруженными пиками")
        plt.xlabel("Индекс отсчёта")
        plt.ylabel("Амплитуда")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fft_spectrum(self):
        """
        Отображает амплитудный спектр фильтрованного сигнала.
        """
        centered = self.filtered - np.mean(self.filtered)
        spec = np.abs(rfft(centered))
        freqs = rfftfreq(self.n, d=1 / self.fps)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, spec)
        bpm = self.compute_fft_hr()
        peak_freq = bpm / 60
        plt.axvline(peak_freq, color='r', linestyle='--', label=f'Пик: {peak_freq:.2f} Гц = {bpm:.1f} BPM')
        plt.title("Амплитудный спектр rPPG-сигнала")
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import pandas as pd

    ppd = pd.read_csv("../data/SCAMPS_smail/output_data/ppg_1.csv").to_numpy().squeeze()
    analyzer = RPPGSignalAnalyzer(ppd, fps=30)

    print(analyzer.summary())
    analyzer.plot_signal_with_peaks()
    analyzer.plot_fft_spectrum()