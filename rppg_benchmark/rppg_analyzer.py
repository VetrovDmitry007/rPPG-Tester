import numpy as np
import pandas as pd
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

    def __init__(self, ppg: np.ndarray, fps: float = 30.0):
        """
        Инициализация на основе DataFrame с одним столбцом сигнала.
        :param ppg: Масcив 1-D с временным рядом rPPG (сырые значения rPPG)
        :param fps: Частота кадров, с которой был получен сигнал
        """
        self.signal =  ppg.astype(np.float32)
        self.fps = fps
        self.n = len(self.signal)
        self.duration_sec = self.n / self.fps

        # Фильтрация сигнала в допустимом диапазоне (0.7–4 Гц)
        self.filtered = self._bandpass_filter(self.signal)

        # Ищутся локальные максимумы (пики), соответствующие ударам сердца
        self.peaks, _ = find_peaks(self.filtered, distance=self.fps * 0.5)
        # Переводим индексы пиков (в отсчётах) во время (в секундах).
        self.peak_times = self.peaks / self.fps
        # ibi -- время между соседними ударами (в секундах !!!)
        self.ibi = np.diff(self.peak_times)
        self.hr_from_peaks = 60 / self.ibi if len(self.ibi) > 0 else np.array([])

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