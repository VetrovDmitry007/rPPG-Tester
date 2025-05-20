import numpy as np
from scipy.signal import welch, find_peaks, detrend, butter, filtfilt

class RawPPGFeatureExtractor:
    """
    Извлечение 16 метрик из сырых rPPG‑сигналов для ML‑классификации
    психоэмоционального состояния.

    ────────── Перечень метрик и их относительная важность (0‑10) ──────────
      Амплитудные (качество / микроциркуляция)
      ▸ ampl_min[2]  – минимальное значение сигнала
      ▸ ampl_max[3]  – максимальное значение сигнала
      ▸ ampl_span[4]  – размах (max‑min), индикатор вариабельности волны
      ▸ ampl_mean[2]  – среднее значение (DC‑уровень)
      ▸ ampl_median[2]  – медиана (устойчива к выбросам)

      Частотные (автономная регуляция)
      ▸ psd_vlf[4]  – мощность 0.003‑0.04Гц (вазомоторика)
      ▸ psd_lf[7]  – мощность 0.04‑0.15Гц (симпато‑тония)
      ▸ psd_hf[6]  – мощность 0.15‑0.40Гц (парасимпатика / дыхание)
      ▸ ratio_lf_hf[9]  – LF/HF, ключевой стресс‑индекс
      ▸ ratio_lf_vlf[6]  – LF/VLF, отражает медленные колебания

      Артефакты (качество сигнала)
      ▸ clip_min_ratio [2]  – доля отсчётов на минимуме (cut‑off снизу)
      ▸ clip_max_ratio [2]  – доля отсчётов на максимуме (cut‑off сверху)

      HRV‑метрики (адаптивность ЦНС)
      ▸ hr_mean[8]  – средний пульс, BPM
      ▸ sdnn[9]  – SDNN, общая вариабельность NN‑интервалов
      ▸ rmssd[9]  – RMSSD, мгновенная парасимпатическая активность
      ▸ pnn50[8]  – % разностей NN >50мс, вагусная активность

    Метрики «psd_vlf», «psd_lf», «ratio_*» могли возвращать 0, если
    длина записи <60с или сигнал не был центрирован. В новой версии
    добавлена предобработка (детренд + полосовой фильтр) и контроль
    достаточной длительности для надёжной оценки спектра.
    """

    # ──────────────────────────────── INIT ────────────────────────────────

    def __init__(self, signal: np.ndarray, fs: float = 30.0):
        """Parameters
        ----------
        signal : np.ndarray
            Сырые значения rPPG‑сигнала (shape ≈ [T]).
        fs : float, optional
            Частота дискретизации видео/сигнала, Гц (default 30).
        """
        self.signal = signal.astype(float)
        self.fs = fs
        # предвычислим центрированный сигнал для всех методов
        self._sig = detrend(self.signal - np.mean(self.signal))
        # длина записи в секундах
        self._duration = len(self.signal) / fs

    # ─────────────────────── 1. Амплитудные метрики ───────────────────────

    def _metric_amplitude(self):
        """Возвращает dict с:
        ampl_min, ampl_max, ampl_span, ampl_mean, ampl_median."""
        s = self.signal
        return {
            'ampl_min': np.min(s),
            'ampl_max': np.max(s),
            'ampl_span': np.ptp(s),
            'ampl_mean': np.mean(s),
            'ampl_median': np.median(s)
        }

    # ─────────────────────── 2. Частотные метрики ────────────────────────

    def _metric_frequency(self):
        """Вычисляет мощность спектра в VLF, LF, HF и их соотношения.

        Если запись короче 60 с, VLF считается недостоверным → psd_vlf = nan.
        """
        # минимальная длина для Welch: 8 сегментов по 8 с → 64 с, но позволим 60 с
        if self._duration < 60:
            return {
                'psd_vlf': np.nan,
                'psd_lf': np.nan,
                'psd_hf': np.nan,
                'ratio_lf_hf': np.nan,
                'ratio_lf_vlf': np.nan,
            }
        # лёгкий полосовой фильтр (0.003‑0.5 Гц) перед PSD
        b, a = butter(2, [0.003/(self.fs/2), 0.5/(self.fs/2)], btype='band')
        sig_filt = filtfilt(b, a, self._sig)

        nperseg = int(self.fs * 8)  # окна по 8 с для достаточного разрешения
        f, Pxx = welch(sig_filt, fs=self.fs, nperseg=nperseg, detrend='constant')
        bands = {
            'vlf': (0.003, 0.04),
            'lf':  (0.04,  0.15),
            'hf':  (0.15, 0.40),
        }
        feats = {}
        for name, (fmin, fmax) in bands.items():
            mask = (f >= fmin) & (f < fmax)
            power = np.trapz(Pxx[mask], f[mask])
            feats[f'psd_{name}'] = power if power > 0 else np.nan
        # соотношения (избегаем деления на 0 / nan)
        feats['ratio_lf_hf'] = (
            feats['psd_lf'] / feats['psd_hf'] if feats['psd_hf'] not in [0, np.nan] else np.nan
        )
        feats['ratio_lf_vlf'] = (
            feats['psd_lf'] / feats['psd_vlf'] if feats['psd_vlf'] not in [0, np.nan] else np.nan
        )
        return feats

    # ─────────────────────── 3. Метрики артефактов ────────────────────────

    def _metric_artifacts(self):
        """Определяет степень клиппинга сигнала сверху/снизу."""
        s = self.signal
        mn, mx = s.min(), s.max()
        total = len(s)
        return {
            'clip_min_ratio': np.sum(s == mn) / total,
            'clip_max_ratio': np.sum(s == mx) / total
        }

    # ───────────────────────── 4. HRV‑метрики ────────────────────────────

    def _metric_hrv(self):
        """Вычисляет hr_mean, sdnn, rmssd, pnn50 из последовательности RR."""
        # p‑p детекция с адаптивным порогом (prominence ≈ 0.5*IQR)
        iqr = np.subtract(*np.percentile(self._sig, [75, 25]))
        peaks, _ = find_peaks(self._sig, distance=int(self.fs*0.4), prominence=0.5*iqr)
        if len(peaks) < 2:
            return {k: np.nan for k in ['hr_mean', 'sdnn', 'rmssd', 'pnn50']}
        rr = np.diff(peaks) / self.fs  # сек
        hr = 60.0 / rr
        diff_rr = np.diff(rr)
        return {
            'hr_mean': np.mean(hr),
            'sdnn': np.std(rr) * 1000.0,  # мс
            'rmssd': np.sqrt(np.mean(diff_rr**2)) * 1000.0,  # мс
            'pnn50': np.mean(np.abs(diff_rr) > 0.05),
        }

    # ───────────────────────────── REVIEW ────────────────────────────────

    def review(self) -> dict:
        """Собирает 16 метрик в единый словарь."""
        feats = {}
        feats.update(self._metric_amplitude())
        feats.update(self._metric_frequency())
        feats.update(self._metric_artifacts())
        feats.update(self._metric_hrv())
        return feats

# -------------------------- Пример использования --------------------------
# raw_ppg = np.load('example.npy')
# extractor = RawPPGFeatureExtractor(raw_ppg, fs=30)
# features = extractor.review()
# print(features)
