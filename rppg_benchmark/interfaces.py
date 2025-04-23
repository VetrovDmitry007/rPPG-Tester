"""Определяет общий интерфейс, которую должна следовать каждая модель RPPG."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class IRPPGModel(ABC):
    """Аннотация базового класса для моделей RPPG.

    Один и тот же контракт используется как автономным контрольным двигателем, так и
    Реконструктивный тестер консоли. Любая конкретная реализация или адаптер должны
    реализовывать, по крайней мере, эти три основных метода. Удобство `get_hr`
    при условии, но может быть переопределено для специфической для модели HR экстракцию.
    """

    # ── Streaming API ──────────────────────────────────
    @abstractmethod
    def reset(self) -> None:
        """Сбросьте все внутренние буферы и состояние перед новым сеансом/видео."""

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray, ts: float | None = None) -> None:
        """Потреблять одну рамку RGB (H×W×3, uint8 or float32).

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB image in H×W×3 layout.
        ts : float | None
            Optional timestamp (seconds). The default pipeline ignores it.
        """

    @abstractmethod
    def get_ppg(self) -> np.ndarray:
        """Верните накопленный сигнал RPPG в виде массива 1 -D Float32."""

    # ── Convenience helpers ────────────────────────────
    def get_hr(self, fps: float) -> float:  # noqa: D401
        """Вычислить мгновенный HR (BPM) из текущего сигнала.

        Реализация по умолчанию находит доминирующий пик FFP и преобразует его
        к BPM. Переопределить для более сложных методов (например, Band -Pass,
        Обнаружение пика, фильтрация Калмана…).
        """
        signal = self.get_ppg()
        if signal.size < fps * 2:  # need at least 2 s window
            return float("nan")

        ffreq = np.fft.rfftfreq(signal.size, d=1 / fps)
        spec = np.abs(np.fft.rfft(signal - signal.mean()))
        peak_hz = ffreq[np.argmax(spec)]
        return float(peak_hz * 60.0)