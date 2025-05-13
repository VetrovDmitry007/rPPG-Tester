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
    def load_dataset(self, dataset) -> None:
        """Загрузка данных. """

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray, fps, ts: float | None = None) -> None:
        """Обработка одного кадра RGB (H×W×3, uint8 or float32).
        Заполняет буфер сигналом RPPG.

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB image in H×W×3 layout.
        ts : float | None
            Optional timestamp (seconds). The default pipeline ignores it.
        """

    @abstractmethod
    def get_ppg(self) -> np.ndarray:
        """Возвращает предсказанный временной ряд, отражающим фотоплетизмографическую волну
         в виде массива 1-D Float32."""

    @abstractmethod
    def get_hr(self, fps: float) -> float:
        """Вычислить мгновенный HR (BPM) из текущего сигнала. """
