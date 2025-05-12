"""Lightweight dataset helpers for frame and video input."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import warnings

# Простая загрузка датасетов: одиночные кадры или видео + PPG


class FrameDataset:  # pylint: disable=too-few-public-methods
    """Загружает файлы изображений с одним кадром и эталонные значения HR.

    Ожидаемая картина: `name_someinfo_hr_<number>.png`, где <число> —
    либо целое число BPM или число с плавающей точкой.
    """
    def __init__(self, root: str | Path):
        self.samples: List[Tuple[np.ndarray, float]] = []
        root = Path(root)
        for img_path in sorted(root.glob("*.png")):
            *_, bpm_token = img_path.stem.split("_hr_")
            ref_bpm = float(bpm_token)
            frame_bgr = cv2.imread(str(img_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.samples.append((frame_rgb, ref_bpm))

    def __iter__(self) -> Iterable[Tuple[np.ndarray, float]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


class VideoWaveformDataset:  # pylint: disable=too-few-public-methods
    """Загружает кадры видео и истинный PPG-сигнал из CSV-файла.

    В итераторе отдаёт пары (frame_rgb, ppg_value) для каждого кадра.
    """
    def __init__(self, video_path: str | Path, vitals_path: str | Path):
        video_path = Path(video_path)
        vitals_path = Path(vitals_path)

        # Проверяем наличие CSV-файла
        if not vitals_path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {vitals_path}")
        df = pd.read_csv(vitals_path)
        if 'ppg' not in df.columns:
            raise ValueError("CSV должен содержать колонку 'ppg'")
        ppg_array = df['ppg'].to_numpy(dtype=np.float32)

        # Считываем видео-кадры
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {video_path}")
        frames: List[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        # Выравниваем длины
        n = min(len(frames), len(ppg_array))
        if len(frames) != len(ppg_array):
            warnings.warn(
                f"Число кадров ({len(frames)}) и длина PPG-сигнала ({len(ppg_array)}) не совпадают; используется первые {n} элементов."
            )

        # Формируем список образцов
        self.samples: List[Tuple[np.ndarray, float]] = [
            (frames[i], float(ppg_array[i])) for i in range(n)
        ]

    def __iter__(self) -> Iterable[Tuple[np.ndarray, float]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def load_dataset(
    data_path: str | Path,
    vitals_csv: Optional[str | Path] = None
    ) -> FrameDataset | VideoWaveformDataset:
    """
    Фабрика датасетов. По расширению файла определяет, какой класс вернуть.

    :param data_path: путь к папке с PNG или к видеофайлу (.mpg, .mp4, .avi)
    :param vitals_csv: путь к CSV с сырыми данными PPG; обязателен для видео
    """
    path = Path(data_path)
    ext = path.suffix.lower()
    if ext in ('.mpg', '.mp4', '.avi'):
        if vitals_csv is None:
            raise ValueError("Для видео требуется указать vitals_csv с PPG-сигналом")
        return VideoWaveformDataset(path, vitals_csv)
    else:
        return FrameDataset(path)

__all__ = [
    "FrameDataset",
    "VideoWaveformDataset",
    "load_dataset",
]
