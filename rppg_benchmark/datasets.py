"""Lightweight dataset helpers for frame and video input."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# Простой шаблон имени файла:  image_<idx>_hr_<BPM>.png


class FrameDataset:  # pylint: disable=too-few-public-methods
    """Загружает файлы изображений с одним рамкой и эталонные значения HR.

    Ожидаемая картина: `name_someinfo_hr_<number>.png` где <число> есть
    либо целое число BPM, либо плавающая точка.
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

    def __iter__(self) -> Iterable[Tuple[np.ndarray, float]]:  # noqa: D401
        return iter(self.samples)