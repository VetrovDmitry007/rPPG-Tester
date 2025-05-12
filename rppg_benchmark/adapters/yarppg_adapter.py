"""
Адаптер, который подключает библиотеку `yarppg` к irppgmodel.
"""
from __future__ import annotations
from typing import List
import numpy as np
import yarppg  # pip install git+https://github.com/SamProell/yarppg.git

from ..interfaces import IRPPGModel


class YarppgAdapter(IRPPGModel):  # pylint: disable=too-few-public-methods
    """Thin wrapper around `yarppg.Rppg()` core."""

    def __init__(self):
        self._core = yarppg.Rppg()
        self._buf: List[float] = []

    # interface impl. ─────────────────────────
    def reset(self) -> None:
        self._core = yarppg.Rppg()
        self._buf.clear()

    def process_frame(self, frame_rgb: np.ndarray, fps, ts: float | None = None) -> None:
        """ Заполняет буфер сигналом RPPG.

        :params frame_rgb: RGB image in H×W×3 layout.
        :params ts: Optional timestamp (seconds).
        :params fps: Частота кадров.
        """
        res = self._core.process_frame(frame_rgb)
        if hasattr(res, "hr") and res.hr is not None and res.hr > 0:
            bpm = 60.0 * fps / res.hr
            print(f"{fps=}, HR: {bpm:.1f} BPM")
            self._buf.append(float(bpm))

    def get_ppg(self) -> np.ndarray:  # noqa: D401
        return np.asarray(self._buf, dtype=np.float32)

    def get_hr(self, fps: float) -> float:
        hr_array = np.asarray(self._buf, dtype=np.float32)
        if hr_array.size < fps * 2:
            return float("nan")

        mu = hr_array.mean()
        sigma = hr_array.std()
        filtered = hr_array[np.abs(hr_array - mu) <= 3 * sigma]

        if filtered.size == 0:
            return float("nan")

        return float(filtered.mean())  # или np.median(filtered) если предпочтительна медиана