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
    def reset(self) -> None:  # noqa: D401
        self._core = yarppg.Rppg()
        self._buf.clear()

    def process_frame(self, frame_rgb: np.ndarray, ts: float | None = None) -> None:  # noqa: D401,E501
        res = self._core.process_frame(frame_rgb)
        if hasattr(res, "hr") and res.hr is not None and res.hr > 0:
            fps = 30.0  # предполагаемая частота кадров (или вычисляем динамически)
            bpm = 60.0 * fps / res.hr
            print(f"HR: {bpm:.1f} BPM")
            self._buf.append(float(bpm))

    def get_ppg(self) -> np.ndarray:  # noqa: D401
        return np.asarray(self._buf, dtype=np.float32)