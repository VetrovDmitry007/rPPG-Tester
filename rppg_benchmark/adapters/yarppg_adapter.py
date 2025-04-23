"""Адаптер, который подключает библиотеку `yarppg` к irppgmodel."""
from __future__ import annotations

from typing import List

import numpy as np
import yarppg  # type: ignore  # third‑party C++/cython bindings

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
        # assume `res.raw` holds raw rPPG value per frame; adjust if API differs
        if hasattr(res, "raw") and res.raw is not None:
            self._buf.append(float(res.raw))

    def get_ppg(self) -> np.ndarray:  # noqa: D401
        return np.asarray(self._buf, dtype=np.float32)