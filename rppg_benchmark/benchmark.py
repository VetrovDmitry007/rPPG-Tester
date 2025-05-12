"""Core engine that evaluates an IRPPGModel on a dataset."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .interfaces import IRPPGModel
from .metrics import corr, mae, rmse, snr


class RPPGBenchmark:  # pylint: disable=too-few-public-methods
    """Runs a model through a dataset and computes quality metrics."""

    def __init__(self, dataset, fps: float = 30.0) -> None:
        self.dataset = dataset
        self.fps = float(fps)

    def evaluate(self, model: IRPPGModel, name: str | None = None) -> Dict[str, float]:
        model.reset()
        preds: list[float] = []
        refs: list[float] = []

        for frame_rgb, ref_bpm in self.dataset:
            model.process_frame(frame_rgb)
            hr = model.get_hr(self.fps)
            if np.isfinite(hr):
                preds.append(hr)
                refs.append(ref_bpm)

        preds_arr = np.asarray(preds, dtype=np.float32)
        refs_arr = np.asarray(refs, dtype=np.float32)

        return {
            "model": name or model.__class__.__name__,
            "samples": len(preds_arr),
            "MAE": mae(preds_arr, refs_arr),
            "RMSE": rmse(preds_arr, refs_arr),
            "Pearson r": corr(preds_arr, refs_arr),
            "SNR": snr(preds_arr, refs_arr),
        }
