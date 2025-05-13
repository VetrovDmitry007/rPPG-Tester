"""
Адаптер для модели DeepPhys/DeepPhys
https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/model/DeepPhys.py
"""
import cv2
import torch
import numpy as np

from rppg_benchmark.models.DeepPhys.DeepPhys import DeepPhys
from rppg_benchmark.interfaces import IRPPGModel
from rppg_benchmark.rppg_analyzer import RPPGSignalAnalyzer

class DeepPhysAdapter(IRPPGModel):
    def __init__(self):
        self.model = DeepPhys(img_size=72)
        ckpt_path = "rppg_benchmark/models/DeepPhys/SCAMPS_DeepPhys.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.frames = []

        # Буферы для предыдущего кадра и выходного сигнала
        self._prev_frame = None
        self._ppg_buffer: list[float] = []

    def reset(self) -> None:
        """Сброс внутреннего состояния перед новым сеансом."""
        self._prev_frame = None
        self._ppg_buffer.clear()

    def load_dataset(self, dataset):
        self.frames = list(dataset)

    def process_frame(self, frame_rgb, fps=30, ts=None):
        self.frames.append(frame_rgb)

    def get_ppg(self) -> np.ndarray:
        """
        Возвращает накопленный rPPG-сигнал как numpy-массив 1-D float32.
        """
        x = self._preprocess(self.frames)  # [T−1, 6, 128, 128]
        print("x.shape =", x.shape)
        with torch.no_grad():
            out = self.model(x)  # [T−1, 1]
        rppg = out.squeeze().cpu().numpy()  # [T−1]
        return rppg.astype(np.float32)

    def _preprocess(self, frames):
        """Готовит вход для DeepPhys: [T−1, 6, 72, 72] — [diff RGB | orig RGB]"""

        preprocessed = []
        for i in range(1, len(frames)):
            # кадр i-1 и i
            prev = cv2.cvtColor(cv2.resize(frames[i - 1], (72, 72)), cv2.COLOR_BGR2RGB)
            curr = cv2.cvtColor(cv2.resize(frames[i], (72, 72)), cv2.COLOR_BGR2RGB)

            prev = prev.astype(np.float32) / 255.0
            curr = curr.astype(np.float32) / 255.0

            diff = curr - prev  # разность между кадрами
            stacked = np.concatenate([diff, curr], axis=2)  # [72, 72, 6]
            preprocessed.append(stacked)

        arr = np.stack(preprocessed, axis=0)  # [T−1, 72, 72, 6]
        arr = arr.transpose(0, 3, 1, 2)  # [T−1, 6, 72, 72]
        return torch.tensor(arr, dtype=torch.float32)

    def get_hr(self, fps: float):
        """ Извлечение параметров ЧСС (Heart Rate, BPM) из предсказанного временного ряда

        """
        pass