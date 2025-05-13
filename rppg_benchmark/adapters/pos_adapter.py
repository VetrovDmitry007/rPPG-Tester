"""
Адаптер неконтролируемого метода
https://github.com/ubicomplab/rPPG-Toolbox/blob/main/unsupervised_methods/methods/POS_WANG.py
"""
import cv2
import torch
import numpy as np

from rppg_benchmark.models.POS.POS_WANG import POS_WANG
from rppg_benchmark.interfaces import IRPPGModel

class PosAdapter(IRPPGModel):
    def __init__(self):
        self.frames = []

    def reset(self) -> None:
        """Сброс внутреннего состояния перед новым сеансом."""
        self.frames.clear()

    def load_dataset(self, dataset):
        self.frames = list(dataset)

    def process_frame(self, frame_rgb, fps=30, ts=None):
        pass

    def get_ppg(self) -> np.ndarray:
        """
        Возвращает накопленный rPPG-сигнал как numpy-массив 1-D float32.
        """
        BVP = POS_WANG(self.frames, 30)
        print("BVP.shape =", BVP.shape)
        return BVP



    def get_hr(self, fps: float):
        """ Извлечение параметров ЧСС (Heart Rate, BPM) из предсказанного временного ряда

        """
        pass