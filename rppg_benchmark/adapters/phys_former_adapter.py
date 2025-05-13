"""
Адаптер для модели PhysFormer/ViT_ST_ST_Compact3_TDC_gra_sharp
https://github.com/ZitongYu/PhysFormer
"""
import cv2
import torch
import numpy as np

from rppg_benchmark.models.PhysFormer.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from rppg_benchmark.interfaces import IRPPGModel
from rppg_benchmark.rppg_analyzer import RPPGSignalAnalyzer


class PhysFormerAdapter(IRPPGModel):
    def __init__(self):
        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128),
                                                      patches=(4,4,4),
                                                      dim=96,
                                                      ff_dim=144,
                                                      num_heads=4,
                                                      num_layers=12,
                                                      dropout_rate=0.1,
                                                      theta=0.7)
        ckpt_path = "rppg_benchmark/models/PhysFormer/PURE_PhysFormer_DiffNormalized.pth"
        # ckpt_path = "rppg_benchmark/models/PhysFormer/Physformer_VIPL_fold1.pkl"
        # ckpt_path = "rppg_benchmark/models/PhysFormer/SCAMPS_PhysFormer_DiffNormalized.pth"
        # ckpt_path = "rppg_benchmark/models/PhysFormer/UBFC-rPPG_PhysFormer_DiffNormalized.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.frames = []

    def reset(self):
        self.frames.clear()

    def load_dataset(self, dataset):
        self.frames = list(dataset)

    def process_frame(self, frame_rgb, fps=30, ts=None):
        self.frames.append(frame_rgb)

    def get_ppg(self) -> np.array:
        """ Возвращает предсказанный временной ряд, отражающим фотоплетизмографическую волну,
         извлечённую моделью PhysFormer из видеопоследовательности

        gra_sharp -- управляет резкостью (sharpness) распределения внимания (attention).
        - Чем меньше gra_sharp, тем резче (пиковее) будет attention (softmax станет круче).
        - Чем больше gra_sharp, тем плавнее и расфокусированнее attention.
        """
        ls_res = []

        if len(self.frames) < 160:
            return np.array([], dtype=np.float32)

        while len(self.frames) > 160:
            frames_0 = self.frames[:160]

            x = self._preprocess(frames_0)
            print(f"x.shape = {x.shape}") # Отладка — размерность тензора
            with torch.no_grad():
                rppg, *_ = self.model(x, gra_sharp=2.0)

            # Преобразуем в 1-D numpy и добавляем к общему сигналу
            ls_res.extend(rppg.squeeze().cpu().tolist())

            self.frames = self.frames[160:]

        self.raw_ppd = np.array(ls_res, dtype=np.float32)
        return self.raw_ppd


    def _preprocess(self, frames):
        """ Подготавливает данные к формату модели ViT_ST_ST_Compact3_TDC_gra_sharp — это 5D-тензор

        1. Объединяет список кадров в единый 4D-массив формы (T, H, W, 3)
            3 — каналы RGB,
            T — количество кадров,
            H, W — размер кадра.
        2. Меняет порядок осей на (C, T, H, W), добавляя размерность батча(1)
        3. Нормализует значения пикселей из диапазона [0, 255] в [0.0, 1.0]
        4. Преобразует numpy-массив в torch-тензор
        """
        # Приводим каждый кадр к нужному размеру 128x128
        resized = [cv2.resize(f, (128, 128)) for f in frames]
        arr = np.stack(resized, axis=0)
        arr = arr.transpose(3, 0, 1, 2)[None]
        arr = arr.astype(np.float32) / 255.0
        return torch.tensor(arr)

    def get_hr(self, fps: float):
        """ Извлечение параметров ЧСС (Heart Rate, BPM) из предсказанного временного ряда

        """
        analyzer = RPPGSignalAnalyzer(self.raw_ppd, fps=fps)
        print(analyzer.summary())
        analyzer.plot_signal_with_peaks()
        analyzer.plot_fft_spectrum()


if __name__ == "__main__":
    adapter = PhysFormerAdapter()
    print(adapter.model)

