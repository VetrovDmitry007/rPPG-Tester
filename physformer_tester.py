"""
Пример использования модели PhysFormer/ViT_ST_ST_Compact3_TDC_gra_sharp

Результат работы модели -- предсказанный временной ряд, отражающим фотоплетизмографическую волну,
         извлечённую моделью PhysFormer из видеопоследовательности
"""
import cv2
from rppg_tester import dynamic_import
from rppg_benchmark.datasets import load_dataset

MODEL_PATH = "rppg_benchmark.adapters.phys_former_adapter:PhysFormerAdapter"
ModelCls = dynamic_import(MODEL_PATH)
model = ModelCls()

# print(model.model)

model.reset()

datasets = load_dataset('data/SCAMPS_smail/output_data/video.avi')
model.load_dataset(datasets)

ppg = model.get_ppg()
print(ppg.shape, ppg)  # например (160,), [0.13, 0.09, ...]
model.get_hr(fps=30)