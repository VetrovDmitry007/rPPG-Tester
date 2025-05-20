"""
Модуль сравнения эталонных и прогнозируемых данных rPPG
"""
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt

from rppg_benchmark.datasets import VideoDataset
from rppg_benchmark.rppg_extractor import RawPPGFeatureExtractor
from rppg_tester import dynamic_import
from rppg_benchmark.metrics import *
from config_models import MODEL_PATH, video_path, path_csv


# Загрузка модели
ModelCls = dynamic_import(MODEL_PATH)
model = ModelCls()
# print(model.model)

dataset = VideoDataset(video_path)
print(f"FPS: {dataset.fps}")
model.reset()
model.load_dataset(dataset)
# Получение предсказанного сигнала
pred_ppg = model.get_ppg()
# Получение эталонного сигнала из CSV
ref_ppg  = pd.read_csv(path_csv).to_numpy().squeeze()

# Выравнивание длины сигнала
min_len = min(len(ref_ppg), len(pred_ppg))
ref_ppg  = ref_ppg[:min_len]
pred_ppg = pred_ppg[:min_len]
print(pred_ppg)

# Вычислить стандартные метрики ошибки и сходства
print("MAE       =", mae(pred_ppg, ref_ppg))
print("MAPE       =", mape(pred_ppg, ref_ppg))
print("SMAPE       =", smape(pred_ppg, ref_ppg))
print("RMSE      =", rmse(pred_ppg, ref_ppg))
print("Pearson r =", corr(pred_ppg, ref_ppg))
print("SNR (дБ)  =", snr(pred_ppg, ref_ppg))
print(f"\nСредний (ЧСС) HR по пикам (BPM): ref = {ref_mean_hr(ref_ppg, fps=dataset.fps)}, pred = {pred_mean_hr(pred_ppg, fps=dataset.fps)} ")
print(f'Ошибка среднего (ЧСС) HR (BPM): {errors_mean_hr(ref_ppg, pred_ppg, fps=dataset.fps)}%')
print(f"Медианный (ЧСС) HR (BPM): ref = {ref_median_hr(ref_ppg, fps=dataset.fps)}, pred = {pred_median_hr(pred_ppg, fps=dataset.fps)} ")
print(f'Ошибка медианного (ЧСС) HR (BPM): {errors_median_hr(ref_ppg, pred_ppg, fps=dataset.fps)}%')
print(f'Вариабельность сердечного ритма HRV: ref = {ref_hrv(ref_ppg, fps=dataset.fps)}, pred = {pred_hrv(pred_ppg, fps=dataset.fps)}')

# Визуальная оценка
# plt.figure(figsize=(10,4))
# plt.plot(ref_ppg,  label="Эталонный PPG")
# plt.plot(pred_ppg, label="Предсказанный PPG")
# plt.legend()
# plt.xlabel("Отсчёт")
# plt.ylabel("Амплитуда")
# plt.title("Сравнение эталонного и предсказанного PPG")
# plt.tight_layout()
# plt.show()

rppg_metrics = RawPPGFeatureExtractor(ref_ppg, dataset.fps)
res = rppg_metrics.review()
pprint(f'\n{res}')
