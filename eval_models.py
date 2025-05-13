"""
Модуль сравнения эталонных и прогнозируемых данных rPPG
"""

import pandas as pd
import matplotlib.pyplot as plt

from rppg_benchmark.datasets import VideoDataset
from rppg_benchmark.rppg_analyzer import RPPGSignalAnalyzer
from rppg_tester import dynamic_import
from rppg_benchmark.metrics import mae, rmse, corr, snr, mape, smape

# path_csv = "data/SCAMPS_smail/output_data/ppg_1.csv"
# video_path = "data/SCAMPS_smail/output_data/video_1.avi"

# path_csv = "data/sample/sample_vitals_2.csv"
# video_path = "data/sample/sample_video_2.mp4"

path_csv = "data/UBFC-Phys/bvp_s11_T1.csv"
video_path = "data/UBFC-Phys/vid_s11_T1.avi"

# supervised_methods
MODEL_PATH = "rppg_benchmark.adapters.phys_former_adapter:PhysFormerAdapter"
# MODEL_PATH = "rppg_benchmark.adapters.deep_phys_adapter:DeepPhysAdapter"

# unsupervised_methods
# MODEL_PATH = "rppg_benchmark.adapters.pos_adapter:PosAdapter"
# MODEL_PATH = "rppg_benchmark.adapters.chrome_adapter:ChromeAdapter"


ModelCls = dynamic_import(MODEL_PATH)
model = ModelCls()
dataset = VideoDataset(video_path)
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

ref_analyzer = RPPGSignalAnalyzer(ref_ppg, fps=30)
pred_analyzer = RPPGSignalAnalyzer(pred_ppg, fps=30)

# Вычислить стандартные метрики ошибки и сходства
print("MAE       =", mae(pred_ppg, ref_ppg))
print("MAPE       =", mape(pred_ppg, ref_ppg))
print("SMAPE       =", smape(pred_ppg, ref_ppg))
print("RMSE      =", rmse(pred_ppg, ref_ppg))
print("Pearson r =", corr(pred_ppg, ref_ppg))
print("SNR (дБ)  =", snr(pred_ppg, ref_ppg))
print(f"Средний (ЧСС) HR по пикам (BPM): ref = {ref_analyzer.mean_hr}, pred = {pred_analyzer.mean_hr} ")
print(f"Медианный (ЧСС) HR (BPM): ref = {ref_analyzer.median_hr}, pred = {pred_analyzer.median_hr} ")

# Визуальная оценка
plt.figure(figsize=(10,4))
plt.plot(ref_ppg,  label="Эталонный PPG")
plt.plot(pred_ppg, label="Предсказанный PPG")
plt.legend()
plt.xlabel("Отсчёт")
plt.ylabel("Амплитуда")
plt.title("Сравнение эталонного и предсказанного PPG")
plt.tight_layout()
plt.show()

