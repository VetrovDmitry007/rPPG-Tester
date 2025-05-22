"""
Модуль сравнения эталонных и прогнозируемых данных rPPG
"""
from pprint import pprint

import pandas as pd
from emo_state.standard_metrics import show_metrics
from emo_state.hrv_classifier import classify_emotional_state
from rppg_benchmark.datasets import VideoDataset
from rppg_tester import dynamic_import
from emo_state.metrics import *
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
# print(pred_ppg)

show_metrics(pred_ppg=pred_ppg, ref_ppg=ref_ppg, fps=dataset.fps)

analyzer = RPPGSignalAnalyzer(pred_ppg, fps=dataset.fps)
print(analyzer.summary())
index_emo, text = classify_emotional_state(analyzer.index_hrv)
pprint(analyzer.index_hrv)
print(f'{index_emo=}, {text}')