"""
Извлекает из .mat файла видео и csv файлы.
"""

# pip install h5py

import h5py
import numpy as np
import pandas as pd
import cv2
import os

mat_path = "../data/SCAMPS_smail/P000001.mat"
out_dir = "../data/SCAMPS_smail/output_data"
os.makedirs(out_dir, exist_ok=True)

with h5py.File(mat_path, 'r') as f:
    raw = np.array(f['RawFrames'])  # (3, 320, 240, 600)
    d_ppg = np.array(f['d_ppg'])    # (600, 1)

# Переставим оси: (C, H, W, T) → (T, H, W, C)
video = np.transpose(raw, (3, 1, 2, 0))  # → (600, 320, 240, 3)

# Приводим к диапазону [0, 255] и uint8, если нужно
video = np.clip(video, 0, 1)  # если значения уже в [0,1]
video = (video * 255).astype(np.uint8)

# Сохраняем видео
h, w = video.shape[1:3]
fps = 30
writer = cv2.VideoWriter(os.path.join(out_dir, "video.avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
for frame in video:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()

# Сохраняем PPG
pd.DataFrame({'ppg': d_ppg.flatten()}).to_csv(os.path.join(out_dir, 'ppg.csv'), index=False)

print("✅ Готово: видео и CSV сохранены.")