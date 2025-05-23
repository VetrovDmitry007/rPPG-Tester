"""
Классификация HRV-словаря при помощи обученного классификатора KMeans
"""
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from model_autoencoder import HRVAutoEncoder

# 1. Константный порядок признаков
COLS: list[str] = [
    "mean_rr", "median_hr", "pnn20", "pnn50", "rmssd",
    "sd1", "sd1_sd2", "sd2", "sdnn", "csi", "cvi"]

scaler = joblib.load("hrv_std_scaler.pkl")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HRVAutoEncoder()
model.encoder.load_state_dict(torch.load('hrv_encoder.pt'))
model.eval()

def hrv_to_z(hrv_dict: dict[str, float]):
    """Преобразует необработанный HRV-словарь → стандартизированный → 4‑мерный z."""
    # 3.1 словарь → ndarray формы (1, 11)
    x_raw = np.array([[hrv_dict[col] for col in COLS]], dtype=np.float32)

    # 3.2 стандартизация (z-score)
    x_std = scaler.transform(x_raw)

    # 3.3 энкодер → z вектор
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(x_std).to(device)).cpu().numpy()
    return z.ravel()


def f_1():
    # ---- 2.2 создаём z-матрицу для всего датасета ----
    df = pd.read_csv("dataset_to_autoencoder.csv")
    Z = np.vstack([hrv_to_z(r._asdict()) for r in df.itertuples(index=False)])
    print(Z)

if __name__ == '__main__':
    f_1()