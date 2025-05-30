"""
Что здесь происходит
1. hrv_to_z: стандартизируем входной HRV-словарь и пропускаем через энкодер.
2. Собираем все z-векторы датасета → обучаем KMeans(K=5).
3. Группируем исходные показатели по метке кластера и смотрим, чем они различаются.

4. Делаем прогноз на обученом автоэнкодер + класторезатор
"""

import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from fit_clusters import hrv_to_z

df = pd.read_csv("dataset_to_autoencoder.csv")
Z = np.vstack([hrv_to_z(r._asdict()).astype(np.float64) for r in df.itertuples(index=False)])


# ---- 2.3 обучаем кластеризатор ----
kmeans = KMeans(n_clusters=5, random_state=42).fit(Z)
print("silhouette:", silhouette_score(Z, kmeans.labels_, sample_size=1000))

# ---- 2.4 интерпретация кластеров ----
df["cluster"] = kmeans.labels_
summary = df.groupby("cluster").agg({
    "median_hr": ["mean", "std"],
    "rmssd":     ["mean"],
    "pnn50":     ["mean"],
    "csi":       ["mean"]
})
print(summary)

# ---- 2.5 онлайн-функция ----
CLUSTER_LABELS = {
    0: "Relaxed",          # вручную после анализа summary
    1: "Moderately calm",
    2: "Neutral",
    3: "Moderate stress",
    4: "High stress"
}

def emotion_from_hrv(hrv_dict: dict[str, float]) -> str:
    z = hrv_to_z(hrv_dict).astype(np.float64)
    print("dtype z:", z.dtype)
    cid = kmeans.predict([z])[0]
    return CLUSTER_LABELS[cid]


if __name__ == '__main__':
    sample = {
        'mean_rr': 811.1, 'median_hr': 72.1, 'pnn20': 91.3, 'pnn50': 78.3,
        'rmssd': 278.8, 'sd1': 197.1, 'sd1_sd2': 1.0,
        'sd2': 196.9, 'sdnn': 197.0, 'csi': 1.0, 'cvi': 5.19
    }

    state = emotion_from_hrv(sample)
    print(f"Эмоциональное состояние: {state}")
