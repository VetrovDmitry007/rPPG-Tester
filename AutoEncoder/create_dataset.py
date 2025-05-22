"""
Создание синтетического датасета для тестирования AutoEncoder
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Параметры синтетических кластеров
# -----------------------------
n_clusters = 5
records_per_cluster = 800  # 5 × 800 = 4000
rng = np.random.default_rng(seed=42)

centers = [
    [950, 60, 90, 80, 270, 195, 1.05, 190, 195, 1.0, 5.0],
    [850, 70, 70, 55, 210, 150, 0.95, 160, 160, 2.0, 4.0],
    [750, 80, 50, 35, 150, 110, 0.85, 140, 140, 3.0, 3.0],
    [650, 90, 25, 12, 90, 65, 0.55, 120, 100, 4.0, 2.0],
    [600, 100, 10, 5, 55, 35, 0.30, 110, 80, 5.0, 1.0],
]

stds = [40, 3, 5, 4, 20, 15, 0.05, 15, 15, 0.2, 0.4]
columns = ['mean_rr', 'median_hr', 'pnn20', 'pnn50', 'rmssd',
           'sd1', 'sd1_sd2', 'sd2', 'sdnn', 'csi', 'cvi']

data = []
labels_true = []

for idx, center in enumerate(centers):
    cluster_data = rng.normal(loc=center, scale=stds, size=(records_per_cluster, len(center)))
    data.append(cluster_data)
    labels_true.extend([idx] * records_per_cluster)

data = np.vstack(data)
df = pd.DataFrame(data, columns=columns)

# -----------------------------
# 2. Кластеризация для проверки
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
labels_pred = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, labels_pred, sample_size=1000, random_state=42)

# -----------------------------
# 3. Вывод результатов
# -----------------------------
print(f"Silhouette score with k=5 (sampled 1000): {sil_score:.3f}")
print("\nCluster sizes (predicted):", np.bincount(labels_pred))
print("Cluster sizes (ground truth):", np.bincount(labels_true))

# Показать первые 12 строк датасета
print("\nПервые 12 строк синтетического датасета:")
print(df.head(12))

df.to_csv('dataset_to_autoencoder.csv', index=False)
