"""
Стандартизация (z-нормализация) 11-мерного HRV-датасета.
✓ сохраняем обученный scaler → можно применять к новым записям
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib                           # для сохранения scaler

# # ------------ 1. читаем (или получаем) сырые данные ------------
# df = pd.read_csv("dataset_to_autoencoder.csv")      # shape = (4000, 11)
#
# # ------------ 2. обучаем StandardScaler ------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df.values)      # ndarray (4000, 11)
#
# # ------------ 3. сохраняем scaler на диск ------------
# joblib.dump(scaler, "hrv_std_scaler.pkl")
#
# # ------------ 4. превращаем обратно в DataFrame (по желанию) ------------
# df_scaled = pd.DataFrame(X_scaled, columns=df.columns)
# print(df_scaled.head())
# df_scaled.to_csv("dataset_to_autoencoder_scaled.csv", index=False)

# ===============================================================
#        Использование сохранённого скейлера для новых данных
# ===============================================================
def hrv_standardize(sample: dict[str, float]) -> np.ndarray:
    """
    Во входе — сырой HRV-словарь (11 ключей в том же порядке, что и в df.columns).
    Возвращает стандартизованный numpy-вектор shape (11,).
    """
    df = pd.DataFrame(sample, index=[0])

    # загружаем scaler (один раз где-то при старте скрипта)
    scaler_loaded = joblib.load("hrv_std_scaler.pkl")

    # превращаем словарь → массив формы (1, 11)
    x = np.array([[sample[col] for col in df.columns]], dtype=np.float32)

    # стандартизируем
    return scaler_loaded.transform(x).ravel()

# пример
sample_std = hrv_standardize({
    'mean_rr': 811.111111111111,
    'median_hr': 72.1,
    'pnn20': 91.30434782608695,
    'pnn50': 78.26086956521739,
    'rmssd': 278.8000512895101,
    'sd1': 197.14140686196984,
    'sd1_sd2': 1.0014533382525992,
    'sd2': 196.85530951043108,
    'sdnn': 196.99841012298089,
    'csi': 0.9985487708742027,
    'cvi': 5.190984975735425
})
print("Стандартизованный вектор:", sample_std)
