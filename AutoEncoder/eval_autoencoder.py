"""
Прогнозирование при помощи HRVAutoEncoder
"""
import pandas as pd
import numpy as np
import torch

from model_autoencoder import HRVAutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HRVAutoEncoder()
model.encoder.load_state_dict(torch.load('hrv_encoder.pt'))
model.eval()

def hrv_to_z(hrv_vec: np.ndarray) -> np.ndarray:
    """ 1. Принимает нормализованный HRV-вектор формы (10,),
        2. Возвращает 4-мерный z-эмбеддинг.
    """
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.as_tensor(hrv_vec, dtype=torch.float32).to(device)).cpu().numpy()
    # ravel() -- многомерный -> одномерный, аналог reshape(-1)
    return z.ravel()


if __name__ == '__main__':
    X_norm = pd.read_csv('dataset_to_autoencoder_scaled.csv').to_numpy()
    # res = hrv_to_z(X_norm[:1])
    res = [hrv_to_z(x) for x in X_norm]
    df =pd.DataFrame({'embedding': res})
    # Сохранение вычисленных эмбедингов
    df.to_csv('embedding_from_autoencoder.csv', index=False)

    print(res)