import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# 1. Загружаем и готовим данные
# ----------------------------
# X_norm.shape == (4000, 10) – уже нормализованный датасет
X_norm = pd.read_csv('dataset_to_autoencoder_scaled.csv').to_numpy()
X_train = torch.as_tensor(X_norm, dtype=torch.float32)

batch_size = 128
loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

# ----------------------------
# 2. Определяем автоэнкодер
# ----------------------------
class HRVAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 11 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Linear(11, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4)          # bottleneck → z
        )

        # Decoder: 4 -> 8 -> 11
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 11)
        )

    def forward(self, x):
        z = self.encoder(x)          # сжатие
        x_hat = self.decoder(z)      # восстановление
        return x_hat, z              # возвращаем и реконструкцию, и z

# ----------------------------
# 3. Настройка обучения
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = HRVAutoEncoder().to(device)

criterion = nn.MSELoss()                       # реконструкционная ошибка
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4                          # β = 1e-4 (L2-рег.)
)

num_epochs   = 1000
best_val_loss = float('inf')

# ----------------------------
# 4. Цикл обучения
# ----------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        # x_batch.shape -> torch.Size([32, 11])

        # Прямой проход
        x_hat, _ = model(x_batch)

        # Считаем MSE-лосс; L2 добавляется автоматически через weight_decay
        loss = criterion(x_hat, x_batch)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.size(0)

    epoch_loss /= len(loader.dataset)
    print(f'Epoch {epoch:03d}: loss = {epoch_loss:.6f}')

# ----------------------------
# 5. Сохранение encoder-части
# ----------------------------
torch.save(model.encoder.state_dict(), 'hrv_encoder.pt')

# ----------------------------
# 6. Получение z-вектора для нового HRV-набора
# ----------------------------
def hrv_to_z(hrv_vec: np.ndarray) -> np.ndarray:
    """
    Принимает нормализованный HRV-вектор формы (10,),
    возвращает 4-мерный z-эмбеддинг.
    """
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.as_tensor(hrv_vec, dtype=torch.float32).to(device))
    return z.cpu().numpy()

# пример вызова
sample_z = hrv_to_z(X_norm[0])
print('z-vector:', sample_z)
