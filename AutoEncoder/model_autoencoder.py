import torch.nn as nn


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