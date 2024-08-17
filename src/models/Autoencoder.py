import torch.nn as nn
from torch._tensor import Tensor
import torch


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),  # 16x16x3 -> 768
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 30),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.ReLU(),
            nn.Linear(100, 3 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (16, 16, 3)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        return x
