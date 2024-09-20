from json import decoder
import torch.nn as nn
from torch._tensor import Tensor
import torch
from torch.nn.functional import mse_loss
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.nn.utils.clip_grad import clip_grad_norm_

"""you can use this formula [(W-K+2P)/S]+1.

W is the input volume - in your case 128
K is the Kernel size - in your case 5
P is the padding - in your case 0 i believe
S is the stride - which you have not provided."""

"""
After Pooling: W - F + 1
"""


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super(ConvEncoder, self).__init__()
        self.z_dim = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=(4, 4), padding="same"
            ),  # mx16x16x3 -> mx16x16x16
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),  # mx16x8x8
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),  # mx32x8x8
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),  # mx32x4x4
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(2, 2), padding="same"
            ),  # mx64x4x4
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),  # mx64x2x2,
            # nn.Conv2d(
            #     in_channels=64, out_channels=128, kernel_size=(2, 2), padding="same"
            # ),  # mx128x2x2
            # nn.MaxPool2d(kernel_size=(2, 2)),  # mx128x1x1,
            nn.Flatten(),  # mx128 #mx64x2x2
            nn.Linear(64 * 2 * 2, self.z_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self) -> None:
        super(ConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)),  # mx128x1x1
            nn.Upsample(scale_factor=2),  # mx128x2x2
            nn.Conv2d(128, 64, kernel_size=(2, 2), padding="same"),  # mx128x2x2
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(2, 2), padding="same"),  # mx32x4x4
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(2, 2), padding="same"),  # mx16x8x8
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(2, 2), padding="same"),  # mx16x16x16
            nn.BatchNorm2d(16),
            # nn.Conv2d(16, 3, kernel_size=(2, 2), padding="same"),  # mx3x16x16
            nn.Flatten(1),
            nn.Linear(16 * 16 * 16, 3 * 16 * 16),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 16, 16)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x: Tensor) -> Tensor:
        o = self.autoencoder(x)
        return o

    @torch.no_grad()
    def sample_images(
        self, dataloader, n_images=8, device="cpu", last_images=False
    ) -> list[tuple[plt.Figure, str]]:
        self.eval()

        dataset_images, _ = next(iter(dataloader))
        dataset_images = dataset_images[:n_images]

        images = torch.stack([image for image in dataset_images])
        dataset_images = dataset_images.to(device)
        output = self(dataset_images).cpu()
        output = torch.stack([image for image in output])
        images = torch.cat([images, output], dim=0)

        grid = make_grid(images, nrow=8, padding=2)
        figure1 = plt.figure()
        plt.title("Reconstructed Images")
        plt.imshow(grid.permute(1, 2, 0).cpu())

        dataset_images = torch.randn(
            (16, self.encoder.z_dim), dtype=torch.float32, device=device
        )
        output = self.decoder(dataset_images).cpu()
        output = torch.stack([image for image in output])

        grid = make_grid(output, nrow=8, padding=2)

        figure2 = plt.figure()
        plt.title("Randomly Sampled Images")
        plt.imshow(grid.permute(1, 2, 0).cpu())

        return [(figure1, "Reconstructed Images"), (figure2, "Randomly Sampled Images")]


def train_step(model, X, optimizer) -> Tensor:
    pred = model(X)

    loss = mse_loss(pred, X)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss.item()


def get_model(pretrained=None) -> nn.Sequential:
    model = ConvAutoEncoder()
    if pretrained:
        model.load_state_dict(pretrained)
    return model
