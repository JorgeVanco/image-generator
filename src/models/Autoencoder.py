from json import decoder
import torch.nn as nn
from torch._tensor import Tensor
import torch
from torch.nn.functional import mse_loss
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),  # 16x16x3 -> 768
            nn.Dropout(0.3),
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


def train_step(model, X) -> Tensor:
    pred = model(X)
    loss = mse_loss(pred, X)
    return loss


def get_model(pretrained=False) -> nn.Sequential:
    # TODO Add pretrained model loading
    encoder, decoder = Encoder(), Decoder()
    model = nn.Sequential(encoder, decoder)
    return model


@torch.no_grad()
def plot_results(model, dataloader, logging_dir, device="cpu") -> None:
    # TODO Add better visualization
    autoencoder = model
    autoencoder.eval()

    figure = plt.figure()
    dataset_images, _ = next(iter(dataloader))
    dataset_images = dataset_images[:8]

    images = torch.stack([image.permute(2, 0, 1).int() for image in dataset_images])
    dataset_images = dataset_images.to(device)
    output = autoencoder(dataset_images).cpu()
    output = torch.stack([image.permute(2, 0, 1).int() for image in output])
    images = torch.cat([images, output], dim=0)

    grid = make_grid(images, nrow=8, padding=2)

    figure = plt.figure()
    plt.title("Reconstructed Images")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    figure.savefig(os.path.join(logging_dir, "images.png"))

    dataset_images = torch.randint(0, 255, (16, 30), dtype=torch.float32, device=device)
    output = autoencoder[1](dataset_images).cpu()
    output = torch.stack([image.permute(2, 0, 1).int() for image in output])

    grid = make_grid(output, nrow=8, padding=2)
    figure = plt.figure()
    plt.title("Randomly Sampled Images")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    figure.savefig(os.path.join(logging_dir, "random_images.png"))
