from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn.utils.clip_grad import clip_grad_norm_


def kl_divergence_loss(q_dist) -> torch.Tensor:
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)


reconstruction_loss = nn.MSELoss(reduction="sum")


def vae_loss(reconstructed_images, images, encoding) -> float:
    return (
        reconstruction_loss(reconstructed_images, images)
        + kl_divergence_loss(encoding).sum()
    )


class Encoder(nn.Module):
    """
    Encoder Class
    Values:
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white (1 channel), so that's our default.
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=1, output_chan=128, hidden_dim=16) -> None:
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        self.disc = nn.Sequential(
            # self.make_disc_block(im_chan, hidden_dim),
            # self.make_disc_block(hidden_dim, hidden_dim * 2),
            # self.make_disc_block(
            #     hidden_dim * 2, output_chan * 2, kernel_size=2, final_layer=True
            # ),
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
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(2, 2), padding="same"
            ),  # mx64x4x4
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(kernel_size=(2, 2)),  # mx64x2x2,
            # nn.Conv2d(
            #     in_channels=128, out_channels=256, kernel_size=(2, 2), padding="same"
            # ),  # mx64x4x4
            # nn.BatchNorm2d(256),
            # nn.MaxPool2d(kernel_size=(2, 2)),  # mx64x2x2,
            nn.Flatten(),  # mx128 #mx64x2x2
            nn.Linear(128 * 2 * 2, 2 * self.z_dim),
        )

    def make_disc_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        final_layer=False,
    ) -> nn.Sequential:
        """
        Function to return a sequence of operations corresponding to a encoder block of the VAE,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function for completing a forward pass of the Encoder: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        """
        disc_pred = self.disc(image)
        encoding = disc_pred.view(len(disc_pred), -1)
        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding[:, : self.z_dim], encoding[:, self.z_dim :].exp()


class Decoder(nn.Module):
    """
    Decoder Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=128, im_chan=1, hidden_dim=16) -> None:
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            # self.make_gen_block(z_dim, hidden_dim * 4),  # 3x3
            # self.make_gen_block(
            #     hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            # ),  # 6x6
            # self.make_gen_block(
            #     hidden_dim * 2, hidden_dim, kernel_size=2, stride=2
            # ),  # 12x12
            # self.make_gen_block(
            #     hidden_dim, im_chan, kernel_size=5, stride=1, final_layer=True
            # ),  # 26x26
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

    def make_gen_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ) -> nn.Sequential:
        """
        Function to return a sequence of operations corresponding to a Decoder block of the VAE,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.Sigmoid(),
            )

    def forward(self, noise) -> torch.Tensor:
        """
        Function for completing a forward pass of the Decoder: Given a noise vector,
        returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        """
        # x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(noise)


class VAE(nn.Module):
    """
    VAE Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=128, im_chan=3, hidden_dim=64) -> None:
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(im_chan, z_dim)
        self.decoder = Decoder(z_dim, im_chan)

    def forward(self, images):
        """
        Function for completing a forward pass of the Decoder: Given a noise vector,
        returns a generated image.
        Parameters:
        images: an image tensor with dimensions (batch_size, im_chan, im_height, im_width)
        Returns:
        decoding: the autoencoded image
        q_dist: the z-distribution of the encoding
        """
        q_mean, q_stddev = self.encoder(images)
        q_dist = Normal(q_mean, q_stddev)
        z_sample = (
            q_dist.rsample()
        )  # Sample once from each distribution, using the `rsample` notation

        decoding = self.decoder(z_sample)
        return decoding, q_dist

    @torch.no_grad()
    def sample_images(
        self, dataloader, n_images=8, device="cpu", last_images=False
    ) -> list[tuple[plt.Figure, str]]:
        self.eval()

        dataset_images, _ = next(iter(dataloader))
        dataset_images = dataset_images[:n_images]

        images = torch.stack([image for image in dataset_images])
        dataset_images = dataset_images.to(device)
        output, _ = self(dataset_images)
        output = torch.stack([image for image in output.cpu()])
        images = torch.cat([images, output], dim=0)

        grid = make_grid(images, nrow=8, padding=2)
        figure1 = plt.figure()
        plt.title("Reconstructed Images")
        plt.imshow(grid.permute(1, 2, 0).cpu())

        dataset_images = torch.randn(
            (16, self.z_dim), dtype=torch.float32, device=device
        )
        output = self.decoder(dataset_images).cpu()
        output = torch.stack([image for image in output])

        grid = make_grid(output, nrow=8, padding=2)

        figure2 = plt.figure()
        plt.title("Randomly Sampled Images")
        plt.imshow(grid.permute(1, 2, 0).cpu())

        return [(figure1, "Reconstructed Images"), (figure2, "Randomly Sampled Images")]


def train_step(model, X, optimizer) -> torch.Tensor:
    pred, encoding = model(X)

    loss = vae_loss(pred, X, encoding)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss.item()


def get_model(pretrained=None) -> nn.Sequential:
    model = VAE()
    if pretrained:
        model.load_state_dict(pretrained)
    return model
