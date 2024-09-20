import torch.nn as nn
from torch._tensor import Tensor
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.nn.utils.clip_grad import clip_grad_norm_


def get_noise(n_samples, z_dim, device="cpu") -> Tensor:
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """

    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=3 * 16 * 16, hidden_dim=128) -> None:
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, 2 * hidden_dim),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(8 * hidden_dim, im_dim),
            nn.Sigmoid(),
        )

    def get_generator_block(self, input_dim, output_dim) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise) -> Tensor:
        out: Tensor = self.encoder(noise)
        return out.view(-1, 3, 16, 16)


class Discriminator(nn.Module):
    def __init__(self, im_dim=3 * 16 * 16, hidden_dim=128) -> None:
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Flatten(),
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def get_discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.disc(x)
        return x


class GAN(nn.Module):
    def __init__(self, z_dim=64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim=z_dim)
        self.discriminator = Discriminator()

    def forward(self, x: Tensor) -> Tensor:
        x = self.generator(x)
        return x

    @torch.no_grad()
    def sample_images(
        self, dataloader, n_images=8, device="cpu", last_images=False
    ) -> list[tuple[plt.Figure, str]]:
        self.eval()

        dataset_images, _ = next(iter(dataloader))
        dataset_images = dataset_images[:n_images]

        dataset_images = torch.randn(
            (16, self.z_dim), dtype=torch.float32, device=device
        )
        output = self.generator(dataset_images).cpu()
        output = torch.stack([image for image in output])

        grid = make_grid(output, nrow=8, padding=2)

        figure2 = plt.figure()
        plt.title("Randomly Sampled Images")
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

        return [(figure2, "Randomly Sampled Images")]

    def get_optimizer(self, args, optimizers):
        optimizer_class = optimizers[args.optimizer]
        generator_optimizer = optimizer_class(
            self.generator.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        discriminator_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        return {
            "generator_optimizer": generator_optimizer,
            "discriminator_optimizer": discriminator_optimizer,
        }


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    """
    z = get_noise(num_images, z_dim, device=device)
    with torch.no_grad():
        fake_images = gen(z)

    fake_preds = disc(fake_images)
    fake_labels = torch.zeros_like(fake_preds)
    fake_loss = criterion(fake_preds, fake_labels)

    real_preds = disc(real)
    real_labels = torch.ones_like(real_preds)
    real_loss = criterion(real_preds, real_labels)

    disc_loss = (fake_loss + real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    """
    z = get_noise(num_images, z_dim, device=device)
    fake_images = gen(z)
    fake_preds = disc(fake_images)
    fake_labels = torch.ones_like(fake_preds)
    gen_loss = criterion(fake_labels, fake_preds)
    return gen_loss


def train_step(model: GAN, X, optimizer, device="cpu") -> Tensor:
    n_images = X.shape[0]
    disc_loss = get_disc_loss(
        model.generator,
        model.discriminator,
        binary_cross_entropy_with_logits,
        X,
        n_images,
        model.z_dim,
        device,
    )
    disc_opt = optimizer["discriminator_optimizer"]
    disc_opt.zero_grad()
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    gen_loss = get_gen_loss(
        model.generator,
        model.discriminator,
        binary_cross_entropy_with_logits,
        n_images,
        model.z_dim,
        device,
    )
    gen_opt = optimizer["generator_optimizer"]
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()

    # clip_grad_norm_(model.parameters(), 5)

    return gen_loss.item()  # [gen_loss.item(), disc_loss.item()]


def get_model(pretrained=None) -> nn.Sequential:
    model = GAN()
    if pretrained:
        model.load_state_dict(pretrained)
    return model
