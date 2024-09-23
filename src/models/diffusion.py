import torch.nn as nn
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from torch._tensor import Tensor
import torch
from torch.nn import functional as F
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, 1, 1
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.GELU(),  # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, 3, 1, 1
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.GELU(),  # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(
                    x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0
                ).to(x.device)
                out = shortcut(x) + x2
            # print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        """
        self.input_dim = input_dim

        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)


class DiffusionModel(nn.Module):
    def __init__(
        self,in_channels=3,
        out_channels=16,
        num_layers=4,
        nc_feat=10,
    ):  # cfeat - context features
        super(DiffusionModel, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  # assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)  # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 #[10, 256, 4,  4]

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4
            ),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(
                2 * n_feat, n_feat, 3, 1, 1
            ),  # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(
                n_feat, self.in_channels, 3, 1, 1
            ),  # map to same number of channels as input
        )
        self.a_t, self.b_t, self.ab_t = self.get_ddpm_noise_schedule(500)

    @staticmethod
    def get_ddpm_noise_schedule(
        timesteps, beta1=1e-4, beta2=0.02, device="cpu"
    ) -> Tensor:
        # construct DDPM noise schedule
        b_t = (beta2 - beta1) * torch.linspace(
            0, 1, timesteps + 1, device=device
        ) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        ab_t[0] = 1
        return a_t, b_t, ab_t

    @torch.no_grad()
    def sample_images(
        self, dataloader, n_images=8, device="cpu", last_images=False
    ) -> list[tuple[plt.Figure, str]]:
        self.eval()

        samples, intermediate_ddpm = sample_ddpm(self, 2 * n_images, device=device)
        # sx_gen_store = np.moveaxis(
        #     samples, 1, 3
        # )  # change to Numpy image format (h,w,channels) vs (channels,h,w)
        # samples = norm_all(
        #     sx_gen_store, sx_gen_store.shape[0], sx_gen_store.shape[1]
        # )  # unity norm to put in range [0,1] for np.imshow
        grid = plot_grid(samples, 2 * n_images, 4, "logs/diffusion", "run_image")
        if last_images:

            animation_ddpm = plot_sample(
                intermediate_ddpm,
                2 * n_images,
                4,
                "logs/diffusion",
                "ani_run",
                "",
                save=True,
            )
            # x_T ~ N(0, 1), sample initial noise

        # output = samples.cpu()
        # # print([(torch.max(image), torch.min(image)) for image in output])
        # output = torch.stack([image for image in output])
        # grid = make_grid(output, nrow=8, padding=2)

        figure2 = plt.figure()
        plt.title("Randomly Sampled Images")
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

        return [(figure2, "Randomly Sampled Images")]

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1)  # [10, 256, 4, 4]

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(
            -1, self.n_feat * 2, 1, 1
        )  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# class UnetUp(nn.Module):
#     def __init__(self, in_channels, out_channels, last_layer=False) -> None:
#         super(UnetUp, self).__init__()

#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(2 * in_channels, out_channels, kernel_size=1),
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True) if not last_layer else nn.Sigmoid(),
#         )

#     def forward(self, x, resx):
#         x1 = torch.cat([x, resx], dim=1)
#         x1 = self.up(x1)
#         x2 = self.conv(x1)
#         return x2 + x1


# class UnetDown(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super(UnetDown, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.down = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.down(x)
#         return x

# class EmbedFC(nn.Module):
#     def __init__(self, input_dim, emb_dim):
#         super(EmbedFC, self).__init__()
#         self.input_dim = input_dim
#         self.emb_dim = emb_dim
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, emb_dim),
#             nn.GELU(),
#             nn.Linear(emb_dim, emb_dim),
#         )

#     def forward(self, x):
#         x = x.view(-1, self.input_dim)
#         return self.model(x)


# class DiffusionModel(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=16,
#         num_layers=4,
#         nc_feat=10,
#     ) -> None:
#         super(DiffusionModel, self).__init__()
#         self.nc_feat = nc_feat
#         self.downs = nn.ModuleList(
#             [
#                 UnetDown(
#                     in_channels if i == 0 else out_channels * i, out_channels * (i + 1)
#                 )
#                 for i in range(num_layers)
#             ]
#         )
#         self.ups = nn.ModuleList(
#             [
#                 UnetUp(
#                     out_channels * (num_layers - i),
#                     (
#                         out_channels * (num_layers - 1 - i)
#                         if i != num_layers - 1
#                         else in_channels
#                     ),
#                     last_layer=(i == num_layers - 1),
#                 )
#                 for i in range(num_layers)
#             ]
#         )
#         self.time_embeddings = nn.ModuleList(
#             [
#                 EmbedFC(1, out_channels * (num_layers - i))
#                 for i in range(num_layers // 2)
#             ]
#         )
#         self.context_embeddings = nn.ModuleList(
#             [
#                 EmbedFC(self.nc_feat, out_channels * (num_layers - i))
#                 for i in range(num_layers // 2)
#             ]
#         )
#         self.a_t, self.b_t, self.ab_t = self.get_ddpm_noise_schedule(500)

#     @staticmethod
#     def get_ddpm_noise_schedule(
#         timesteps, beta1=1e-4, beta2=0.02, device="cpu"
#     ) -> Tensor:
#         # construct DDPM noise schedule
#         b_t = (beta2 - beta1) * torch.linspace(
#             0, 1, timesteps + 1, device=device
#         ) + beta1
#         a_t = 1 - b_t
#         ab_t = torch.cumsum(a_t.log(), dim=0).exp()
#         ab_t[0] = 1
#         return a_t, b_t, ab_t

#     def forward(self, x, t, c=None):

#         if c is None:
#             c = torch.zeros(x.shape[0], self.nc_feat).to(x)

#         resx = []
#         for down in self.downs:
#             x = down(x)
#             resx.append(x)
#         resx = resx[::-1]
#         for index, (up, res) in enumerate(zip(self.ups, resx)):

#             # embed the timestep and context labels
#             if index < len(self.time_embeddings):
#                 temb = self.time_embeddings[index]
#                 cemb = self.context_embeddings[index]
#                 x = x * temb(t).view(-1, temb.emb_dim, 1, 1) + cemb(c).view(
#                     -1, cemb.emb_dim, 1, 1
#                 )

#             x = up(x, res)

#         return x

#     @torch.no_grad()
#     def sample_images(
#         self, dataloader, n_images=8, device="cpu", last_images=False
#     ) -> list[tuple[plt.Figure, str]]:
#         self.eval()

#         samples, intermediate_ddpm = sample_ddpm(self, 2 * n_images, device=device)
#         # sx_gen_store = np.moveaxis(
#         #     samples, 1, 3
#         # )  # change to Numpy image format (h,w,channels) vs (channels,h,w)
#         # samples = norm_all(
#         #     sx_gen_store, sx_gen_store.shape[0], sx_gen_store.shape[1]
#         # )  # unity norm to put in range [0,1] for np.imshow
#         grid = plot_grid(samples, 2 * n_images, 4, "logs/diffusion", "run_image")
#         if last_images:

#             animation_ddpm = plot_sample(
#                 intermediate_ddpm,
#                 2 * n_images,
#                 4,
#                 "logs/diffusion",
#                 "ani_run",
#                 "",
#                 save=True,
#             )
#             # x_T ~ N(0, 1), sample initial noise

#         # output = samples.cpu()
#         # # print([(torch.max(image), torch.min(image)) for image in output])
#         # output = torch.stack([image for image in output])
#         # grid = make_grid(output, nrow=8, padding=2)

#         figure2 = plt.figure()
#         plt.title("Randomly Sampled Images")
#         plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

#         return [(figure2, "Randomly Sampled Images")]


def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)


def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)


def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],  # human, non-human, food, spell, side-facing
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]  # human, non-human, food, spell, side-facing
    )
    return len(vec), vec


def plot_grid(x, n_sample, n_rows, save_dir, w) -> Tensor:
    # x:(n_sample, 3, h, w)
    ncols = n_sample // n_rows
    grid = make_grid(
        norm_torch(x), nrow=ncols
    )  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print("saved image at " + save_dir + f"run_image_w{w}.png")
    return grid


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False):
    ncols = n_sample // nrows
    sx_gen_store = np.moveaxis(
        x_gen_store, 2, 4
    )  # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(
        sx_gen_store, sx_gen_store.shape[0], n_sample
    )  # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols, nrows)
    )

    def animate_diff(i, store):
        print(f"gif animating frame {i} of {store.shape[0]}", end="\r")
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row * ncols) + col]))
        return plots

    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[nsx_gen_store],
        interval=200,
        blit=False,
        repeat=True,
        frames=nsx_gen_store.shape[0],
    )
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print("saved gif at " + save_dir + f"{fn}_w{w}.gif")
    return ani


# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
n_feat = 64  # 64 hidden dimension feature
n_cfeat = 5  # context vector is of size 5
height = 16  # 16x16 image


# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise, ab_t) -> Tensor:
    return (
        ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    )


def train_step(
    model: DiffusionModel, X, optimizer, device="cpu", timesteps=500
) -> Tensor:
    optimizer.zero_grad()
    # perturb data
    noise = torch.randn_like(X)
    t = torch.randint(1, timesteps + 1, (X.shape[0],)).to(device)
    x_pert = perturb_input(X, t, noise, model.ab_t.to(device))

    pred_noise = model(x_pert, t / timesteps)
    # loss is mean squared error between the predicted and true noise
    loss = F.mse_loss(pred_noise, noise)
    loss.backward()

    optimizer.step()

    return loss.item()  # [gen_loss.item(), disc_loss.item()]


def get_model(pretrained=None) -> DiffusionModel:
    model = DiffusionModel()
    if pretrained:
        model.load_state_dict(pretrained)
    return model


# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, a_t, b_t, ab_t, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


@torch.no_grad()
def sample_ddpm(model, n_sample, save_rate=20, device="cpu"):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        # print(f"sampling timestep {i:3d}", end="\r")

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t)  # predict noise e_(x_t,t)
        a_t = model.a_t.to(device)
        b_t = model.b_t.to(device)
        ab_t = model.ab_t.to(device)

        samples = denoise_add_noise(samples, i, eps, a_t, b_t, ab_t, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
