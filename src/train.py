import importlib
import argparse
import os
from pyexpat import model

from utils import get_project_root, download_dataset, train_loop

from data_processing.Dataset import PixelDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ExponentialLR  # , CosineAnnealingLR
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
schedulers = {"exponential": ExponentialLR}  # , "cosine": CosineAnnealingLR}
# TODO Add more optimizers and schedulers as needed
# TODO Add cosine annealing scheduler parameters


def get_dataset_dataloader(
    root_path: str, data_dir: str, batch_size: int, num_workers: int
) -> DataLoader:
    dataset_path = os.path.join(root_path, data_dir)
    sprites_path = os.path.join(dataset_path, "sprites.npy")
    labels_path = os.path.join(dataset_path, "sprites_labels.npy")

    if not os.path.exists(sprites_path):
        print("Dataset not found. Downloading...")
        download_dataset(dataset_path)
    else:
        print("Dataset found.")

    dataset = PixelDataset(sprites_path=sprites_path, labels_path=labels_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataset, dataloader


def main(args) -> None:

    root_path = get_project_root(__file__)

    dataset, dataloader = get_dataset_dataloader(
        root_path, args.data_dir, args.batch_size, args.num_workers
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    model_module = importlib.import_module(f"models.{args.model}")
    model = model_module.get_model(pretrained=args.pretrained)

    optimizer = optimizers[args.optimizer]
    # TODO Change how the optimizer is initialized based on the scheduler selected
    optimizer = optimizer(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        # momentum=args.momentum,
    )
    scheduler = schedulers[args.scheduler](optimizer, gamma=args.scheduler_gamma)

    losses, gradients = train_loop(
        dataloader=dataloader,
        model=model,
        train_step=model_module.train_step,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
    )

    # TODO Add logging and checkpointing
    fig, axes = plt.subplots(2)
    axes[0].plot(torch.log10(torch.tensor(losses)))
    axes[1].plot(gradients)
    plt.show()

    # TODO Add visualization of model predictions
    images_not, _ = next(iter(dataloader))
    images = torch.stack([image.permute(2, 0, 1).int() for image in images_not[:8]])
    print(images[3].int().shape)
    print(images.shape)
    grid = make_grid(images, nrow=4, padding=2)
    print(grid.shape)
    plt.imshow(grid.permute(1, 2, 0))

    plt.show()

    model_module.plot_results(model, images_not)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train your image generation model")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output verbosity"
    )

    # Data parameters
    data_params_group = parser.add_argument_group("Data parameters")
    data_params_group.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset from root directory",
    )
    data_params_group.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    data_params_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Model parameters
    model_params_group = parser.add_argument_group("Model parameters")
    model_params_group.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=["autoencoder"],
        help="Model architecture",
    )
    model_params_group.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )

    # Training parameters
    training_params_group = parser.add_argument_group("Training parameters")
    training_params_group.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    training_params_group.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    training_params_group.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD"
    )
    training_params_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    training_params_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=optimizers.keys(),
        help="Optimizer type",
    )
    training_params_group.add_argument(
        "--scheduler",
        type=str,
        default="exponential",
        choices=schedulers.keys(),
        help="Learning rate scheduler",
    )
    training_params_group.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.95,
        help="Gamma parameter for the learning rate exponential scheduler",
    )

    # Checkpointing and logging
    log_params_group = parser.add_argument_group("Checkpointing and logging")
    log_params_group.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    log_params_group.add_argument(
        "--save-frequency",
        type=int,
        default=1,
        help="How often (in epochs) to save checkpoints",
    )
    log_params_group.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory to save training logs"
    )
    log_params_group.add_argument(
        "--resume", type=str, help="Path to a checkpoint to resume training from"
    )

    # Miscellaneous
    miscellaneous_group = parser.add_argument_group("Miscellaneous")
    miscellaneous_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    miscellaneous_group.add_argument("--gpu", type=int, help="GPU id to use (if any)")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
