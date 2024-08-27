import importlib
import argparse
import os


from utils import (
    get_project_root,
    train_loop,
    get_loggers,
    get_dataset_dataloader,
    get_logging_dir,
)

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

# TODO Add Validation set


def main(args) -> None:

    root_path = get_project_root(__file__)
    logging_dir = get_logging_dir(root_path, args)

    logger, writer = get_loggers(
        logging_dir=logging_dir, verbose=args.verbose, writer=args.writer
    )

    dataset, dataloader = get_dataset_dataloader(
        root_path, args.data_dir, args.batch_size, args.num_workers, logger=logger
    )

    device = torch.device("cpu")
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.warning("GPU not available. Using CPU.")

    model_module = importlib.import_module(f"models.{args.model}")
    model = model_module.get_model(pretrained=args.pretrained).to(device)

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
        logger=logger,
        writer=writer,
        device=device,
    )

    # Save the model
    model_path = os.path.join(logging_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # TODO Add checkpointing

    # TODO Add visualization of losses and gradients
    fig, axes = plt.subplots(2)
    axes[0].plot(torch.log10(torch.tensor(losses)))
    axes[0].set_title("Losses")
    axes[1].plot(gradients)
    axes[1].set_title("Gradients")
    plt.tight_layout()
    try:
        fig.savefig(os.path.join(logging_dir, "losses.png"))
    except Exception as e:
        logger.error(e)

    # TODO Add visualization of model predictions
    model_module.plot_results(model, dataloader, logging_dir=logging_dir, device=device)


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
        "--log-dir", type=str, default="logs", help="Directory to save training logs"
    )
    log_params_group.add_argument(
        "--resume", type=str, help="Path to a checkpoint to resume training from"
    )
    log_params_group.add_argument(
        "--writer", action="store_true", help="Use Tensorboard writer"
    )

    # Miscellaneous
    miscellaneous_group = parser.add_argument_group("Miscellaneous")
    miscellaneous_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    # miscellaneous_group.add_argument("--gpu", type=int, help="GPU id to use (if any)")
    miscellaneous_group.add_argument(
        "--gpu", action="store_true", help="Use GPU for training"
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
