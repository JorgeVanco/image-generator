import json
from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
import os
from utils import download_dataset
from data_processing.Dataset import PixelDataset
from torch.utils.data import DataLoader


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_dataset_dataloader(
    root_path: str, data_dir: str, batch_size: int, num_workers: int, logger=None
) -> DataLoader:
    dataset_path = os.path.join(root_path, data_dir)
    sprites_path = os.path.join(dataset_path, "sprites.npy")
    labels_path = os.path.join(dataset_path, "sprites_labels.npy")

    if not os.path.exists(sprites_path):
        if logger:
            logger.warning("Dataset not found. Downloading...")
        download_dataset(dataset_path)

    dataset = PixelDataset(sprites_path=sprites_path, labels_path=labels_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataset, dataloader


def train_loop(
    dataloader,
    model,
    train_step,
    optimizer,
    scheduler,
    epochs=5,
    logger=None,
    device="cpu",
) -> tuple[list, list]:
    model.train()
    model.to(device)
    size = len(dataloader.dataset)
    losses = []
    gradients = []
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (X, label) in enumerate(dataloader):
                # Compute prediction error
                X = X.to(device)
                loss = train_step(model, X)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                gradients.append(get_gradient_norm(model))
                losses.append(loss.item())
                running_loss += loss.item()
                if logger and batch_idx % 32 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    logger.info(
                        f"Epoch [{epoch:>5d}/{epochs}], Batch [{batch_idx:>5d}/{len(dataloader)}], Samples [{current:>5d}/{size}], Loss: {loss:.4f}"
                    )

            avg_loss = running_loss / len(dataloader)
            logger.info(f"Epoch [{epoch:>5d}/{epochs}], Average Loss: {avg_loss:.4f}")
            scheduler.step()

    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return losses, gradients


def get_logging_dir(root_path: str, args) -> str:
    logging_dir = os.path.join(root_path, args.log_dir, args.model)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    current_run = len(os.listdir(logging_dir))
    logging_dir = os.path.join(logging_dir, f"run_{current_run}")
    os.makedirs(logging_dir)

    if args:
        with open(os.path.join(logging_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

    return logging_dir


def get_loggers(logging_dir, verbose: bool = False, args=None) -> logging.Logger:
    # TODO Add support for Tensorboard Writer

    logging_file = os.path.join(logging_dir, "training.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the minimum log level

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logging_file)

    # Set log level for handlers
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    if verbose:
        logger.addHandler(console_handler)

    writer = None
    return logger, writer
