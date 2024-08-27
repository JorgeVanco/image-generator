import os
import json
import logging
from torch.utils.tensorboard import SummaryWriter


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


def get_loggers(
    logging_dir, verbose: bool = False, use_writer: bool = False
) -> logging.Logger:
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
    if use_writer:
        writer = SummaryWriter(logging_dir)
    return logger, writer
