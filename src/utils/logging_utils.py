import os
import json
import logging
from torch.utils.tensorboard import SummaryWriter
import subprocess


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


class MyWriter(SummaryWriter):
    def __init__(self, logging_dir) -> None:
        super().__init__(log_dir=logging_dir)
        self.tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", logging_dir, "--port", "6006", "--bind_all"],
        )

    def close(self) -> None:
        self.tb_process.terminate()
        super().close()


def get_loggers(
    logging_dir, verbose: bool = False, use_writer: bool = False
) -> tuple[logging.Logger, MyWriter]:

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
        writer_logging = logging_dir  # .rsplit("/", 1)[0]
        print(writer_logging)
        writer = MyWriter(writer_logging)

    return logger, writer
