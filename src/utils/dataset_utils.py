import numpy as np
from torch import Tensor, from_numpy
import kaggle


def read_npy(path) -> Tensor:
    return from_numpy(np.load(path)).float()


def download_dataset(path: str, unzip: bool = True, quiet: bool = False) -> None:
    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(
        "ebrahimelgazar/pixel-art", path=path, unzip=unzip, quiet=quiet
    )
