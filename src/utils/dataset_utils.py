import numpy as np
from torch import Tensor, from_numpy


def read_npy(path) -> Tensor:
    return from_numpy(np.load(path)).float()
