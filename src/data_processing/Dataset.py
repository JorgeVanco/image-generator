from torch._tensor import Tensor
from torch.utils.data import Dataset
from utils import read_npy
import torchvision.transforms as transforms


# TODO Change image to -1 1
class PixelDataset(Dataset):
    def __init__(self, sprites_path, labels_path) -> None:
        self.sprites = read_npy(sprites_path)
        self.labels = read_npy(labels_path)
        # self.transform = transforms.Lambda(
        #     lambda x: x / 255.0
        # )  # Normalize to range [0, 1]

    def __len__(self) -> int:
        return len(self.sprites)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image = self.sprites[index].permute(2, 0, 1) / 255.0
        label = self.labels[index]
        return image, label
