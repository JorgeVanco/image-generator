from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import os


class PixelDataset(Dataset):
    def __init__(self, pixel_csv, img_dir) -> None:
        self.pixel_csv = pd.read_csv(pixel_csv)
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.pixel_csv)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.pixel_csv.iloc[index, 1])
        image = read_image(img_path)
        label = self.pixel_csv.iloc[index, 2]
        return image, label
