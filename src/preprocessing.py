from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

@dataclass
class PreprocessConfig:
    image_size: int = 32
    batch_size: int = 64
    val_size: float = 0.2
    seed: int = 42

def get_train_transforms(image_size: int = 32) -> transforms.Compose:
    """Return training transforms with light augmentation."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

def get_eval_transforms(image_size: int = 32) -> transforms.Compose:
    """Return evaluation transforms without augmentation."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image, image_path.name
        label = int(self.labels[index])
        return image, label

def split_train_validation(df, val_size: float = 0.2, seed: int = 42):
    """Create a stratified train/validation split."""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(df, test_size=val_size, andom_state=seed,stratify=df["Category"],)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
