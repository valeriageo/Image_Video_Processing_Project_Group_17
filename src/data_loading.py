from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
TRAIN_DIR = ROOT / "train" / "train"
TEST_DIR = ROOT / "test" / "test"

def load_csv_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test CSV files."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df

def build_image_path(
    image_id: Union[int, str], split: str = "train", label: Optional[Union[int, str]] = None
) -> Path:
    """Return the image path for a given image ID.

    Train images live in label folders, while test images are stored directly in the test folder.
    """
    image_name = f"{image_id}.png"
    if split == "train":
        if label is None:
            raise ValueError("label is required when split='train'")
        return TRAIN_DIR / str(label) / image_name
    if split == "test":
        return TEST_DIR / image_name
    raise ValueError("split must be 'train' or 'test'")


def open_image(image_path: Path) -> Image.Image:
    """Open a single image from disk."""
    return Image.open(image_path)


def show_basic_dataset_info(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print the most useful first checks for the dataset."""
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train columns:", list(train_df.columns))
    print("Test columns:", list(test_df.columns))
    print("Train labels:", sorted(train_df["Category"].unique()))
    print("Train label counts:\n", train_df["Category"].value_counts().sort_index())


if __name__ == "__main__":
    train_df, test_df = load_csv_files()
    show_basic_dataset_info(train_df, test_df)
