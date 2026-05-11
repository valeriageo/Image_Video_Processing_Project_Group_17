#!/usr/bin/env python3
from pathlib import Path
import sys

# Using a non-interactive backend for matplotlib when running in terminal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Project root not found")

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loading import load_csv_files, build_image_path
from src.preprocessing import (
    split_train_validation,
    get_train_transforms,
    get_eval_transforms,
    ImageClassificationDataset,)

from src.baseline_cnn import BaselineCNN
from src.training import get_device, fit

def main():
    train_df, _ = load_csv_files()
    train_split_df, val_split_df = split_train_validation(train_df, val_size=0.2, seed=42)

    train_transform = get_train_transforms(32)
    val_transform = get_eval_transforms(32)

    train_paths = [
        build_image_path(row.Id, split="train", label=row.Category)
        for row in train_split_df.itertuples(index=False)]
    val_paths = [
        build_image_path(row.Id, split="train", label=row.Category)
        for row in val_split_df.itertuples(index=False)]

    train_dataset = ImageClassificationDataset(
        train_paths, train_split_df["Category"].tolist(), transform=train_transform)
    val_dataset = ImageClassificationDataset(
        val_paths, val_split_df["Category"].tolist(), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = get_device()
    model = BaselineCNN(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)  # Best hyperparams from sweep

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=30,  # Train longer with early stopping
        patience=7,  # Early stopping patience
        save_path=ROOT / "outputs" / "baseline_cnn.pt",)
    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_PATH = OUTPUT_DIR / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(history["train_loss"], label="train loss", color="red")
    axes[0].plot(history["val_loss"], label="val loss", color="blue")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train accuracy", color="red")
    axes[1].plot(history["val_accuracy"], label="val acc", color="blue")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")

    print("saved training curves to:", PLOT_PATH)
    print("saved model to:", ROOT / "outputs" / "baseline_cnn.pt")
    print("final train accuracy:", history["train_accuracy"][-1])
    print("final val accuracy:", history["val_accuracy"][-1])

if __name__ == "__main__":
    main()
