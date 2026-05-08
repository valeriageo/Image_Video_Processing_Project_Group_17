#!/usr/bin/env python3
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from PIL import Image

from src.data_loading import load_csv_files, build_image_path
from src.preprocessing import get_eval_transforms


def save_preprocessing_figure(out_path: Path) -> None:
    train_df, _ = load_csv_files()
    sample = train_df.iloc[0]
    image_path = build_image_path(sample.Id, split="train", label=sample.Category)

    raw_image = Image.open(image_path).convert("L")
    transform = get_eval_transforms(32)
    transformed = transform(Image.open(image_path).convert("RGB"))
    transformed_image = transformed.squeeze(0).numpy()
    transformed_image = transformed_image * 0.5 + 0.5

    class_counts = train_df["Category"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Raw image")
    axes[0].axis("off")

    axes[1].imshow(transformed_image, cmap="gray")
    axes[1].set_title("Preprocessed image")
    axes[1].axis("off")

    axes[2].bar(class_counts.index.astype(str), class_counts.values, color="#4c72b0")
    axes[2].set_title("Class balance")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Count")

    fig.suptitle("Preprocessing overview", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def main():
    outputs = ROOT / "outputs"
    report_dir = outputs / "report_figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_path = report_dir / "preprocessing_overview.png"
    training_curves_path = report_dir / "training_curves.png"
    confusion_matrix_path = report_dir / "confusion_matrix.png"

    save_preprocessing_figure(preprocessing_path)
    print(f"✓ Saved preprocessing figure: {preprocessing_path}")

    training_ok = copy_if_exists(outputs / "training_curves.png", training_curves_path)
    confusion_ok = copy_if_exists(outputs / "confusion_improved.png", confusion_matrix_path)

    if training_ok:
        print(f"✓ Saved training curves figure: {training_curves_path}")
    else:
        print(f"✗ Missing source figure: {outputs / 'training_curves.png'}")

    if confusion_ok:
        print(f"✓ Saved confusion matrix figure: {confusion_matrix_path}")
    else:
        print(f"✗ Missing source figure: {outputs / 'confusion_improved.png'}")

    ready = preprocessing_path.exists() and training_curves_path.exists() and confusion_matrix_path.exists()
    if ready:
        print("✓ Report figure set is ready in outputs/report_figures/")
    else:
        print("✗ Report figure set is incomplete")
    return ready


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)