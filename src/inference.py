from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.baseline_cnn import BaselineCNN


def load_model(model_path: Path, device: torch.device) -> BaselineCNN:
    """Load a saved BaselineCNN checkpoint."""
    model = BaselineCNN(num_classes=10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_labels(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> List[int]:
    """Predict class labels for a dataloader."""
    predictions: List[int] = []
    model.eval()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)

    return predictions


def build_submission(test_ids: List[int], predicted_labels: List[int]) -> pd.DataFrame:
    """Create the submission dataframe in the required format."""
    return pd.DataFrame({"Id": test_ids, "Category": predicted_labels})


def save_submission(submission_df: pd.DataFrame, output_path: Path) -> None:
    """Save the submission file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
