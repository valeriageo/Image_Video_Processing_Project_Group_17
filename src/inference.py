from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
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

def predict_labels_with_tta(model: torch.nn.Module, dataloaders: List[DataLoader], device: torch.device) -> List[int]:
    """Predict class labels using Test-Time Augmentation (TTA).
    
    Averages predictions across multiple augmentation strategies and returns class with highest vote.
    
    Args:
        model: BaselineCNN model
        dataloaders: List of DataLoaders with different augmentation strategies
        device: torch device (cpu or cuda)
    
    Returns:
        List of predicted class labels (0-9)
    """
    model.eval()
    num_samples = len(dataloaders[0].dataset)
    all_logits = np.zeros((num_samples, 10))  # Accumulate logits across TTA
    
    with torch.no_grad():
        for dataloader in dataloaders:
            idx = 0
            for images, _ in dataloader:
                images = images.to(device)
                logits = model(images).cpu().numpy()
                batch_size = logits.shape[0]
                all_logits[idx:idx+batch_size] += logits
                idx += batch_size
    
    # Average logits and take argmax
    avg_logits = all_logits / len(dataloaders)
    predictions = np.argmax(avg_logits, axis=1).tolist()
    return predictions

def predict_probabilities_with_tta(
    model: torch.nn.Module,
    dataloaders: List[DataLoader],
    device: torch.device,
) -> np.ndarray:
    """Predict class probabilities using Test-Time Augmentation (TTA).
    Returns averaged softmax probabilities with shape (N, num_classes).
    """
    model.eval()
    num_samples = len(dataloaders[0].dataset)
    all_probs = np.zeros((num_samples, 10), dtype=np.float32)

    with torch.no_grad():
        for dataloader in dataloaders:
            idx = 0
            for images, _ in dataloader:
                images = images.to(device)
                probs = torch.softmax(model(images), dim=1).cpu().numpy()
                batch_size = probs.shape[0]
                all_probs[idx:idx + batch_size] += probs
                idx += batch_size

    return all_probs / len(dataloaders)

def build_submission(test_ids: List[int], predicted_labels: List[int]) -> pd.DataFrame:
    """Create the submission dataframe in the required format."""
    return pd.DataFrame({"Id": test_ids, "Category": predicted_labels})

def save_submission(submission_df: pd.DataFrame, output_path: Path) -> None:
    """Save the submission file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
