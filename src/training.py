from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def get_device() -> torch.device:
    """Return the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy for a batch."""
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochResult:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    batch_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += accuracy_from_logits(logits.detach(), labels)
        batch_count += 1

    return EpochResult(
        loss=running_loss / max(batch_count, 1),
        accuracy=running_accuracy / max(batch_count, 1),
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Evaluate the model and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    batch_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += accuracy_from_logits(logits, labels)
            batch_count += 1

    return EpochResult(
        loss=running_loss / max(batch_count, 1),
        accuracy=running_accuracy / max(batch_count, 1),
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> Dict[str, list]:
    """Train the model for multiple epochs and collect metrics."""
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_result.loss)
        history["train_accuracy"].append(train_result.accuracy)
        history["val_loss"].append(val_result.loss)
        history["val_accuracy"].append(val_result.accuracy)

        print(
            f"Epoch {epoch:02d}: "
            f"train loss {train_result.loss:.4f} acc {train_result.accuracy:.4f}: "
            f"val loss {val_result.loss:.4f} acc {val_result.accuracy:.4f}"
        )

    return history
