from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            images, labels = images.to(device), labels.to(device)
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
    epochs: int = 30,
    patience: int = 7,
    save_path: Optional[Path] = None,
) -> Dict[str, list]:

    # Cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    history: Dict[str, list] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_result.loss)
        history["train_accuracy"].append(train_result.accuracy)
        history["val_loss"].append(val_result.loss)
        history["val_accuracy"].append(val_result.accuracy)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"lr={current_lr:.2e}  "
            f"train loss={train_result.loss:.4f} acc={train_result.accuracy:.4f}  "
            f"val loss={val_result.loss:.4f} acc={val_result.accuracy:.4f}"
        )

        # Save best checkpoint
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            patience_counter = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  New best - saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    return history
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Convenience wrapper for validation evaluation.
    Returns:
        accuracy, loss
    """

    criterion = nn.CrossEntropyLoss()

    result = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    return result.accuracy, result.loss