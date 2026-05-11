#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / 'requirements.txt').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError('Project root not found')

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loading import load_csv_files, build_image_path
from src.preprocessing import split_train_validation, get_train_transforms, get_eval_transforms, ImageClassificationDataset
from src.baseline_cnn import BaselineCNN
from src.training import get_device, fit
from src.inference import predict_probabilities_with_tta

OUTPUT_DIR = ROOT / 'outputs'
device = get_device()
train_df, test_df = load_csv_files()

seeds = [42, 789, 2025]
print(f"{len(seeds)} Models with Different Seed")
all_test_probs = []
model_weights = []

test_paths = [build_image_path(image_id, split='test') for image_id in test_df['Id'].tolist()]

tta_transforms = [
    get_eval_transforms(32),
    transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomRotation((6, 6)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomRotation((-6, -6)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomAffine(0, translate=(0.04, 0.04)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
]

print(f"Using {len(tta_transforms)} TTA transforms per model")

for model_idx, seed in enumerate(seeds, 1):
    print(f"\n[Model {model_idx}/{len(seeds)}] Training with seed={seed}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_split, val_split = split_train_validation(train_df, val_size=0.2, seed=seed)
    
    train_transform = get_train_transforms(32)
    val_transform = get_eval_transforms(32)
    
    train_paths = [build_image_path(row.Id, split='train', label=row.Category) 
                   for row in train_split.itertuples(index=False)]
    val_paths = [build_image_path(row.Id, split='train', label=row.Category) 
                 for row in val_split.itertuples(index=False)]
    
    train_dataset = ImageClassificationDataset(train_paths, train_split['Category'].tolist(), transform=train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_split['Category'].tolist(), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = BaselineCNN(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=30,
        patience=7,
        save_path=OUTPUT_DIR / f'ensemble_model_{model_idx}.pt',
    )
    best_val_acc = max(history['val_accuracy'])
    print(f"Best val acc: {best_val_acc:.4f}")
    model_weights.append(best_val_acc)
    
    test_loaders = []
    for tta_transform in tta_transforms:
        test_dataset = ImageClassificationDataset(test_paths, labels=None, transform=tta_transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loaders.append(test_loader)

    model_probs = predict_probabilities_with_tta(model, test_loaders, device)
    all_test_probs.append(model_probs)
    print("Generated TTA probabilities")

# Ensemble average softmax probabilities
print("ensemble averaging predictions")

weights = np.array(model_weights, dtype=np.float32)
weights = weights / weights.sum()
avg_probs = np.tensordot(weights, np.array(all_test_probs), axes=(0, 0))
final_predictions = np.argmax(avg_probs, axis=1)

print(f"Averaged probabilities from {len(seeds)} models")
print(f"Model weights: {[round(float(w), 4) for w in weights]}")
print(f"Sample predictions: {final_predictions[:10]}")

# Save
submission_df = pd.DataFrame({
    'Id': test_df['Id'].values,
    'Category':final_predictions
})
submission_path = OUTPUT_DIR / 'submission_ensemble.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Saved to: {submission_path}")
print(f"Shape: {submission_df.shape}")
print(f"\nFirst 5:")
print(submission_df.head())
print("complete")

