#!/usr/bin/env python3
"""
FAST ENSEMBLE SUBMISSION (2 models)
Faster than 3 models but still gives ~0.3-0.5% boost
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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

OUTPUT_DIR = ROOT / 'outputs'
device = get_device()
train_df, test_df = load_csv_files()

print("=" * 80)
print("FAST ENSEMBLE — 2 Models with Different Seeds")
print("=" * 80)

all_test_probs = []

for model_idx, seed in enumerate([42, 789], 1):
    print(f"\n[Model {model_idx}/2] Training with seed={seed}")
    print("-" * 80)
    
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
    
    print(f"✓ Best val acc: {max(history['val_accuracy']):.4f}")
    
    # Get test predictions with probabilities
    test_paths = [build_image_path(image_id, split='test') for image_id in test_df['Id'].tolist()]
    test_dataset = ImageClassificationDataset(test_paths, labels=None, transform=get_eval_transforms(32))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model.eval()
    test_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_probs.extend(probs)
    
    all_test_probs.append(np.array(test_probs))
    print(f"✓ Generated test probabilities")

# Ensemble: average softmax probabilities
print("\n" + "=" * 80)
print("ENSEMBLE AVERAGING")
print("=" * 80)

avg_probs = np.mean(np.array(all_test_probs), axis=0)
final_predictions = np.argmax(avg_probs, axis=1)

print(f"✓ Averaged probabilities from 2 models")
print(f"  - Sample predictions: {final_predictions[:10]}")

# Save
submission_df = pd.DataFrame({
    'Id': test_df['Id'].values,
    'Category': final_predictions
})

submission_path = OUTPUT_DIR / 'submission_ensemble.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✓ Saved to: {submission_path}")
print(f"  - Shape: {submission_df.shape}")
print(f"\nFirst 5:")
print(submission_df.head())

print("\n" + "=" * 80)
print("✓ ENSEMBLE COMPLETE")
print("  Upload to Kaggle for better score!")
print("=" * 80)
