#!/usr/bin/env python3
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
OUTPUT_DIR.mkdir(exist_ok=True)

print("ENSEMBLE SUBMISSION — Train 3 models with different seeds")

device = get_device()
train_df, test_df = load_csv_files()

# Train 3 models with different random seed for diversity
seeds = [42, 123, 999]  # Different seed for variety
all_test_predictions = []

for model_idx, seed in enumerate(seeds, 1):
    print(f"\n[Model {model_idx}/3] Training with seed={seed}")
    
    # Load data with different seed
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
    
    # Train model
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
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Generateing predictions on test set
    test_paths = [build_image_path(image_id, split='test') for image_id in test_df['Id'].tolist()]
    test_transform_noaug = get_eval_transforms(32)
    test_transform_aug = get_train_transforms(32)
    
    test_dataset_noaug = ImageClassificationDataset(test_paths, labels=None, transform=test_transform_noaug)
    test_dataset_aug = ImageClassificationDataset(test_paths, labels=None, transform=test_transform_aug)
    
    test_loader_noaug = DataLoader(test_dataset_noaug, batch_size=64, shuffle=False, num_workers=0)
    test_loader_aug = DataLoader(test_dataset_aug, batch_size=64, shuffle=False, num_workers=0)
    
    # TTA predictions
    model.eval()
    all_logits = np.zeros((len(test_df), 10))
    
    for test_loader in [test_loader_noaug, test_loader_aug]:
        idx = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images).cpu().numpy()
                batch_size = logits.shape[0]
                all_logits[idx:idx+batch_size] += logits
                idx += batch_size
    
    avg_logits = all_logits / 2  # Average TTA predictions
    predictions = np.argmax(avg_logits, axis=1)
    all_test_predictions.append(predictions)
    print(f"Generated test predictions")

# Ensemble voting for all 3 models
print("ENSEMBLE VOTING")

all_test_predictions = np.array(all_test_predictions)
print(f"Ensemble shape: {all_test_predictions.shape}  (3 models × 3000 test samples)")

# Majority voting
final_predictions = np.mode(all_test_predictions, axis=0)[0].flatten()

print(f"Final ensemble predictions generated")
print(f"  - Unique classes: {sorted(np.unique(final_predictions).tolist())}")
print(f"  - Sample predictions: {final_predictions[:10]}")


# Save ensemble submission
submission_df = pd.DataFrame({
    'Id': test_df['Id'].values,
    'Category': final_predictions
})
submission_path = OUTPUT_DIR / 'submission_ensemble.csv'
submission_df.to_csv(submission_path, index=False)

print("SUBMISSION SAVED")
print(f"Ensemble submission: {submission_path}")
print(f"Shape: {submission_df.shape}")
print(f"First 5 rows:")
print(submission_df.head())

