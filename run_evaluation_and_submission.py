#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from src.data_loading import load_csv_files, build_image_path
from src.preprocessing import split_train_validation, get_eval_transforms, get_train_transforms, ImageClassificationDataset
from src.baseline_cnn import BaselineCNN
from src.inference import load_model, predict_labels, predict_labels_with_tta, build_submission, save_submission
from src.training import get_device

def main():
    device = get_device()
    OUT = ROOT / 'outputs'
    OUT.mkdir(exist_ok=True)
    MODEL_PATH = OUT / 'baseline_cnn.pt'
    SUBMISSION_PATH = OUT / 'submission.csv'

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    
    print(f"\n✓ Model checkpoint found: {MODEL_PATH}")

    # Loading train/val data
    train_df, test_df = load_csv_files()
    _, val_split = split_train_validation(train_df, val_size=0.2, seed=42)
    print(f"Loaded train.csv and test.csv")
    print(f"Validation set size: {len(val_split)}")
    print(f"Test set size: {len(test_df)}")

    # evaluating in valuatuion set
    print("evaluating the model on validation set ")

    val_transform = get_eval_transforms(32)
    val_paths = [build_image_path(row.Id, split='train', label=row.Category) for row in val_split.itertuples(index=False)]
    val_dataset = ImageClassificationDataset(val_paths, val_split['Category'].tolist(), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = load_model(MODEL_PATH, device)
    
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_y_pred.extend(preds)
            all_y_true.extend(labels.numpy().tolist())

    val_accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("nPer-class metrics:")
    print(classification_report(all_y_true, all_y_pred))

    # running predictions on test set with Test-Time Augmentation (TTA)
    print("\n2. RUNNING PREDICTIONS ON TEST SET (with TTA)")

    test_paths = [build_image_path(image_id, split='test') for image_id in test_df['Id'].tolist()]
    
    #Creating dataloaders with different augmentation strategies
    transform_noaug = get_eval_transforms(32)  # no augmentation
    transform_aug = get_train_transforms(32)    # full augmentation
    
    test_dataset_noaug = ImageClassificationDataset(test_paths, labels=None, transform=transform_noaug)
    test_dataset_aug = ImageClassificationDataset(test_paths, labels=None, transform=transform_aug)
    
    test_loader_noaug = DataLoader(test_dataset_noaug, batch_size=64, shuffle=False, num_workers=0)
    test_loader_aug = DataLoader(test_dataset_aug, batch_size=64, shuffle=False, num_workers=0)
    
    print("  Using Test-Time Augmentation (TTA) with 2 strategies...")
    predicted_labels = predict_labels_with_tta(model, [test_loader_noaug, test_loader_aug], device)
    
    print(f"Generated {len(predicted_labels)} test predictions (with TTA)")
    print(f"Unique classes predicted: {sorted(set(predicted_labels))}")
    print(f"Sample predictions (first 10): {predicted_labels[:10]}")

    # generating submission csv
    print("generating submission csv")

    submission_df = build_submission(test_df['Id'].tolist(), predicted_labels)
    save_submission(submission_df, SUBMISSION_PATH)
    print(f"✓ Submission saved to: {SUBMISSION_PATH}")
    print(f"  - Shape: {submission_df.shape}")
    print(f"  - Columns: {list(submission_df.columns)}")
    print(f"\nFirst 5 rows of submission:")
    print(submission_df.head())
    print(f"\nLast 5 rows of submission:")
    print(submission_df.tail())

    # verifying end to end pipline
    print("verifying end to end pipeline ")

    checks = [
        (MODEL_PATH.exists(), f"Model checkpoint exists: {MODEL_PATH}"),
        (SUBMISSION_PATH.exists(), f"Submission CSV exists: {SUBMISSION_PATH}"),
        (len(predicted_labels) == len(test_df), f"Predictions match test set size ({len(predicted_labels)} == {len(test_df)})"),
        (all(0 <= p < 10 for p in predicted_labels), "All predictions are valid class indices [0-9]"),
        (len(submission_df) == len(test_df), f"Submission rows match test set ({len(submission_df)} == {len(test_df)})"),
        (val_accuracy > 0.95, f"Validation accuracy is strong (>0.95): {val_accuracy:.4f}"),
    ]
    all_passed = True
    for passed, check_msg in checks:
        status = "passed" if passed else "failed"
        print(f"{status} {check_msg}")
        if not passed:
            all_passed = False

    if all_passed:
        print("complete")
    else:
        print("failed")
    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
