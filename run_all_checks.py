#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

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
    ImageClassificationDataset,
)
from src.baseline_cnn import BaselineCNN
from src.simple_baseline import SimpleBaseline
from src.training import fit, get_device

def build_loaders(train_df, val_df, batch_size=64):
    train_transform = get_train_transforms(32)
    val_transform = get_eval_transforms(32)

    train_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in train_df.itertuples(index=False)]
    val_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in val_df.itertuples(index=False)]

    train_dataset = ImageClassificationDataset(train_paths, train_df["Category"].tolist(), transform=train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_df["Category"].tolist(), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_dataset, val_dataset

def eval_and_report(model, dataloader, device, out_prefix: Path):
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_y_pred.extend(preds)
            all_y_true.extend(labels.numpy().tolist())

    cm = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(all_y_true, all_y_pred, digits=4)
    (out_prefix.parent).mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix('.txt'), 'w') as f:
        f.write(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {out_prefix.stem}')
    plt.savefig(out_prefix.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return report, cm

def save_sample_predictions(model, val_dataset, val_loader, device, out_path: Path, num_images=25):
    import random
    model.eval()
    idxs = random.sample(range(len(val_dataset)), min(num_images, len(val_dataset)))
    images = []
    trues = []
    preds = []
    with torch.no_grad():
        for i in idxs:
            x, y = val_dataset[i]
            images.append(x.squeeze(0).cpu().numpy())
            trues.append(int(y))
            xi = x.unsqueeze(0).to(device)
            logits = model(xi)
            preds.append(int(logits.argmax(dim=1).cpu().item()))

    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes[len(images):]:
        ax.axis('off')
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'p={preds[i]} / t={trues[i]}')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def randomized_label_test(train_df, val_df, epochs=3):
    # Shuffling train labels while keeping validation labels intact
    import random
    # Building train paths from original train_df but assign shuffled labels separately so image files remain valid
    train_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in train_df.itertuples(index=False)]
    labels = [int(row.Category) for row in train_df.itertuples(index=False)]
    random.shuffle(labels)

    train_dataset = ImageClassificationDataset(train_paths, labels, transform=get_train_transforms(32))
    val_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in val_df.itertuples(index=False)]
    val_dataset = ImageClassificationDataset(val_paths, val_df['Category'].tolist(), transform=get_eval_transforms(32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = get_device()
    model = SimpleBaseline(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs, save_path=None)
    return history

def main():
    train_df, _ = load_csv_files()
    train_split, val_split = split_train_validation(train_df, val_size=0.2, seed=42)

    # Build loaders for standard training
    train_loader, val_loader, train_dataset, val_dataset = build_loaders(train_split, val_split)

    device = get_device()

    OUT = ROOT / 'outputs'
    OUT.mkdir(exist_ok=True)

    #Training SimpleBaseline
    simple_model = SimpleBaseline(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    print('Training SimpleBaseline...')
    simple_history = fit(simple_model, train_loader, val_loader, criterion, optimizer, device, epochs=3, save_path=OUT / 'simple_baseline.pt')

    #Evaluating improved model (existing saved checkpoint)
    improved_model = BaselineCNN(num_classes=10).to(device)
    improved_model.load_state_dict(torch.load(ROOT / 'outputs' / 'baseline_cnn.pt', map_location=device))

    print('Evaluating improved model...')
    report_improved, cm_improved = eval_and_report(improved_model, val_loader, device, OUT / 'confusion_improved')
    print('Improved model classification report:\n', report_improved)

    # Evaluating SimpleBaseline
    print('Evaluating simple baseline...')
    report_simple, cm_simple = eval_and_report(simple_model, val_loader, device, OUT / 'confusion_simple')
    print('Simple baseline classification report:\n', report_simple)

    # Saving sample validation predictions (improved)
    save_sample_predictions(improved_model, val_dataset, val_loader, device, OUT / 'val_predictions_improved.png')
    print('Saved sample predictions to', OUT / 'val_predictions_improved.png')

    # Randomized label sanity test
    print('Running randomized-label sanity test (SimpleBaseline)...')
    rand_history = randomized_label_test(train_split, val_split, epochs=3)
    print('Randomized-label test done. Last val accuracy:', rand_history['val_accuracy'][-1])

if __name__ == '__main__':
    main()
