#!/usr/bin/env python3
from pathlib import Path
import sys
import csv

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loading import load_csv_files, build_image_path
from src.preprocessing import split_train_validation, get_train_transforms, get_eval_transforms, ImageClassificationDataset
from src.simple_baseline import SimpleBaseline
from src.training import fit, get_device
from torch.utils.data import DataLoader
import torch


def build_loaders(train_df, val_df, batch_size=64):
    train_transform = get_train_transforms(32)
    val_transform = get_eval_transforms(32)

    train_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in train_df.itertuples(index=False)]
    val_paths = [build_image_path(row.Id, split="train", label=row.Category) for row in val_df.itertuples(index=False)]

    train_dataset = ImageClassificationDataset(train_paths, train_df["Category"].tolist(), transform=train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_df["Category"].tolist(), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main():
    train_df, _ = load_csv_files()
    train_split, val_split = split_train_validation(train_df, val_size=0.2, seed=42)

    lrs = [1e-4, 1e-3, 3e-3]
    drops = [0.2, 0.3, 0.5]
    results = []

    device = get_device()

    train_loader, val_loader = build_loaders(train_split, val_split, batch_size=64)

    OUT = ROOT / 'outputs'
    OUT.mkdir(exist_ok=True)
    csv_path = OUT / 'hparam_sweep_results.csv'

    for lr in lrs:
        for drop in drops:
            print(f'Running lr={lr}, dropout={drop}')
            model = SimpleBaseline(num_classes=10, dropout=drop).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            history = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=3, save_path=None)
            final_val_acc = history['val_accuracy'][-1]
            results.append({'lr': lr, 'dropout': drop, 'val_accuracy': final_val_acc})
            print(f'  -> val_acc={final_val_acc:.4f}')

    # save results
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lr', 'dropout', 'val_accuracy'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # print best
    best = max(results, key=lambda r: r['val_accuracy'])
    print('\nBest:', best)
    print('Results saved to', csv_path)


if __name__ == '__main__':
    main()
