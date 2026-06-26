"""
Training script for RetinaTriage — Diabetic Retinopathy classification.

ResNet50 fine-tuned on the APTOS 2019 Blindness Detection dataset.
5 classes: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR

Usage:
    python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out weights/model.pth
    python train.py --data_dir data/ --smoke_test --out weights/model.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetinopathyDataset(Dataset):
    """Loads retinal fundus images by id_code from a directory of PNGs."""

    def __init__(self, dataframe: pd.DataFrame, images_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, f"{row['id_code']}.png")
        image = Image.open(img_path).convert("RGB")
        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Val transform MUST match backend/model.py preprocess()
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 5) -> nn.Module:
    """ResNet50 with a replaced final FC layer for DR classification."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Class weights (inverse frequency)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """Return inverse-frequency weights for each class."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    # Avoid division by zero for classes that may not appear
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so they sum to num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return epoch_loss, epoch_acc, epoch_kappa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RetinaTriage DR classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data/ directory containing train.csv and train_images/")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--out", type=str, default="weights/model.pth",
                        help="Output path for saved model weights")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick sanity check: 100 images, 1 epoch")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Smoke test override
    # ------------------------------------------------------------------
    if args.smoke_test:
        print("SMOKE TEST MODE: Using 100 images, 1 epoch")
        args.epochs = 1

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load CSV & split
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.data_dir, "train.csv")
    images_dir = os.path.join(args.data_dir, "train_images")

    df = pd.read_csv(csv_path)
    if args.smoke_test:
        df = df.head(100)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["diagnosis"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = RetinopathyDataset(train_df, images_dir, transform=train_transform)
    val_ds = RetinopathyDataset(val_df, images_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = build_model(num_classes=5).to(device)

    class_weights = compute_class_weights(train_df["diagnosis"].values, num_classes=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_kappa = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc, val_kappa = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  "
              f"val_acc: {val_acc:.4f}  |  "
              f"val_kappa: {val_kappa:.4f}")

        # Save best model (state_dict only)
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_epoch = epoch
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"  ✓ Best model saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"  Best val kappa : {best_kappa:.4f}  (epoch {best_epoch})")
    print(f"  Model saved to : {args.out}")
    print("=" * 50)


if __name__ == "__main__":
    main()
