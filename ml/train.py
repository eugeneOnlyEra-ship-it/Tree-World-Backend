"""
Fine-tune EfficientNet-B0 on a soil image dataset
---------------------------------------------------
Expected dataset structure:
  ml/dataset/
    train/
      clay/       *.jpg
      laterite/   *.jpg
      loam/       *.jpg
      peat/       *.jpg
      sandy/      *.jpg
      silt/       *.jpg
    val/
      clay/  ...  (same structure)

Run:
  python ml/train.py

The best checkpoint is saved to ml/checkpoints/efficientnet_b0_soil.pth
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from app.services.classifier import build_model, SOIL_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("ml/dataset")
CHECKPOINT_DIR = Path("ml/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS        = 30
BATCH_SIZE    = 32
LR            = 1e-4          # lower LR — we fine-tune, not train from scratch
LR_WARMUP     = 5             # epochs before unfreezing backbone
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
PATIENCE      = 7             # early stopping patience

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transforms)
    val_dataset   = datasets.ImageFolder(DATASET_DIR / "val",   transform=val_transforms)

    # Verify class order matches SOIL_CLASSES
    assert list(train_dataset.class_to_idx.keys()) == SOIL_CLASSES, (
        f"Class mismatch!\nDataset: {list(train_dataset.class_to_idx.keys())}\n"
        f"Expected: {SOIL_CLASSES}\n"
        "Rename dataset folders to match exactly."
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    # Start from ImageNet pretrained weights
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(SOIL_CLASSES)),
    )
    model = base.to(device)

    # Phase 1: freeze backbone, train head only
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Phase 1: training classifier head only (backbone frozen)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # Phase 2: unfreeze backbone after warmup
        if epoch == LR_WARMUP + 1:
            logger.info("Phase 2: unfreezing backbone for full fine-tuning")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - LR_WARMUP)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        train_loss /= len(train_dataset)
        train_acc   = train_correct / len(train_dataset)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc   = val_correct / len(val_dataset)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
        )

        # ── Checkpoint best model ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": SOIL_CLASSES,
                },
                CHECKPOINT_DIR / "efficientnet_b0_soil.pth",
            )
            logger.info(f"  ✓ New best model saved (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
    logger.info(f"Checkpoint saved to: {CHECKPOINT_DIR / 'efficientnet_b0_soil.pth'}")


if __name__ == "__main__":
    train()
