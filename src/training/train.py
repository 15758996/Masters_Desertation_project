"""
Training Script — Multi-Label Chest X-Ray Classification
Author: Chenduluru Siva | 7151CEM
Usage:
  python src/training/train.py
  python src/training/train.py --backbone densenet121 --epochs 30 --batch_size 32
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    CHECKPOINTS_DIR, SAVED_MODELS_DIR, LOGS_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, BACKBONE, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, SEED, DISEASE_LABELS, USE_WEIGHTED_LOSS
)
from src.preprocessing.dataset import get_dataloaders
from src.models.densenet_model import build_model, load_checkpoint


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[✓] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[i] No GPU — training on CPU (slow)")
    return device


# ── Loss ──────────────────────────────────────────────────────────────────────
def build_criterion(pos_weights=None, device=None):
    if pos_weights is not None and USE_WEIGHTED_LOSS:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        print("[✓] Weighted BCEWithLogitsLoss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("[✓] Standard BCEWithLogitsLoss")
    return criterion


# ── AUROC ─────────────────────────────────────────────────────────────────────
def compute_auroc(targets, probs):
    per_class = {}
    valid = []
    for i, label in enumerate(DISEASE_LABELS):
        y_true  = targets[:, i]
        y_score = probs[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class[label] = float("nan")
        else:
            try:
                auc = roc_auc_score(y_true, y_score)
                per_class[label] = auc
                valid.append(auc)
            except Exception:
                per_class[label] = float("nan")
    macro = np.nanmean(valid) if valid else 0.0
    return {"per_class": per_class, "macro": round(float(macro), 4)}


# ── One Epoch ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    all_targets = []
    all_probs   = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, targets, _ in tqdm(loader, desc="  batch", leave=False):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits  = model(images)
            loss    = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss  += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).cpu().detach().numpy())
            all_targets.append(targets.cpu().detach().numpy())

    avg_loss    = total_loss / len(loader.dataset)
    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    auroc       = compute_auroc(all_targets, all_probs)
    return avg_loss, auroc, all_targets, all_probs


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_auroc, backbone, is_best=False):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auroc":  val_auroc,
        "backbone":   backbone,
        "disease_labels": DISEASE_LABELS
    }
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch:03d}_auroc_{val_auroc:.4f}.pt")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(SAVED_MODELS_DIR, f"best_{backbone}.pt")
        torch.save(state, best_path)
        print(f"    [★] Best model saved → {best_path}")


# ── Logger ────────────────────────────────────────────────────────────────────
class TrainingLogger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","train_loss","val_loss","train_auroc","val_auroc","lr","time_s"]
            )

    def log(self, epoch, tl, vl, ta, va, lr, elapsed):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(tl,4), round(vl,4), round(ta,4), round(va,4), lr, round(elapsed,1)]
            )


# ── Main Training Loop ────────────────────────────────────────────────────────
def train(backbone=BACKBONE, num_epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
          resume_from=None):

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()

    print("\n=== Loading Dataset ===")
    train_loader, val_loader, _, label_weights = get_dataloaders(batch_size=batch_size)

    print("\n=== Building Model ===")
    model     = build_model(backbone).to(device)
    criterion = build_criterion(label_weights, device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_auroc = 0.0
    patience_count = 0

    if resume_from and os.path.exists(resume_from):
        model = load_checkpoint(model, resume_from, device)

    logger = TrainingLogger(os.path.join(LOGS_DIR, f"training_{backbone}.csv"))

    print(f"\n=== Training: {backbone} for {num_epochs} epochs ===\n")
    print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'ValLoss':>9} | {'TrainAUC':>9} | {'ValAUC':>8} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_auroc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_auroc, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        logger.log(epoch, train_loss, val_loss,
                   train_auroc["macro"], val_auroc["macro"], lr_now, elapsed)

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
            f"{train_auroc['macro']:>9.4f} | {val_auroc['macro']:>8.4f} | "
            f"{lr_now:>8.6f}  ({elapsed:.0f}s)"
        )

        is_best = val_auroc["macro"] > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc["macro"]
            patience_count = 0
        else:
            patience_count += 1

        save_checkpoint(model, optimizer, epoch, val_auroc["macro"], backbone, is_best)

        if patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n[i] Early stopping at epoch {epoch}.")
            break

    print(f"\n[✓] Training complete.  Best Val AUROC: {best_val_auroc:.4f}")
    return model, best_val_auroc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",    default=BACKBONE)
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--resume_from", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.backbone, args.epochs, args.batch_size, args.lr, args.resume_from)