"""
Evaluation Script — Test Set Metrics + Plots
Author: Chenduluru Siva | 7151CEM
Usage: python src/evaluation/evaluate.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support,
    multilabel_confusion_matrix
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    DISEASE_LABELS, SAVED_MODELS_DIR, OUTPUTS_DIR,
    CLASSIFICATION_THRESHOLD, BACKBONE, SPLITS_DIR, BATCH_SIZE
)
from src.preprocessing.dataset import ChestXrayDataset, get_eval_transform
from src.models.densenet_model import build_model, load_checkpoint

EVAL_OUTPUT = os.path.join(OUTPUTS_DIR, "evaluation")


def run_inference(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="  Inference"):
            probs = model.predict_proba(images.to(device))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs)


def compute_full_metrics(targets, probs, threshold=CLASSIFICATION_THRESHOLD):
    preds = (probs >= threshold).astype(int)
    rows, aurocs = [], []
    for i, label in enumerate(DISEASE_LABELS):
        y_true  = targets[:, i]
        y_score = probs[:, i]
        y_pred  = preds[:, i]
        try:
            auroc = roc_auc_score(y_true, y_score) if y_true.sum() > 0 else float("nan")
            if not np.isnan(auroc):
                aurocs.append(auroc)
        except Exception:
            auroc = float("nan")
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append({"Disease": label, "AUROC": round(auroc,4),
                     "Precision": round(p,4), "Recall": round(r,4),
                     "F1": round(f1,4), "Positives": int(y_true.sum())})

    df = pd.DataFrame(rows)
    macro = np.nanmean(aurocs)
    print("\n" + "=" * 65)
    print("  Per-Class Results")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\n  Macro-Average AUROC: {macro:.4f}")
    print("=" * 65)
    return df, macro


def plot_roc_curves(targets, probs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    fig.suptitle("ROC Curves — NIH ChestX-ray14 (DenseNet-121)", fontsize=13)
    for i, label in enumerate(DISEASE_LABELS):
        ax = axes[i]
        if targets[:, i].sum() == 0:
            ax.text(0.5, 0.5, "No positives", ha="center"); ax.set_title(label); continue
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={roc_auc:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_title(label, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
    for j in range(len(DISEASE_LABELS), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] ROC curves → {path}")


def plot_confusion_matrices(targets, probs, save_dir, threshold=CLASSIFICATION_THRESHOLD):
    os.makedirs(save_dir, exist_ok=True)
    preds = (probs >= threshold).astype(int)
    cms   = multilabel_confusion_matrix(targets, preds)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices", fontsize=13)
    for i, (label, cm) in enumerate(zip(DISEASE_LABELS, cms)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Pred-", "Pred+"],
                    yticklabels=["True-", "True+"])
        axes[i].set_title(label, fontsize=9)
    for j in range(len(DISEASE_LABELS), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] Confusion matrices → {path}")


def plot_auroc_bar(metrics_df, save_dir, model_name="DenseNet-121"):
    os.makedirs(save_dir, exist_ok=True)
    df = metrics_df.dropna(subset=["AUROC"]).sort_values("AUROC", ascending=True)
    colors = ["#d73027" if v < 0.7 else "#fee08b" if v < 0.8 else "#1a9850"
              for v in df["AUROC"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["Disease"], df["AUROC"], color=colors, edgecolor="grey")
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.axvline(x=df["AUROC"].mean(), color="navy", linestyle=":", linewidth=1.5,
               label=f"Mean ({df['AUROC'].mean():.3f})")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(f"Per-Class AUROC — {model_name}", fontsize=13)
    ax.legend()
    for bar, val in zip(bars, df["AUROC"]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "auroc_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] AUROC bar chart → {path}")


def plot_ablation_comparison(baseline_auroc, qaware_auroc, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    labels = list(baseline_auroc.keys())
    base_v = [baseline_auroc[l] for l in labels]
    qa_v   = [qaware_auroc[l]   for l in labels]
    delta  = [q - b for q, b in zip(qa_v, base_v)]
    x     = np.arange(len(labels))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.bar(x - width/2, base_v, width, label="Baseline",     color="#4393c3", alpha=0.85)
    ax1.bar(x + width/2, qa_v,   width, label="Quality-Aware", color="#d6604d", alpha=0.85)
    ax1.set_ylabel("AUROC"); ax1.set_title("Baseline vs Quality-Aware AUROC")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(); ax1.set_ylim(0.4, 1.0)
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    ax2.barh(labels, delta, color=colors, edgecolor="grey")
    ax2.axvline(x=0, color="black"); ax2.set_xlabel("AUROC Δ (QA − Baseline)")
    ax2.set_title("AUROC Improvement from Quality Assessment")
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] Ablation comparison → {path}")


def evaluate(model_path=None, backbone=BACKBONE):
    os.makedirs(EVAL_OUTPUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(backbone).to(device)

    checkpoint = model_path or os.path.join(SAVED_MODELS_DIR, f"best_{backbone}.pt")
    if os.path.exists(checkpoint):
        model = load_checkpoint(model, checkpoint, device)
    else:
        print("[!] No checkpoint found — random weights (demo only)")

    test_dataset = ChestXrayDataset(
        os.path.join(SPLITS_DIR, "test.csv"), get_eval_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"[✓] Test set: {len(test_dataset):,} images")

    targets, probs = run_inference(model, test_loader, device)
    metrics_df, macro_auroc = compute_full_metrics(targets, probs)

    csv_path = os.path.join(EVAL_OUTPUT, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[✓] Metrics CSV → {csv_path}")

    plot_roc_curves(targets, probs, EVAL_OUTPUT)
    plot_confusion_matrices(targets, probs, EVAL_OUTPUT)
    plot_auroc_bar(metrics_df, EVAL_OUTPUT, backbone)

    return targets, probs, metrics_df, macro_auroc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--backbone",   default=BACKBONE)
    args = p.parse_args()
    evaluate(args.model_path, args.backbone)