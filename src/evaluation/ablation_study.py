"""
Ablation Study — Baseline vs Quality-Aware Pipeline
Author: Chenduluru Siva | 7151CEM
Usage: python src/evaluation/ablation_study.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    SPLITS_DIR, IMAGES_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR,
    DISEASE_LABELS, BACKBONE, BATCH_SIZE
)
from src.preprocessing.dataset import ChestXrayDataset, get_eval_transform
from src.preprocessing.quality_assessment import ImageQualityAssessor
from src.models.densenet_model import build_model, load_checkpoint
from src.evaluation.evaluate import compute_full_metrics, plot_ablation_comparison

ABLATION_OUTPUT = os.path.join(OUTPUTS_DIR, "ablation")


def run_ablation(model_path=None, backbone=BACKBONE, n_samples=None):
    os.makedirs(ABLATION_OUTPUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(backbone).to(device)
    ckpt  = model_path or os.path.join(SAVED_MODELS_DIR, f"best_{backbone}.pt")
    if os.path.exists(ckpt):
        model = load_checkpoint(model, ckpt, device)
    model.eval()

    test_dataset = ChestXrayDataset(
        os.path.join(SPLITS_DIR, "test.csv"), get_eval_transform()
    )
    if n_samples:
        idx          = np.random.choice(len(test_dataset), n_samples, replace=False)
        test_dataset = Subset(test_dataset, idx)

    print(f"\n[✓] Test samples: {len(test_dataset):,}")

    # ── A: Baseline ──────────────────────────────────────────
    print("\n[1] Baseline (no QA filter) …")
    loader_a = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    base_t, base_p = [], []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader_a):
            base_p.append(model.predict_proba(images.to(device)).cpu().numpy())
            base_t.append(targets.numpy())
    base_targets = np.concatenate(base_t)
    base_probs   = np.concatenate(base_p)
    base_df, base_macro = compute_full_metrics(base_targets, base_probs)
    print(f"  Baseline Macro AUROC: {base_macro:.4f}")

    # ── B: Quality-Aware ─────────────────────────────────────
    print("\n[2] Quality-Aware (QA filtered) …")
    assessor = ImageQualityAssessor()
    rows     = (test_dataset.dataset.df.iloc[test_dataset.indices].reset_index(drop=True)
                if isinstance(test_dataset, Subset) else test_dataset.df)

    qa_passed, qa_rejected = [], 0
    for idx, row in tqdm(rows.iterrows(), total=len(rows), desc="  QA"):
        img_path = os.path.join(IMAGES_DIR, row["Image Index"])
        if os.path.exists(img_path):
            report = assessor.assess(img_path)
            if report.is_acceptable:
                qa_passed.append(idx)
            else:
                qa_rejected += 1
        else:
            qa_passed.append(idx)

    print(f"  QA accepted: {len(qa_passed):,}  |  QA rejected: {qa_rejected:,}")

    if isinstance(test_dataset, Subset):
        qa_subset = Subset(test_dataset.dataset,
                           [test_dataset.indices[i] for i in qa_passed])
    else:
        qa_subset = Subset(test_dataset, qa_passed)

    loader_b = DataLoader(qa_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    qa_t, qa_p = [], []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader_b):
            qa_p.append(model.predict_proba(images.to(device)).cpu().numpy())
            qa_t.append(targets.numpy())
    qa_targets = np.concatenate(qa_t)
    qa_probs   = np.concatenate(qa_p)
    qa_df, qa_macro = compute_full_metrics(qa_targets, qa_probs)
    print(f"  QA-Aware Macro AUROC: {qa_macro:.4f}")

    # ── Summary ───────────────────────────────────────────────
    base_pc = dict(zip(base_df["Disease"], base_df["AUROC"]))
    qa_pc   = dict(zip(qa_df["Disease"],   qa_df["AUROC"]))

    print("\n" + "=" * 60)
    print(f"  {'Disease':<25} {'Baseline':>9} {'QA-Aware':>9} {'Delta':>8}")
    print("  " + "-" * 55)
    for label in DISEASE_LABELS:
        b = base_pc.get(label, float("nan"))
        q = qa_pc.get(label,   float("nan"))
        d = q - b if not (np.isnan(b) or np.isnan(q)) else float("nan")
        arrow = "▲" if not np.isnan(d) and d > 0 else "▼" if not np.isnan(d) else "?"
        print(f"  {label:<25} {b:>9.4f} {q:>9.4f} {d:>+7.4f} {arrow}")
    print("  " + "-" * 55)
    print(f"  {'MACRO':<25} {base_macro:>9.4f} {qa_macro:>9.4f} {qa_macro-base_macro:>+7.4f}")
    print("=" * 60)

    comparison = pd.DataFrame({
        "Disease":        DISEASE_LABELS,
        "Baseline_AUROC": [base_pc.get(l, float("nan")) for l in DISEASE_LABELS],
        "QA_AUROC":       [qa_pc.get(l,   float("nan")) for l in DISEASE_LABELS],
        "Delta":          [qa_pc.get(l, float("nan")) - base_pc.get(l, float("nan"))
                           for l in DISEASE_LABELS]
    })
    comparison.to_csv(os.path.join(ABLATION_OUTPUT, "ablation_comparison.csv"), index=False)

    summary = {
        "baseline_macro_auroc":  round(base_macro, 4),
        "qa_aware_macro_auroc":  round(qa_macro,   4),
        "delta":                 round(qa_macro - base_macro, 4),
        "qa_rejection_rate_pct": round(100 * qa_rejected / len(rows), 2),
    }
    with open(os.path.join(ABLATION_OUTPUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_ablation_comparison(base_pc, qa_pc, ABLATION_OUTPUT)
    print(f"\n[✓] Ablation outputs saved to {ABLATION_OUTPUT}/")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--backbone",   default=BACKBONE)
    p.add_argument("--n_samples",  type=int, default=None)
    args = p.parse_args()
    run_ablation(args.model_path, args.backbone, args.n_samples)