"""
Master Pipeline Script
Author: Chenduluru Siva | 7151CEM
Usage: python run_pipeline.py --step all
"""

import os, sys, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import BACKBONE, NUM_EPOCHS, BATCH_SIZE


def step1(): 
    from src.preprocessing.download_dataset import download_metadata, verify_metadata, check_images
    download_metadata(); verify_metadata(); check_images()

def step2():
    from src.preprocessing.split_dataset import run_split
    run_split()

def step3(backbone, epochs, batch_size):
    from src.training.train import train
    train(backbone=backbone, num_epochs=epochs, batch_size=batch_size)

def step4(backbone):
    from src.evaluation.evaluate import evaluate
    evaluate(backbone=backbone)

def step5(backbone):
    from src.evaluation.ablation_study import run_ablation
    run_ablation(backbone=backbone)

def step6(backbone, n):
    from src.explainability.gradcam import run_gradcam_on_test_samples
    run_gradcam_on_test_samples(backbone=backbone, n_samples=n)

def step7():
    import numpy as np
    from src.reporting.report_generator import ReportGenerator
    np.random.seed(42)
    probs      = np.random.rand(14)
    probs[7]   = 0.88; probs[6] = 0.72
    gen        = ReportGenerator()
    report     = gen.generate(probs, "demo.png", qa_score=91.2)
    path       = gen.save_report(report)
    print(report.full_text)
    print(f"\n[✓] Saved → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step",       default="all",
                   help="setup|split|train|eval|ablation|gradcam|report|all")
    p.add_argument("--backbone",   default=BACKBONE)
    p.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--n_gradcam",  type=int, default=10)
    args = p.parse_args()
    s    = args.step.lower()

    if s in ("setup",    "all"): step1()
    if s in ("split",    "all"): step2()
    if s in ("train",    "all"): step3(args.backbone, args.epochs, args.batch_size)
    if s in ("eval",     "all"): step4(args.backbone)
    if s in ("ablation", "all"): step5(args.backbone)
    if s in ("gradcam",  "all"): step6(args.backbone, args.n_gradcam)
    if s in ("report",   "all"): step7()

    print("\n✅ Done! Launch UI with:  streamlit run streamlit_app/app.py")