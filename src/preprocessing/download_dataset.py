"""
NIH ChestX-ray14 Dataset Setup
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RAW_DIR, IMAGES_DIR, DISEASE_LABELS


def find_metadata_csv():
    candidates = [
        "Data_Entry_2017_v2020.csv",
        "Data_Entry_2017.csv",
    ]
    for name in candidates:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            print(f"[✓] Found metadata: {name}")
            return path
    return None


def verify_metadata():
    path = find_metadata_csv()
    if not path:
        print("[✗] No metadata CSV found in data/raw/")
        print("    Place Data_Entry_2017_v2020.csv in data/raw/")
        return None
    df = pd.read_csv(path)
    print(f"[✓] Loaded: {len(df):,} rows  |  Columns: {list(df.columns)}")
    return df


def check_images():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    imgs = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
    print(f"[i] PNG images found in data/raw/images/: {len(imgs):,}")
    if len(imgs) == 0:
        print()
        print("=" * 60)
        print("  DOWNLOAD REQUIRED: NIH ChestX-ray14 Images")
        print("=" * 60)
        print("  1. Go to: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("  2. Download all 12 archives:")
        print("     images_001.tar.gz through images_012.tar.gz")
        print("  3. Extract all .png files into:")
        print(f"     {IMAGES_DIR}")
        print("  4. Re-run: python src/preprocessing/download_dataset.py")
        print("=" * 60)
    return len(imgs)


def print_label_stats(df):
    # Handle both column name versions
    label_col = "Finding Labels" if "Finding Labels" in df.columns else df.columns[1]
    print(f"\n[i] Disease Distribution (column: '{label_col}'):")
    print("-" * 45)
    all_labels = df[label_col].str.split("|").explode()
    counts     = all_labels.value_counts()
    for label in DISEASE_LABELS:
        cnt = counts.get(label, 0)
        print(f"  {label:<25} {cnt:>7,}  ({100*cnt/len(df):.1f}%)")
    nf = counts.get("No Finding", 0)
    print(f"  {'No Finding':<25} {nf:>7,}  ({100*nf/len(df):.1f}%)")


if __name__ == "__main__":
    print("\n=== NIH ChestX-ray14 Dataset Setup ===\n")
    df = verify_metadata()
    if df is not None:
        print_label_stats(df)
    check_images()
    print("\n[✓] Setup check complete.")