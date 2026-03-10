"""
Patient-Level Dataset Splitting
Author: Chenduluru Siva | 7151CEM
Handles: Data_Entry_2017_v2020.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    METADATA_CSV, SPLITS_DIR, IMAGES_DIR, DISEASE_LABELS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
)


def find_metadata_csv(raw_dir):
    """
    Auto-detect the metadata CSV regardless of exact filename.
    Checks for both Data_Entry_2017_v2020.csv and Data_Entry_2017.csv
    """
    candidates = [
        "Data_Entry_2017_v2020.csv",
        "Data_Entry_2017.csv",
        "Data_Entry_2017_v2020 (1).csv",
    ]
    for name in candidates:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            print(f"[✓] Found metadata CSV: {name}")
            return path

    # Search directory for any CSV
    for f in os.listdir(raw_dir):
        if f.endswith(".csv") and "Entry" in f:
            print(f"[✓] Found metadata CSV: {f}")
            return os.path.join(raw_dir, f)

    return None


def inspect_csv(df):
    """Print column names so we know exactly what we're working with."""
    print(f"\n[i] CSV shape     : {df.shape}")
    print(f"[i] CSV columns   : {list(df.columns)}")
    print(f"[i] First row     : \n{df.iloc[0]}")


def standardise_columns(df):
    """
    Rename columns to standard names regardless of CSV version.
    Data_Entry_2017_v2020.csv uses slightly different column names.
    """
    rename_map = {}

    # Image filename column
    for col in df.columns:
        if "image" in col.lower() and "index" in col.lower():
            rename_map[col] = "Image Index"
        elif col.lower() in ("image_index", "imageid", "filename"):
            rename_map[col] = "Image Index"

    # Finding labels column
    for col in df.columns:
        if "finding" in col.lower() and "label" in col.lower():
            rename_map[col] = "Finding Labels"
        elif col.lower() in ("labels", "findings", "finding_labels"):
            rename_map[col] = "Finding Labels"

    # Patient ID column
    for col in df.columns:
        if "patient" in col.lower() and "id" in col.lower():
            rename_map[col] = "Patient ID"
        elif col.lower() in ("patient_id", "patientid"):
            rename_map[col] = "Patient ID"

    if rename_map:
        print(f"[i] Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)

    # Verify required columns exist
    required = ["Image Index", "Finding Labels", "Patient ID"]
    for col in required:
        if col not in df.columns:
            print(f"[!] Column '{col}' not found. Available: {list(df.columns)}")
            # Try fuzzy match
            for existing in df.columns:
                if col.lower().replace(" ", "") in existing.lower().replace(" ", ""):
                    print(f"    → Using '{existing}' as '{col}'")
                    df = df.rename(columns={existing: col})
                    break

    return df


def build_label_columns(df):
    """Create binary columns for each of the 14 disease labels."""
    for label in DISEASE_LABELS:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1 if label in str(x).split("|") else 0
        )
    df["No_Finding"] = df["Finding Labels"].apply(
        lambda x: 1 if "No Finding" in str(x).split("|") else 0
    )
    return df


def filter_existing_images(df):
    """Only keep rows whose image file exists in data/raw/images/."""
    if not os.path.isdir(IMAGES_DIR):
        print(f"[!] Images directory not found: {IMAGES_DIR}")
        print("    Keeping all metadata rows (images not yet downloaded).")
        return df

    existing = set(os.listdir(IMAGES_DIR))
    if not existing:
        print("[!] Images directory is empty — keeping all rows for now.")
        return df

    before = len(df)
    df     = df[df["Image Index"].isin(existing)].reset_index(drop=True)
    print(f"[i] Filtered to existing images: {before:,} → {len(df):,} rows")
    return df


def patient_level_split(df):
    """
    Split at patient level — same patient NEVER appears in train + test.
    This prevents data leakage.
    """
    patient_ids = df["Patient ID"].unique()
    print(f"[i] Unique patients: {len(patient_ids):,}")
    np.random.seed(SEED)

    # Step 1: train vs (val + test)
    train_ids, valtest_ids = train_test_split(
        patient_ids,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )

    # Step 2: val vs test
    val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_ids, test_ids = train_test_split(
        valtest_ids,
        test_size=(1.0 - val_fraction),
        random_state=SEED
    )

    train_df = df[df["Patient ID"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["Patient ID"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["Patient ID"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def print_split_summary(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n[i] Split Summary:")
    print(f"  {'Split':<10} {'Images':>10} {'Patients':>10} {'%':>6}")
    print("  " + "-" * 42)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pct = 100 * len(df) / total
        print(f"  {name:<10} {len(df):>10,} "
              f"{df['Patient ID'].nunique():>10,} {pct:>5.1f}%")

    print("\n[i] Label distribution in train set:")
    print(f"  {'Label':<25} {'Count':>8} {'%':>6}")
    print("  " + "-" * 42)
    for label in DISEASE_LABELS:
        if label in train_df.columns:
            cnt = int(train_df[label].sum())
            pct = 100 * cnt / len(train_df)
            print(f"  {label:<25} {cnt:>8,}  {pct:>5.1f}%")


def run_split():
    print("\n=== Patient-Level Dataset Splitting ===\n")
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Auto-detect CSV
    from src.config import RAW_DIR
    csv_path = find_metadata_csv(RAW_DIR)

    if csv_path is None:
        print(f"[✗] No metadata CSV found in {RAW_DIR}")
        print("    Please place Data_Entry_2017_v2020.csv in data/raw/")
        return

    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded CSV: {len(df):,} rows")

    # Inspect and standardise columns
    inspect_csv(df)
    df = standardise_columns(df)

    # Filter to existing images (skips if none downloaded yet)
    df = filter_existing_images(df)

    # Build binary label columns
    df = build_label_columns(df)

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(df)
    print_split_summary(train_df, val_df, test_df)

    # Save splits
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path   = os.path.join(SPLITS_DIR, "val.csv")
    test_path  = os.path.join(SPLITS_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n[✓] Splits saved to {SPLITS_DIR}/")
    print(f"    train.csv → {len(train_df):,} images")
    print(f"    val.csv   → {len(val_df):,} images")
    print(f"    test.csv  → {len(test_df):,} images")

    # Verify no patient overlap
    tr = set(train_df["Patient ID"].unique())
    vl = set(val_df["Patient ID"].unique())
    te = set(test_df["Patient ID"].unique())
    assert len(tr & vl) == 0, "LEAKAGE: train/val overlap!"
    assert len(tr & te) == 0, "LEAKAGE: train/test overlap!"
    assert len(vl & te) == 0, "LEAKAGE: val/test overlap!"
    print("[✓] No patient overlap confirmed — zero data leakage.")


if __name__ == "__main__":
    run_split()