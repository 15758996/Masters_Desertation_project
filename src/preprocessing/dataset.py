"""
ChestXrayDataset — PyTorch Dataset with Augmentation
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    IMAGES_DIR, SPLITS_DIR, DISEASE_LABELS,
    IMAGE_SIZE, PIXEL_MEAN, PIXEL_STD,
    BATCH_SIZE, NUM_WORKERS, SEED
)


def get_train_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])


def get_eval_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform=None, images_dir=IMAGES_DIR):
        self.df         = pd.read_csv(csv_path)
        self.transform  = transform
        self.images_dir = images_dir
        self.labels     = DISEASE_LABELS

        for label in self.labels:
            if label not in self.df.columns:
                self.df[label] = self.df["Finding Labels"].apply(
                    lambda x: 1 if label in str(x).split("|") else 0
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["Image Index"])

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        target = torch.FloatTensor([row[label] for label in self.labels])
        return image, target, row["Image Index"]

    def get_label_weights(self):
        counts  = self.df[self.labels].sum(axis=0)
        n_total = len(self.df)
        weights = (n_total - counts) / (counts + 1e-6)
        return torch.FloatTensor(weights.values)


def get_dataloaders(splits_dir=SPLITS_DIR, batch_size=BATCH_SIZE):
    train_dataset = ChestXrayDataset(
        os.path.join(splits_dir, "train.csv"), get_train_transform()
    )
    val_dataset = ChestXrayDataset(
        os.path.join(splits_dir, "val.csv"), get_eval_transform()
    )
    test_dataset = ChestXrayDataset(
        os.path.join(splits_dir, "test.csv"), get_eval_transform()
    )

    torch.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"[✓] Train : {len(train_dataset):,} images")
    print(f"[✓] Val   : {len(val_dataset):,} images")
    print(f"[✓] Test  : {len(test_dataset):,} images")

    return train_loader, val_loader, test_loader, train_dataset.get_label_weights()


if __name__ == "__main__":
    train_loader, val_loader, test_loader, weights = get_dataloaders()
    images, targets, names = next(iter(train_loader))
    print(f"Image batch : {images.shape}")
    print(f"Target batch: {targets.shape}")