"""
DenseNet-121 Multi-Label Classification Model
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import NUM_CLASSES, BACKBONE, DROPOUT_RATE, DISEASE_LABELS


class ChestXrayDenseNet(nn.Module):
    """
    DenseNet-121 for 14-class multi-label chest X-ray classification.
    Returns raw logits — apply sigmoid for probabilities.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        base           = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features  = base.features
        self.relu      = nn.ReLU(inplace=True)
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        in_features    = base.classifier.in_features   # 1024
        self.dropout   = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        f    = self.features(x)
        f    = self.relu(f)
        f    = self.pool(f)
        f    = torch.flatten(f, 1)
        f    = self.dropout(f)
        return self.classifier(f)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def get_feature_maps(self, x):
        return self.relu(self.features(x))


class ChestXrayResNet(nn.Module):
    """ResNet-50 alternative for ablation comparison."""
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        base            = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder    = nn.Sequential(*list(base.children())[:-1])
        in_features     = base.fc.in_features   # 2048
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        f = torch.flatten(self.encoder(x), 1)
        return self.classifier(self.dropout(f))

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def build_model(backbone=BACKBONE, num_classes=NUM_CLASSES):
    backbone = backbone.lower()
    if backbone == "densenet121":
        model = ChestXrayDenseNet(num_classes=num_classes)
        print(f"[✓] Built DenseNet-121  —  {num_classes} output classes")
    elif backbone == "resnet50":
        model = ChestXrayResNet(num_classes=num_classes)
        print(f"[✓] Built ResNet-50  —  {num_classes} output classes")
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total params    : {total:,}")
    print(f"    Trainable params: {trainable:,}")
    return model


def load_checkpoint(model, checkpoint_path, device):
    ckpt      = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch     = ckpt.get("epoch", "?")
    val_auroc = ckpt.get("val_auroc", "?")
    print(f"[✓] Loaded checkpoint  epoch={epoch}  val_auroc={val_auroc}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("densenet121").to(device)
    dummy  = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        probs = model.predict_proba(dummy)
    print(f"Output shape: {probs.shape}")   # (2, 14)
    print(f"Labels: {DISEASE_LABELS}")