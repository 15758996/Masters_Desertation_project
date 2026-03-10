"""
Grad-CAM Explainability Module
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    DISEASE_LABELS, OUTPUTS_DIR, IMAGE_SIZE,
    PIXEL_MEAN, PIXEL_STD, SAVED_MODELS_DIR, BACKBONE
)
from src.models.densenet_model import build_model, load_checkpoint
from src.preprocessing.dataset import get_eval_transform

GRADCAM_OUTPUT = os.path.join(OUTPUTS_DIR, "gradcam")


class GradCAM:
    """Grad-CAM for DenseNet-121."""
    def __init__(self, model, target_layer="features.denseblock4"):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks(target_layer)

    def _register_hooks(self, layer_name):
        layer = self.model
        for part in layer_name.split("."):
            layer = getattr(layer, part)

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, gin, gout):
            self.gradients = gout[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor, class_idx):
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)
        logits = self.model(image_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        weights = self.gradients[0].mean(dim=(1, 2))
        cam     = torch.zeros(self.activations.shape[2:])
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = F.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def overlay(self, original_image, heatmap, alpha=0.4):
        hm_uint8  = np.uint8(255 * heatmap)
        hm_color  = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_rgb    = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image * 255)
        return cv2.addWeighted(original_image, 1 - alpha, hm_rgb, alpha, 0)


def denormalise(tensor):
    mean = np.array(PIXEL_MEAN)
    std  = np.array(PIXEL_STD)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
    return img


def generate_gradcam_panel(model, image_tensor, probs, filename,
                            top_k=4, save_dir=GRADCAM_OUTPUT, device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    gcam       = GradCAM(model)
    orig_np    = denormalise(image_tensor.squeeze(0))

    fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))
    axes[0].imshow(orig_np); axes[0].set_title("Original X-Ray"); axes[0].axis("off")

    for col, idx in enumerate(sorted_idx, start=1):
        heatmap = gcam.generate(image_tensor.to(device), int(idx))
        overlay = gcam.overlay(orig_np, heatmap)
        axes[col].imshow(overlay)
        axes[col].set_title(f"{DISEASE_LABELS[idx]}\np={probs[idx]:.3f}", fontsize=9)
        axes[col].axis("off")

    plt.suptitle(f"Grad-CAM — {filename}", fontsize=10)
    plt.tight_layout()
    safe    = filename.replace("/", "_").replace(".png", "")
    path    = os.path.join(save_dir, f"gradcam_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] Grad-CAM → {path}")
    return path


def run_gradcam_on_test_samples(model_path=None, backbone=BACKBONE, n_samples=10):
    import pandas as pd
    from src.config import IMAGES_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(backbone).to(device)
    ckpt   = model_path or os.path.join(SAVED_MODELS_DIR, f"best_{backbone}.pt")
    if os.path.exists(ckpt):
        model = load_checkpoint(model, ckpt, device)

    from src.config import SPLITS_DIR
    transform = get_eval_transform()
    test_csv  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    sample    = test_csv.sample(n=min(n_samples, len(test_csv)), random_state=42)

    from src.preprocessing.quality_assessment import ImageQualityAssessor
    assessor = ImageQualityAssessor()

    for _, row in sample.iterrows():
        img_path = os.path.join(IMAGES_DIR, row["Image Index"])
        if not os.path.exists(img_path):
            continue
        qa      = assessor.assess(img_path)
        print(f"  {row['Image Index']}  QA={qa.overall_score}/100")
        pil_img = Image.open(img_path).convert("RGB")
        tensor  = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.predict_proba(tensor).cpu().numpy()[0]
        generate_gradcam_panel(model, tensor, probs, row["Image Index"],
                                top_k=4, device=device)


if __name__ == "__main__":
    run_gradcam_on_test_samples(n_samples=5)