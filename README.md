# 🫁 Intelligent Multi-Stage Chest X-Ray Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

**A production-grade multi-stage deep learning pipeline for automated chest X-ray analysis**

*7151CEM Computing Individual Research Project — MSc Computing — Coventry University*

[Overview](#-overview) • [Pipeline](#-pipeline-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Code Guide](#-code-guide) • [References](#-references)

</div>

---

## 🔬 Overview

Chest X-rays are the most commonly performed medical imaging procedure globally, with approximately **2 billion examinations** conducted annually. Despite this volume, a recognised global shortage of radiologists creates diagnostic backlogs. Existing AI systems such as CheXNet (Rajpurkar et al., 2017) focus exclusively on classification accuracy, ignoring clinical workflow requirements like image quality, explainability, and structured reporting.

This project proposes and implements an **Intelligent Multi-Stage Chest X-Ray Analysis System** — a five-stage end-to-end pipeline that mirrors real clinical diagnostic workflow:

| Problem with Existing Systems | How This Project Solves It |
|-------------------------------|---------------------------|
| No image quality check before classification | Stage 1: Heuristic QA assessment rejects/flags suboptimal images |
| Black-box predictions with no explanation | Stage 3: Grad-CAM heatmaps show model attention regions |
| Classification only — no clinical output | Stage 5: Structured clinical report with urgency triage |
| No integrated workflow | All stages connected in one end-to-end pipeline |
| No quality impact measurement | Ablation study: Baseline vs Quality-Aware AUROC comparison |

### Research Question

> *Can an integrated multi-stage deep learning system combining image quality assessment, multi-label disease classification, and automated structured reporting improve diagnostic robustness compared to conventional single-stage classification approaches?*

---

## 🏗 Pipeline Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Chest   │    │   Stage 01   │    │   Stage 02   │    │   Stage 03   │    │   Stage 04   │    │ Clinical │
│  X-Ray   │───▶│   Quality    │───▶│  DenseNet-   │───▶│  Grad-CAM   │───▶│  Attention   │───▶│  Report  │
│  Input   │    │  Assessment  │    │  121 + Sig.  │    │ Heatmaps    │    │ Feature Maps │    │ Output   │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
                  OpenCV             BCEWithLogits        Selvaraju           DenseBlock-4        Template
                  Laplacian          14-class             et al. 2017         Channel Attn.       NLG +
                  Brightness         Multi-Label          Gradient            Mean/Max/Top-K      Urgency
                  Contrast           Classification       Weighted CAM        Activation          Triage
                  Black Ratio        Sigmoid Output       DenseBlock-4        Maps
```

### Stage Details

| Stage | Component | Technology | Research Basis |
|-------|-----------|------------|----------------|
| **01** | Image Quality Assessment | OpenCV — Laplacian variance, brightness, contrast, black border ratio | Original contribution |
| **02** | Multi-Label Classification | DenseNet-121, BCEWithLogitsLoss + class weights, AdamW | Rajpurkar et al. (2017) |
| **03** | Grad-CAM Explainability | Gradient hooks on DenseBlock-4, ReLU activation, JET colourmap | Selvaraju et al. (2017) |
| **04** | Attention Feature Maps | DenseBlock-4 channel activations, mean/max/combined/per-disease | Original contribution |
| **05** | Structured Report Generation | Template NLG, urgency triage (URGENT/PRIORITY/ROUTINE) | Extends Liu et al. (2019) |

---

## 📁 Project Structure

```
Dissertation-Project-S/
│
├── 📂 data/
│   ├── raw/
│   │   ├── images/                    ← NIH ChestX-ray14 images (112,120 PNGs)
│   │   ├── Data_Entry_2017_v2020.csv  ← NIH metadata CSV
│   │   └── BBox_List_2017.csv         ← Bounding box annotations
│   ├── processed/                     ← Preprocessed data cache
│   └── splits/
│       ├── train.csv                  ← 70% patient-level split (generated)
│       ├── val.csv                    ← 10% patient-level split (generated)
│       └── test.csv                   ← 20% patient-level split (generated)
│
├── 📂 src/
│   ├── config.py                      ← Central configuration (all hyperparameters)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── download_dataset.py        ← Dataset setup and verification
│   │   ├── split_dataset.py           ← Patient-level train/val/test splitting
│   │   ├── dataset.py                 ← PyTorch Dataset class + DataLoaders
│   │   └── quality_assessment.py     ← Stage 1: Image QA module
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── densenet_model.py          ← Stage 2: DenseNet-121 + ResNet-50
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py                   ← Training loop with early stopping
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py                ← Test set metrics and visualisations
│   │   └── ablation_study.py          ← Baseline vs Quality-Aware experiment
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py                 ← Stage 3: Grad-CAM heatmaps
│   │
│   └── reporting/
│       ├── __init__.py
│       └── report_generator.py        ← Stage 5: Structured clinical reports
│
├── 📂 streamlit_app/
│   ├── __init__.py
│   └── app.py                         ← Full interactive web UI (v3.0)
│
├── 📂 models/
│   ├── checkpoints/                   ← Per-epoch saved checkpoints
│   └── saved/
│       └── best_densenet121.pt        ← Best model by validation AUROC
│
├── 📂 outputs/
│   ├── evaluation/                    ← ROC curves, confusion matrices, AUROC plots
│   ├── ablation/                      ← Ablation comparison CSV and plots
│   ├── gradcam/                       ← Grad-CAM heatmap panels
│   └── reports/                       ← Generated structured text reports
│
├── 📂 logs/
│   └── training_densenet121.csv       ← Epoch-by-epoch training history
│
├── 📂 notebooks/                      ← Jupyter notebooks for exploration
├── 📂 tests/                          ← Unit tests
│
├── requirements.txt                   ← All Python dependencies
├── run_pipeline.py                    ← Master pipeline runner
├── create_dummy_data.py               ← Generate fake data for testing
└── README.md                          ← This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended)
- GPU with CUDA (optional but recommended — CPU training is ~15x slower)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/15758996/Dissertation-Project-S.git
cd Dissertation-Project-S
```

### Step 2 — Create Virtual Environment

```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
# CPU only (Mac/Windows without NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python Pillow
pip install tqdm streamlit plotly scipy requests openpyxl reportlab

# GPU (NVIDIA CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python Pillow
pip install tqdm streamlit plotly scipy requests openpyxl reportlab
```

### Step 4 — Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python src/models/densenet_model.py
```

Expected output:
```
[✓] Built DenseNet-121  —  14 output classes
    Total params    : 6,968,206
    Trainable params: 6,968,206
Output shape: torch.Size([2, 14])
```

---

## 📥 Dataset Setup

This project uses the **NIH ChestX-ray14** dataset (Wang et al., 2017).

> ⚠️ **Note:** Do NOT use Kaggle or UCI ML Repository datasets. The NIH dataset must be downloaded directly from the NIH.

### Step 1 — Download Metadata

```bash
python src/preprocessing/download_dataset.py
```

### Step 2 — Download Images

1. Visit: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. Download all 12 image archives:
   - `images_001.tar.gz` through `images_012.tar.gz`
3. Extract all PNG files into `data/raw/images/`

**Mac/Linux extraction:**
```bash
cd ~/Downloads
for i in $(seq -w 1 12); do tar -xzf images_0${i}.tar.gz; done
mv images/ /path/to/project/data/raw/images/
```

**Windows extraction:**
```cmd
# Use 7-Zip or WinRAR to extract all archives
# Move extracted images to data\raw\images\
```

### Step 3 — Verify Dataset

```bash
python src/preprocessing/download_dataset.py
```

Expected output:
```
[✓] Found metadata: Data_Entry_2017_v2020.csv
[✓] Loaded: 112,120 rows
[i] PNG images found: 112,120
[i] Disease Label Distribution:
  Atelectasis               11,559  (10.3%)
  Cardiomegaly               2,776   (2.5%)
  Effusion                  13,317  (11.8%)
  Infiltration              19,894  (17.7%)
  ...
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Images | 112,120 |
| Unique Patients | 30,805 |
| Disease Labels | 14 thoracic conditions |
| Label Type | Weak labels (NLP-extracted from reports) |
| Image Format | PNG, grayscale |
| Image Resolution | 1024 × 1024 (resized to 224×224 for training) |
| Label Type | Multi-label (patient can have multiple diseases) |
| Most Common | Infiltration (17.7%), Effusion (11.8%), Atelectasis (10.3%) |
| Rarest | Hernia (0.2%), Pneumonia (1.3%) |

---

## 🚀 Usage

### Complete Pipeline (Recommended)

```bash
# Run all stages in order
python run_pipeline.py --step all --backbone densenet121 --epochs 30

# Run individual steps
python run_pipeline.py --step setup      # Dataset verification
python run_pipeline.py --step split      # Create patient-level splits
python run_pipeline.py --step train      # Train the model
python run_pipeline.py --step eval       # Evaluate on test set
python run_pipeline.py --step ablation   # Run ablation study
python run_pipeline.py --step gradcam    # Generate Grad-CAM heatmaps
python run_pipeline.py --step report     # Generate sample report
```

### Step-by-Step Manual Execution

#### 1. Create Patient-Level Data Splits

```bash
python src/preprocessing/split_dataset.py
```

This performs **patient-level splitting** to prevent data leakage:
- Train: 70% of patients (~78,484 images)
- Val: 10% of patients (~11,212 images)
- Test: 20% of patients (~22,424 images)

No patient appears in more than one split. Zero data leakage is guaranteed by explicit assertions.

#### 2. Train the Model

```bash
# Standard training
python src/training/train.py --backbone densenet121 --epochs 30 --batch_size 32

# Windows (CPU) — use smaller batch size
python src/training/train.py --backbone densenet121 --epochs 30 --batch_size 8

# Resume from a checkpoint
python src/training/train.py --resume_from models/checkpoints/epoch_005_auroc_0.6800.pt

# Train ResNet-50 for comparison
python src/training/train.py --backbone resnet50 --epochs 30 --batch_size 8
```

**Training output:**
```
=== Training: densenet121 for 30 epochs ===

 Epoch | TrainLoss |  ValLoss | TrainAUC |  ValAUC |       LR
 ----------------------------------------------------------------
     1 |    0.2314 |   0.2198 |   0.6823 |  0.6233 | 0.000100  (320s)
     2 |    0.2187 |   0.2101 |   0.7102 |  0.6845 | 0.000100  (298s)
    [★] Best model saved → models/saved/best_densenet121.pt
   ...
    30 |    0.1654 |   0.1823 |   0.8124 |  0.7980 | 0.000001
[✓] Training complete. Best Val AUROC: 0.7980
```

#### 3. Evaluate on Test Set

```bash
python src/evaluation/evaluate.py

# With specific model
python src/evaluation/evaluate.py --model_path models/saved/best_densenet121.pt
```

Outputs saved to `outputs/evaluation/`:
- `metrics.csv` — Per-class AUROC, precision, recall, F1
- `roc_curves.png` — ROC curves for all 14 disease classes
- `confusion_matrices.png` — 14 binary confusion matrices
- `auroc_bar.png` — Colour-coded AUROC bar chart

#### 4. Run Ablation Study

```bash
# Full ablation
python src/evaluation/ablation_study.py

# Fast test with 200 samples
python src/evaluation/ablation_study.py --n_samples 200
```

Outputs:
- `outputs/ablation/ablation_comparison.csv`
- `outputs/ablation/ablation_comparison.png`
- `outputs/ablation/summary.json`

#### 5. Generate Grad-CAM Heatmaps

```bash
python src/explainability/gradcam.py
```

Generates heatmap panels in `outputs/gradcam/` showing:
- Original X-ray
- Heatmap overlay for top-4 predicted diseases
- Each overlay shows which regions influenced the prediction

#### 6. Generate Sample Report

```bash
python src/reporting/report_generator.py
```

#### 7. Test Image Quality Assessment

```bash
python src/preprocessing/quality_assessment.py data/raw/images/00000001_000.png
```

Output:
```
=== Quality Assessment Report ===
  File          : 00000001_000.png
  Acceptable    : True
  Overall Score : 87.3/100
  Brightness    : 128.4
  Contrast      : 52.1
  Sharpness     : 312.6
  Black Ratio   : 0.0821
  Issues        : None
  Recommendation: Accept — proceed to classification
```

#### 8. Launch Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

Opens at `http://localhost:8501`

---

## 🖥️ Streamlit Web Interface

The interactive web application provides a complete clinical interface:

### Tab 1 — Single Image Analysis
Upload any chest X-ray and get:
- **Quality Score** — Gauge chart (0–100) with brightness, contrast, sharpness, black ratio metrics
- **Disease Classification** — 14-class predictions with urgency badges (URGENT/PRIORITY/ROUTINE)
- **Probability Bars** — Interactive Plotly bar chart with threshold line
- **Probability Radar** — Polar chart showing all 14 probabilities
- **Grad-CAM Heatmaps** — Top-K disease attention overlays
- **Attention Feature Maps** — DenseBlock-4 mean/max/combined/per-disease activation maps
- **Structured Report** — Colour-coded clinical report with download button

### Tab 2 — Batch Processing
- Upload multiple X-rays simultaneously
- Urgency triage (URGENT/PRIORITY/NORMAL) per image
- Progress bar with real-time status
- Downloadable CSV summary

### Tab 3 — Model Dashboard
- Training AUROC and Loss curves (auto-loaded from logs)
- CheXNet benchmark line at 0.84
- Per-class test set AUROC bar chart
- Ablation study delta chart (if ablation has been run)

### Tab 4 — About & References
- Project information
- Pipeline architecture table
- APA 7 references

---

## 🔧 Configuration

All settings are in `src/config.py`. Change here and it updates everywhere:

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKBONE` | `densenet121` | Model architecture (`densenet121` / `resnet50`) |
| `IMAGE_SIZE` | `224` | Input resolution (px) |
| `BATCH_SIZE` | `32` | Images per gradient step (use 8 on CPU) |
| `NUM_EPOCHS` | `30` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | AdamW initial learning rate |
| `WEIGHT_DECAY` | `1e-5` | L2 regularisation |
| `LR_STEP_SIZE` | `10` | Epochs between LR decay |
| `LR_GAMMA` | `0.1` | LR decay factor |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs without improvement |
| `NUM_WORKERS` | `0` | DataLoader workers (0 for Windows) |

### Quality Assessment Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QA_BRIGHTNESS_MIN` | `30` | Minimum mean pixel value |
| `QA_BRIGHTNESS_MAX` | `220` | Maximum mean pixel value |
| `QA_CONTRAST_MIN` | `20` | Minimum pixel std deviation |
| `QA_BLUR_THRESHOLD` | `100.0` | Minimum Laplacian variance |
| `QA_BLACK_RATIO_MAX` | `0.30` | Maximum black pixel fraction |

### Classification

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASSIFICATION_THRESHOLD` | `0.5` | Sigmoid threshold for positive prediction |
| `CONFIDENCE_HIGH` | `0.70` | High confidence cutoff |
| `CONFIDENCE_MEDIUM` | `0.50` | Medium confidence cutoff |

---

## 📊 Results

### Current Results (Epoch 1 of 30)

| Disease | AUROC | Precision | Recall | F1 |
|---------|-------|-----------|--------|----|
| Atelectasis | 0.6714 | 0.1703 | 0.2627 | 0.2067 |
| Cardiomegaly | 0.6954 | 0.1326 | 0.3158 | 0.1868 |
| Effusion | 0.7052 | 0.2000 | 0.6471 | 0.3056 |
| Infiltration | 0.6437 | 0.2376 | 0.6349 | 0.3458 |
| Mass | 0.5730 | 0.0000 | 0.0000 | 0.0000 |
| Nodule | 0.5019 | 0.0354 | 0.6571 | 0.0672 |
| Pneumonia | 0.6391 | 0.0000 | 0.0000 | 0.0000 |
| Pneumothorax | 0.7623 | 0.0332 | 0.6842 | 0.0633 |
| Consolidation | 0.7064 | 0.1170 | 0.8718 | 0.2064 |
| Edema | **0.8248** | 0.1229 | 0.7073 | 0.2094 |
| Emphysema | 0.3726 | 0.0081 | 0.1333 | 0.0152 |
| Fibrosis | 0.6314 | 0.0402 | 0.2759 | 0.0702 |
| Pleural Thickening | 0.5182 | 0.0000 | 0.0000 | 0.0000 |
| Hernia | 0.7524 | 0.0000 | 0.0000 | 0.0000 |
| **Macro Average** | **0.6427** | — | — | — |

### Literature Benchmarks (Target after 30 epochs)

| Disease | CheXNet (2017) | This Project Target |
|---------|---------------|---------------------|
| Atelectasis | 0.8094 | ~0.78 |
| Cardiomegaly | 0.9248 | ~0.88 |
| Effusion | 0.8638 | ~0.83 |
| Pneumothorax | 0.8887 | ~0.85 |
| Edema | 0.8878 | ~0.86 |
| Emphysema | 0.9371 | ~0.91 |
| **Macro Average** | **0.8407** | **~0.80** |

> **Note:** Current AUROC 0.6427 is from epoch 1 only. The CheXNet benchmark of 0.84 required 30 epochs of GPU training. Training is ongoing.

---

## 📐 Model Architecture

### DenseNet-121 Modification

```python
# Original DenseNet-121 (ImageNet):  Linear(1024, 1000)
# Modified for ChestAI:

class ChestXrayDenseNet(nn.Module):
    def __init__(self, num_classes=14, dropout=0.5):
        base           = densenet121(weights=IMAGENET1K_V1)
        self.features  = base.features      # Keep all pretrained layers
        self.dropout   = nn.Dropout(p=0.5)  # Regularisation
        self.classifier = nn.Linear(1024, 14)  # 14 disease outputs

    def forward(self, x):
        # Returns raw logits — use BCEWithLogitsLoss during training
        f = adaptive_avg_pool2d(relu(self.features(x)), (1,1))
        return self.classifier(self.dropout(flatten(f, 1)))

    def predict_proba(self, x):
        # Sigmoid NOT softmax — independent probabilities for multi-label
        return torch.sigmoid(self.forward(x))
```

### Why DenseNet-121?

- Dense connectivity: every layer receives inputs from ALL preceding layers
- Addresses vanishing gradient problem in deep networks
- Encourages feature reuse — critical for subtle medical image patterns
- Established state-of-the-art on NIH ChestX-ray14 (Rajpurkar et al., 2017)
- Only 6.97M parameters — efficient for fine-tuning

### Why BCEWithLogitsLoss + Class Weights?

```
Infiltration: 19,894 positive / 92,226 negative → weight = 4.6
Pneumonia:     1,431 positive / 110,689 negative → weight = 77.4
Hernia:          227 positive / 111,893 negative → weight = 493.0
```

Without class weights the model learns to always predict "No Finding" and achieves 99% accuracy while being clinically useless.

---

## 🧪 Ablation Study Design

The ablation study is the core research experiment of this project.

### Experimental Design

| Aspect | Condition A — Baseline | Condition B — Quality-Aware |
|--------|------------------------|------------------------------|
| Images | All test images | Only QA-accepted images |
| QA filter | Not applied | Applied (score ≥ 50) |
| Model weights | Identical | Identical |
| Metric | Macro AUROC | Macro AUROC |
| Research claim | — | QA filtering improves robustness |

### Why This Design Is Valid

Both conditions use **exactly the same trained model weights**. The only variable is which images are included. Any AUROC difference is therefore causally attributable to the quality assessment stage alone. This is a controlled experiment with a single independent variable.

### How to Interpret Results

- **QA AUROC > Baseline AUROC** → Quality filtering improves robustness ✅
- **QA AUROC ≈ Baseline AUROC** → DenseNet-121 is already robust to quality
- **QA AUROC < Baseline AUROC** → Filtering removes informative difficult cases

---

## 💻 Code Guide

### `src/config.py` — Central Configuration
Single source of truth for all settings. Change once, updates everywhere.

### `src/preprocessing/split_dataset.py` — Patient-Level Splitting
Prevents data leakage by splitting at patient level. The NIH dataset contains multiple X-rays per patient. Random image-level splitting would allow the same patient in both train and test, artificially inflating results. This file uses `train_test_split` on unique patient IDs and verifies zero overlap with explicit assertions.

### `src/preprocessing/quality_assessment.py` — Stage 1 QA
Four heuristic quality checks using OpenCV:
1. **Brightness** — Mean pixel value. Below 30 = under-exposed
2. **Contrast** — Std deviation. Below 20 = flat, low contrast
3. **Sharpness** — Laplacian variance. Below 100 = blurry
4. **Black ratio** — Fraction near-black pixels. Above 30% = excessive borders

Each image gets a score 0–100. Below 50 = rejected. 50–70 = caution. Above 70 = accepted.

### `src/preprocessing/dataset.py` — PyTorch Dataset
Custom `ChestXrayDataset` class implementing PyTorch `Dataset`. Handles multi-label binary target construction from pipe-separated label strings. Training augmentations: random crop, horizontal flip, rotation, colour jitter. Evaluation: simple resize and normalise only. `get_label_weights()` computes N_neg/N_pos ratios for all 14 classes.

### `src/models/densenet_model.py` — Model Architecture
DenseNet-121 with modified classification head. Key design decision: **sigmoid not softmax** for multi-label output. Xavier initialisation on new layers. `get_feature_maps()` exposes intermediate features for Grad-CAM. `load_checkpoint()` restores full training state from saved `.pt` files.

### `src/training/train.py` — Training Loop
Complete training pipeline: weighted BCEWithLogitsLoss → AdamW with weight decay → StepLR scheduler → gradient clipping (max_norm=1.0) → early stopping → per-epoch checkpointing → CSV training log. AUROC computed on both train and validation after every epoch.

### `src/evaluation/evaluate.py` — Test Set Evaluation
Runs inference on held-out test set. Computes per-class AUROC using `sklearn.metrics.roc_auc_score`. Generates ROC curves, confusion matrices, and AUROC bar chart. All figures auto-saved to `outputs/evaluation/`.

### `src/evaluation/ablation_study.py` — Core Research Experiment
Compares baseline (all images) vs quality-aware (QA-filtered images) on identical model. Reports per-class and macro-average AUROC delta. Generates side-by-side bar chart and delta chart.

### `src/explainability/gradcam.py` — Grad-CAM
Implements Selvaraju et al. (2017). PyTorch hooks capture forward activations and backward gradients on DenseBlock-4. Gradient global average pooling gives per-channel weights. Weighted feature map sum → ReLU → bilinear resize → JET colourmap overlay. `.detach()` required before `.numpy()` on all tensors (fixes Windows compatibility).

### `src/reporting/report_generator.py` — Structured Report
Template-based NLG from classifier outputs. Urgency triage: Pneumothorax=URGENT, Pneumonia/Effusion/Edema/Mass/Consolidation=PRIORITY, others=ROUTINE. Confidence levels: ≥0.70=HIGH, ≥0.50=MEDIUM. Report format mirrors real radiology report: header, findings, impression, recommendation, disclaimer.

### `streamlit_app/app.py` — Interactive UI
Streamlit web application with 4 tabs. `@st.cache_resource` loads model once and caches. All tensors use `.detach().cpu().numpy()` for Windows compatibility. `use_column_width=True` for Streamlit version compatibility.

---

## 🔍 14 Disease Classes

| # | Disease | Urgency | Prevalence | CheXNet AUROC |
|---|---------|---------|------------|---------------|
| 0 | Atelectasis | ROUTINE | 10.3% | 0.8094 |
| 1 | Cardiomegaly | ROUTINE | 2.5% | 0.9248 |
| 2 | Effusion | PRIORITY | 11.8% | 0.8638 |
| 3 | Infiltration | ROUTINE | 17.7% | 0.7345 |
| 4 | Mass | PRIORITY | 5.1% | 0.8676 |
| 5 | Nodule | ROUTINE | 5.6% | 0.7802 |
| 6 | Pneumonia | PRIORITY | 1.3% | 0.7680 |
| 7 | **Pneumothorax** | **URGENT** | 4.7% | 0.8887 |
| 8 | Consolidation | PRIORITY | 4.2% | 0.7901 |
| 9 | Edema | PRIORITY | 2.1% | 0.8878 |
| 10 | Emphysema | ROUTINE | 2.4% | 0.9371 |
| 11 | Fibrosis | ROUTINE | 1.5% | 0.8047 |
| 12 | Pleural Thickening | ROUTINE | 3.0% | 0.8062 |
| 13 | Hernia | ROUTINE | 0.2% | 0.9164 |

---

## 🖥️ Windows-Specific Notes

If running on Windows, make these changes:

```python
# In src/config.py
NUM_WORKERS = 0      # Windows does not support multiprocessing in DataLoader
PIN_MEMORY  = False  # Disable for CPU training

# In all DataLoader calls
DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
```

For Streamlit, replace `use_container_width` with `use_column_width` if you get errors:
```python
st.image(img, use_column_width=True)       # older Streamlit
st.dataframe(df, use_container_width=True)  # newer Streamlit
```

---

## 📚 References

**Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).** ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2097–2106. https://doi.org/10.1109/CVPR.2017.369

**Rajpurkar, P., Irvin, J., Ball, R. L., Zhu, K., Yang, B., Mehta, H., … Ng, A. Y. (2017).** CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*. https://arxiv.org/abs/1711.05225

**Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., … Ng, A. Y. (2019).** CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, *33*(1), 590–597. https://doi.org/10.1609/aaai.v33i01.3301590

**Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618–626. https://doi.org/10.1109/ICCV.2017.74

**Liu, F., Wu, X., Ge, S., Fan, W., & Zou, Y. (2019).** Clinically accurate chest X-ray report generation. *Proceedings of the Machine Learning for Healthcare Conference*, *249*, 249–269. http://proceedings.mlr.press/v106/liu19a.html

**Loshchilov, I., & Hutter, F. (2019).** Decoupled weight decay regularization. *Proceedings of the International Conference on Learning Representations*. https://arxiv.org/abs/1711.05101

---

## 🙏 Acknowledgements

- **Supervisor:** Dr. Diana Hintea, Sheffield Hallam University (d.hintea@shu.ac.uk)
- **Dataset:** NIH Clinical Centre — National Institutes of Health ChestX-ray14
- **Module Leaders:** Dr. Rochelle Sassman & Dr. Omid Chatrabgoun, Coventry University
- **Backbone:** DenseNet-121 pretrained on ImageNet (PyTorch Model Zoo)

---

## ⚠️ Disclaimer

This system is a **research prototype** developed for academic purposes as part of the 7151CEM MSc Computing Individual Research Project at Coventry University. It has **NOT** been validated for clinical use and does **NOT** constitute medical advice. All outputs must be reviewed and verified by a qualified radiologist before any clinical decision is made.

---

## 📄 Licence

This project is developed for academic research purposes only. The NIH ChestX-ray14 dataset is subject to the [NIH Clinical Centre Data Use Agreement](https://nihcc.app.box.com/v/ChestXray-NIHCC). This codebase may be used for academic and research purposes with appropriate citation.

---

<div align="center">

**Chenduluru Siva** | MSc Computing | Coventry University | 2025–26

[GitHub](https://github.com/15758996) • [LinkedIn](https://linkedin.com/in/chenduluru-siva-a16189215)

*7151CEM Computing Individual Research Project*

</div>
