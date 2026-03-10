"""
Central configuration for the Intelligent Multi-Stage Chest X-Ray Analysis System
Author: Chenduluru Siva | 7151CEM
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

METADATA_CSV = os.path.join(RAW_DIR, "Data_Entry_2017_v2020.csv")
IMAGES_DIR = os.path.join(RAW_DIR, "images")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASE_LABELS)
NO_FINDING_LABEL = "No Finding"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# IMAGE QUALITY ASSESSMENT
# ─────────────────────────────────────────────
QA_BRIGHTNESS_MIN  = 30
QA_BRIGHTNESS_MAX  = 220
QA_CONTRAST_MIN    = 20
QA_BLUR_THRESHOLD  = 100.0
QA_BLACK_RATIO_MAX = 0.30

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
BACKBONE     = "densenet121"
PRETRAINED   = True
DROPOUT_RATE = 0.5

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
LR_STEP_SIZE   = 10
LR_GAMMA       = 0.1
NUM_WORKERS    = 4
PIN_MEMORY     = True
USE_WEIGHTED_LOSS         = True
EARLY_STOPPING_PATIENCE   = 5

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
CLASSIFICATION_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
GRADCAM_LAYER = "features.denseblock4"

# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
REPORT_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "reports")
CONFIDENCE_HIGH   = 0.70
CONFIDENCE_MEDIUM = 0.50

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
SEED = 42