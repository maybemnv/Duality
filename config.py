"""
Configuration file for Duality AI Offroad Segmentation Challenge
Contains class mappings, hyperparameters, and constants
"""

import os

# ============================================================================
# Class Mappings
# ============================================================================

# Original class IDs from dataset -> Remapped to [0, N-1] for PyTorch CrossEntropyLoss
ORIGINAL_ID_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

# Reverse mapping for visualization and output
ID_TO_CLASS = {v: k for k, v in ORIGINAL_ID_MAP.items()}

# Class names for visualization and reporting
CLASS_NAMES = [
    'Trees',          # 0
    'Lush Bushes',    # 1
    'Dry Grass',      # 2
    'Dry Bushes',     # 3
    'Ground Clutter', # 4
    'Flowers',        # 5
    'Logs',           # 6
    'Rocks',          # 7
    'Landscape',      # 8
    'Sky',            # 9
]

NUM_CLASSES = len(CLASS_NAMES)

# Color palette for visualization (RGB)
COLOR_PALETTE = [
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [255, 105, 180],  # Flowers - hot pink
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
]

# ============================================================================
# Hyperparameters (SegFormer-B2)
# ============================================================================

# Learning rates
LEARNING_RATE = 6e-5        # For decode head
BACKBONE_LR = 6e-6          # 10x lower for backbone (differential LR)

# Training
BATCH_SIZE = 8              # Adjust based on VRAM availability
EPOCHS = 50                 # With early stopping
PATIENCE = 10               # Early stopping patience

# Image dimensions
IMAGE_SIZE = (512, 512)     # Height, Width

# Optimizer
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-4

# Scheduler
SCHEDULER = "CosineAnnealingLR"
T_MAX = 50                  # Max iterations for cosine scheduler

# Loss
LABEL_SMOOTHING = 0.1       # Label smoothing for CrossEntropyLoss

# ============================================================================
# Paths
# ============================================================================

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths (relative to project root)
DATA_ROOT = os.path.join(SCRIPT_DIR, "Data")
TRAIN_DIR = os.path.join(DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR = os.path.join(DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "val")
TEST_DIR = os.path.join(DATA_ROOT, "Offroad_Segmentation_testImages")

# Output paths
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
PREDICTIONS_DIR = os.path.join(SCRIPT_DIR, "predictions")

# Model checkpoint
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "last_model.pth")

# Pretrained model
PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

# ============================================================================
# Augmentation Parameters
# ============================================================================

# Must-have augmentations (always applied)
AUG_COLOR_JITTER = {
    "brightness": 0.3,
    "contrast": 0.3,
    "hue": 0.1,
    "p": 0.8
}

AUG_HORIZONTAL_FLIP = {"p": 0.5}
AUG_RANDOM_RESIZED_CROP = {"scale": (0.5, 1.0)}

# Recommended augmentations (add after baseline)
AUG_GAUSSIAN_BLUR = {"blur_limit": (3, 7), "p": 0.3}
AUG_RANDOM_GRAYSCALE = {"p": 0.1}
AUG_RANDOM_ROTATION = {"degrees": 10, "p": 0.3}

# ImageNet normalization (required for pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Device
# ============================================================================

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Create directories
# ============================================================================

def setup_directories():
    """Create necessary output directories."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "masks_color"), exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "comparisons"), exist_ok=True)
