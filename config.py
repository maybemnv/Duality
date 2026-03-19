"""
Configuration file for Duality AI Offroad Segmentation Challenge
PRD Version: SegFormer-B2 | Hardware: RTX 3070 Laptop 8GB VRAM

Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda
"""

import os
import torch

# ============================================================================
# Model Configuration
# ============================================================================

# Primary model (only model supported per PRD)
PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_ARCHITECTURE = "SegFormer-B2"

# Model properties
MODEL_PARAMETERS = "25M"  # Approximate parameter count

# ============================================================================
# Class Mappings (PRD-specified)
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

# Class names for visualization and reporting (PRD order)
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

# Color palette for visualization (RGB) - PRD-specified colors
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
# Hardware Configuration (RTX 3070 Laptop 8GB VRAM)
# ============================================================================

# GPU Specifications
GPU_NAME = "RTX 3070 Laptop"
GPU_VRAM = "8GB GDDR6"
SYSTEM_RAM = "16GB"

# Mixed Precision (AMP) - mandatory on laptop GPU per PRD
# AMP reduces VRAM usage by 30-40% with no accuracy loss
USE_AMP = True  # torch.cuda.amp for mixed precision training

# ============================================================================
# Training Hyperparameters (PRD-specified for RTX 3070 Laptop)
# ============================================================================

# Image dimensions
IMAGE_SIZE = (512, 512)  # Height, Width - matches SegFormer pretrained

# Batch size and workers
BATCH_SIZE = 8           # Safe ceiling for 8GB VRAM with AMP
NUM_WORKERS = 4          # Comfortable for 16GB system RAM
PIN_MEMORY = True        # Faster CPU to GPU transfer

# Training duration
EPOCHS = 60              # PRD-specified
PATIENCE = 10            # Early stopping patience

# Learning rates (differential LR per PRD)
LEARNING_RATE = 6e-5     # For decode head
BACKBONE_LR = 6e-6       # 10x lower for backbone (preserve pretrained features)

# Optimizer
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-4

# Gradient clipping (PRD optimization technique)
GRAD_CLIP_MAX_NORM = 1.0

# Scheduler
SCHEDULER = "CosineAnnealingLR"
T_MAX = 60               # Max iterations for cosine scheduler (matches epochs)
ETA_MIN = 1e-6           # Minimum learning rate

# Loss function
LABEL_SMOOTHING = 0.0    # Standard CrossEntropyLoss (PRD uses weighted CE)

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

# Output paths (PRD file structure)
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
PREDICTIONS_DIR = os.path.join(SCRIPT_DIR, "predictions")

# Model checkpoint paths
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "last_model.pth")

# ============================================================================
# Augmentation Parameters (PRD-specified)
# ============================================================================

# Must-have augmentations (always applied during training)
AUG_RANDOM_RESIZED_CROP = {
    "scale": (0.5, 1.0),  # Simulates different camera distances
}

AUG_HORIZONTAL_FLIP = {
    "p": 0.5,  # Free augmentation
}

AUG_COLOR_JITTER = {
    "brightness": 0.3,  # Desert lighting shifts
    "contrast": 0.3,
    "hue": 0.1,
    "p": 0.8,
}

# Recommended augmentations (add after baseline)
AUG_GAUSSIAN_BLUR = {
    "blur_limit": (3, 7),  # Bridges sim-to-real gap
    "p": 0.3,
}

AUG_RANDOM_GRAYSCALE = {
    "p": 0.1,  # Prevents color overreliance
}

AUG_RANDOM_ROTATION = {
    "degrees": 10,  # Desert terrain angle invariance
    "p": 0.3,
}

AUG_GRID_DISTORTION = {
    "distort_limit": 0.2,  # Mimics lens distortion
    "p": 0.3,
}

# ImageNet normalization (required for pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Test-Time Augmentation (TTA) Parameters (PRD optimization)
# ============================================================================

# TTA adds 2-5 IoU points with no additional training
USE_TTA = True  # Default: enabled

# TTA augmentations to average
TTA_FLIP = True       # Horizontal flip
TTA_SCALES = [0.9, 1.0, 1.1]  # Multi-scale averaging

# ============================================================================
# Device Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Benchmark Targets (PRD-specified)
# ============================================================================

IOU_TARGETS = {
    "baseline": (0.25, 0.35),      # Sample train.py expected
    "segformer_no_aug": (0.45, 0.55),  # SegFormer without augmentation
    "segformer_full": (0.55, 0.70),    # SegFormer with full augmentation
    "with_tta": 0.05,              # TTA adds up to +0.05
}

INFERENCE_TIME_TARGET = 50  # ms per image

# ============================================================================
# Directory Setup
# ============================================================================

def setup_directories():
    """Create necessary output directories per PRD file structure."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "masks_color"), exist_ok=True)
    os.makedirs(os.path.join(PREDICTIONS_DIR, "comparisons"), exist_ok=True)
    print(f"Created output directories:")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Training logs: {RUNS_DIR}")
    print(f"  - Predictions: {PREDICTIONS_DIR}")


# ============================================================================
# Hardware Validation
# ============================================================================

def validate_hardware():
    """Validate hardware configuration and print specs."""
    print("\n" + "=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(f"GPU: {GPU_NAME}")
    print(f"VRAM: {GPU_VRAM}")
    print(f"System RAM: {SYSTEM_RAM}")
    print(f"Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Available: Yes")
    else:
        print("CUDA Available: No (using CPU)")
    
    print("=" * 60)
    
    # VRAM warning if < 8GB
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram < 8:
            print(f"\nWARNING: VRAM ({total_vram:.1f}GB) is less than recommended 8GB.")
            print("Consider reducing BATCH_SIZE to 4 in config.py")


if __name__ == "__main__":
    setup_directories()
    validate_hardware()
