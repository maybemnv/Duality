# Duality AI — Offroad Semantic Scene Segmentation

A semantic segmentation solution for the Duality AI Offroad Autonomy Segmentation Challenge. This project implements a **SegFormer-B2** model fine-tuned on synthetic desert imagery, with domain generalization techniques to handle the train-to-test environment shift.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Hardware Requirements](#hardware-requirements)

---

## Overview

**Competition:** Duality AI Offroad Autonomy Segmentation Challenge  
**Primary Metric:** Mean IoU (Intersection over Union)  
**Approach:** Fine-tuning pretrained SegFormer-B2 with weighted loss and data augmentation

### Key Features

- **SegFormer-B2 backbone** - Transformer-based architecture for superior generalization
- **Class-weighted CrossEntropyLoss** - Handles class imbalance (Logs, Flowers, Ground Clutter)
- **Comprehensive augmentation** - ColorJitter, blur, rotation for sim-to-real robustness
- **Test-time augmentation (TTA)** - Optional flip augmentation for improved accuracy
- **Differential learning rates** - Lower LR for backbone, higher for decode head

### Segmentation Classes

| ID | Class Name | Category |
|----|------------|----------|
| 0 | Trees | Vegetation |
| 1 | Lush Bushes | Vegetation |
| 2 | Dry Grass | Ground Cover |
| 3 | Dry Bushes | Vegetation |
| 4 | Ground Clutter | Ground Cover |
| 5 | Flowers | Vegetation |
| 6 | Logs | Object |
| 7 | Rocks | Object |
| 8 | Landscape | Ground |
| 9 | Sky | Environment |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Falcon platform
# Place in Data/Offroad_Segmentation_Training_Dataset/

# 3. Train the model
python train_segmentation.py --epochs 50 --batch_size 8

# 4. Run inference on test images
python test_segmentation.py --use_tta

# 5. Check results
# - Checkpoints: checkpoints/best_model.pth
# - Predictions: predictions/masks_color/
# - Metrics: runs/evaluation_metrics.txt
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended for GPU training)
- 6+ GB VRAM (for SegFormer-B2)

### Step-by-Step

#### Option 1: Using pip

```bash
# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using uv (recommended)

```bash
# Install uv if not already installed
pip install uv

# Sync environment
uv sync
```

#### Option 3: Using conda

```bash
# Create environment
conda create -n duality python=3.10
conda activate duality

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers segmentation-models-pytorch albumentations opencv-contrib-python tqdm matplotlib
```

---

## Dataset Structure

Place the downloaded dataset in the `Data/` directory:

```
Data/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   │   ├── image_001.png
│   │   │   └── ...
│   │   └── Segmentation/
│   │       ├── image_001.png
│   │       └── ...
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    └── Color_Images/
        ├── test_001.png
        └── ...
```

> **Important:** The `testImages` folder must NOT be used for training. It is strictly for final evaluation.

---

## Usage

### Training

#### Basic Training

```bash
python train_segmentation.py
```

#### With Custom Hyperparameters

```bash
python train_segmentation.py \
    --model_type segformer_b2 \
    --epochs 50 \
    --batch_size 8 \
    --lr 6e-5 \
    --backbone_lr 6e-6
```

#### Using Alternative Models

```bash
# DeepLabV3+ (lower VRAM, ~4GB)
python train_segmentation.py --model_type deeplabv3

# UNet + ResNet34 (fastest, ~3GB VRAM)
python train_segmentation.py --model_type unet
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `segformer_b2` | Model architecture |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `8` | Batch size (adjust for VRAM) |
| `--lr` | `6e-5` | Learning rate for decode head |
| `--backbone_lr` | `lr/10` | Learning rate for backbone |
| `--device` | `auto` | Device (cuda/cpu) |

### Inference

#### Basic Inference

```bash
python test_segmentation.py
```

#### With Custom Options

```bash
python test_segmentation.py \
    --model_path checkpoints/best_model.pth \
    --data_dir Data/Offroad_Segmentation_testImages \
    --output_dir predictions \
    --use_tta \
    --batch_size 8 \
    --num_vis 10
```

#### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `checkpoints/best_model.pth` | Path to model weights |
| `--data_dir` | `TEST_DIR` | Path to test images |
| `--output_dir` | `predictions` | Output directory |
| `--use_tta` | `True` | Use test-time augmentation |
| `--no_tta` | - | Disable TTA |
| `--batch_size` | `8` | Batch size |
| `--num_vis` | `5` | Number of visualization samples |

---

## Model Architecture

### SegFormer-B2 (Recommended)

```
┌─────────────────────────────────────────────┐
│           Input Image (512×512)             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│     Mix Transformer (MiT) Backbone          │
│     - Hierarchical feature extraction       │
│     - Global context attention              │
│     - Pretrained on ImageNet-1K + ADE20K    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         MLP Decode Head                     │
│     - Feature fusion across scales          │
│     - Dense prediction (10 classes)         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│    Output: Segmentation Mask (512×512)      │
│    - 10 class channels                       │
│    - Softmax → argmax for final prediction   │
└─────────────────────────────────────────────┘
```

### Why SegFormer?

| Factor | SegFormer-B2 | DeepLabV3+ | UNet |
|--------|--------------|------------|------|
| Mean IoU ceiling | **Highest** | Medium | Lower |
| Domain generalization | **Excellent** | Good | Fair |
| VRAM requirement | ~6 GB | ~4 GB | ~3 GB |
| Training time | ~2-3 hrs | ~1-2 hrs | ~1 hr |
| Inference speed | ~40ms | ~30ms | ~20ms |

---

## Configuration

All hyperparameters and settings are in `config.py`:

```python
# Learning rates
LEARNING_RATE = 6e-5        # Decode head
BACKBONE_LR = 6e-6          # Backbone (10x lower)

# Training
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10               # Early stopping

# Image size
IMAGE_SIZE = (512, 512)

# Augmentation
AUG_COLOR_JITTER = {"brightness": 0.3, "contrast": 0.3, "hue": 0.1}
AUG_HORIZONTAL_FLIP = {"p": 0.5}
AUG_GAUSSIAN_BLUR = {"blur_limit": (3, 7), "p": 0.3}
```

---

## Project Structure

```
Duality/
├── config.py               # Configuration and hyperparameters
├── dataset.py              # Dataset class and transforms
├── model.py                # Model definitions
├── utils.py                # Metrics and visualization
├── train_segmentation.py   # Training script
├── test_segmentation.py    # Inference script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
│
├── Data/                   # Dataset (download separately)
│   ├── Offroad_Segmentation_Training_Dataset/
│   └── Offroad_Segmentation_testImages/
│
├── checkpoints/            # Model checkpoints (created on first run)
│   ├── best_model.pth
│   ├── last_model.pth
│   └── final_model.pth
│
├── runs/                   # Training logs (created on first run)
│   ├── training_curves.png
│   ├── iou_curves.png
│   ├── dice_curves.png
│   ├── all_metrics_curves.png
│   └── evaluation_metrics.txt
│
└── predictions/            # Inference outputs (created on first run)
    ├── masks/              # Raw prediction masks
    ├── masks_color/        # Colored visualizations
    ├── comparisons/        # Side-by-side comparisons
    └── inference_summary.txt
```

---

## Expected Results

### Benchmark Targets

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Mean IoU | 0.25-0.35 | 0.50-0.65 | > 0.70 |
| Inference speed | - | < 100ms | < 50ms |

### Training Timeline

| Phase | Duration | Expected IoU |
|-------|----------|--------------|
| Epoch 1-10 | ~30 min | 0.20-0.35 |
| Epoch 11-30 | ~1 hr | 0.35-0.55 |
| Epoch 31-50 | ~1 hr | 0.55-0.70+ |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_segmentation.py --batch_size 4

# Or use a smaller model
python train_segmentation.py --model_type deeplabv3
```

### Slow Training

```bash
# Enable multi-worker data loading (Linux/Mac)
# Edit dataset.py: num_workers=4

# Or reduce image resolution in config.py
IMAGE_SIZE = (384, 384)
```

### Class Imbalance Issues

The weighted CrossEntropyLoss is automatically computed from training data. If rare classes (Logs, Flowers) still have 0 IoU:

1. Check that class weights are being printed during training
2. Verify mask remapping is correct (visualize a few masks)
3. Consider oversampling rare-class images in `dataset.py`

### Loss Not Decreasing

- Verify learning rate is not too high (try `--lr 1e-5`)
- Check that masks are correctly remapped to [0, N-1]
- Ensure data augmentation is not too aggressive

---

## Hardware Requirements

### Minimum

- CPU: 4+ cores
- RAM: 16 GB
- GPU: 4 GB VRAM (use `--model_type deeplabv3` or `unet`)
- Storage: 10 GB free

### Recommended

- CPU: 8+ cores
- RAM: 32 GB
- GPU: 8+ GB VRAM (RTX 3060 or better)
- Storage: 20 GB free (SSD preferred)

### Training Time Estimates

| GPU | SegFormer-B2 | DeepLabV3+ | UNet |
|-----|--------------|------------|------|
| RTX 4090 | ~1 hr | ~45 min | ~30 min |
| RTX 3080 | ~1.5 hr | ~1 hr | ~45 min |
| RTX 3060 | ~2.5 hr | ~1.5 hr | ~1 hr |
| CPU only | ~12 hr | ~8 hr | ~5 hr |

---

## License

This project is for the Duality AI Hackathon competition.

## Team

- Syed Muhammad Maaz (@Maazsyedm)
- Rebekah Bogdanoff (@rebekah-bogdanoff)
- Evan Goldman (@egold010)

---

*For questions or issues, please open a GitHub issue.*
