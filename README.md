# Duality AI — Offroad Semantic Scene Segmentation

**PRD Version:** SegFormer-B2 | **Hardware:** RTX 3070 Laptop 8GB VRAM

A semantic segmentation solution for the Duality AI Offroad Autonomy Segmentation Challenge using **SegFormer-B2** fine-tuned on synthetic desert imagery with domain generalization techniques.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Hardware Requirements](#hardware-requirements)
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
- [Report Structure](#report-structure)
- [Submission Checklist](#submission-checklist)

---

## Overview

**Competition:** Duality AI Offroad Autonomy Segmentation Challenge  
**Primary Metric:** Mean IoU (80 pts) + Report Clarity (20 pts) = 100 pts  
**Model:** SegFormer-B2 (nvidia/segformer-b2-finetuned-ade-512-512)  
**Core Challenge:** Domain shift — train and test environments are different desert locations

### Why SegFormer-B2?

| Factor | Rationale |
|--------|-----------|
| **Dense prediction** | 512×512 = 262,144 pixel outputs — only encoder-decoder vision architectures work |
| **Pretrained weights** | Converges in hours vs days; ADE20K includes outdoor/vegetation/terrain |
| **Transformer attention** | Self-attention handles domain shift better than CNN local patterns |
| **Hardware fit** | ~5.5-6GB VRAM at batch 8 with AMP on RTX 3070 Laptop |

### Segmentation Classes

| ID | Class Name | Difficulty |
|----|------------|------------|
| 0 | Trees | Easy |
| 1 | Lush Bushes | Medium |
| 2 | Dry Grass | Hard |
| 3 | Dry Bushes | Hard |
| 4 | Ground Clutter | Hard |
| 5 | Flowers | Hard |
| 6 | Logs | Hardest |
| 7 | Rocks | Hard |
| 8 | Landscape | Easy (pixels) |
| 9 | Sky | Easiest |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Activate conda environment (if using)
conda activate EDU

# 3. Train the model (60 epochs, ~4-6 hours on RTX 3070)
python train_segmentation.py

# 4. Run inference with TTA
python test_segmentation.py --use_tta

# 5. Check results
# - Checkpoints: checkpoints/best_model.pth
# - Metrics: runs/evaluation_metrics.txt
# - Predictions: predictions/masks_color/
```

---

## Hardware Requirements

### Tested Configuration

| Component | Spec |
|-----------|------|
| **GPU** | RTX 3070 Laptop |
| **VRAM** | 8GB GDDR6 |
| **RAM** | 16GB |
| **OS** | Windows (Anaconda) |

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB | 8GB+ |
| System RAM | 8GB | 16GB |
| Storage | 10GB free | 20GB SSD |

### Training Time Estimates

| GPU | Time per Epoch | Total (60 ep) |
|-----|---------------|---------------|
| RTX 3070 Laptop | ~4-6 min | ~4-6 hours |
| RTX 3080 | ~3-4 min | ~3-4 hours |
| RTX 4090 | ~1-2 min | ~1-2 hours |
| CPU only | ~20-30 min | ~20-30 hours |

### Laptop Throttling Checklist

Before every training session:

- [ ] Plug into power (never train on battery)
- [ ] Set Windows power plan to **High Performance**
- [ ] Use a cooling pad or laptop stand
- [ ] Monitor with `nvidia-smi dmon` — keep GPU temp < 85°C
- [ ] Ensure GPU utilization > 90%

---

## Installation

### Option 1: Using pip

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda (PRD recommended)

```bash
# Create environment
conda create -n duality python=3.10
conda activate duality

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers segmentation-models-pytorch albumentations opencv-contrib-python tqdm matplotlib accelerate
```

### Option 3: Using uv

```bash
uv sync
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

> ⚠️ **CRITICAL:** The `testImages` folder is **strictly off-limits for training**. Using it for training results in immediate disqualification.

### Class ID Remapping

The dataset uses non-sequential class IDs (100, 200... 10000). PyTorch CrossEntropyLoss requires [0, N-1]. Remapping is **mandatory**:

| Original | Remapped | Class |
|----------|----------|-------|
| 100 | 0 | Trees |
| 200 | 1 | Lush Bushes |
| 300 | 2 | Dry Grass |
| 500 | 3 | Dry Bushes |
| 550 | 4 | Ground Clutter |
| 600 | 5 | Flowers |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape |
| 10000 | 9 | Sky |

**Verification:** After applying remapping, visualize 5 random masks and confirm labels match boundaries before training.

---

## Usage

### Training

#### Basic Training

```bash
python train_segmentation.py
```

#### With Custom Settings

```bash
python train_segmentation.py \
    --epochs 60 \
    --batch_size 8 \
    --lr 6e-5 \
    --backbone_lr 6e-6
```

#### Training Features (PRD-specified)

- ✅ **Mixed Precision (AMP)** — Reduces VRAM by 30-40%
- ✅ **Weighted CrossEntropyLoss** — Handles class imbalance
- ✅ **Differential Learning Rates** — Backbone: 6e-6, Head: 6e-5
- ✅ **Gradient Clipping** — Max norm 1.0
- ✅ **Cosine Annealing LR** — Smooth decay to 1e-6
- ✅ **Early Stopping** — Patience 10 epochs

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 60 | Number of training epochs |
| `--batch_size` | 8 | Batch size (reduce to 4 if OOM) |
| `--lr` | 6e-5 | Learning rate for decode head |
| `--backbone_lr` | 6e-6 | Learning rate for backbone |
| `--amp` | Enabled | Mixed precision training |
| `--no_amp` | - | Disable mixed precision |

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

#### Inference Features (PRD-specified)

- ✅ **Test-Time Augmentation (TTA)** — Adds +2-5 IoU points
- ✅ **Multi-scale averaging** — 0.9x, 1.0x, 1.1x
- ✅ **Horizontal flip averaging** — Flip prediction back
- ✅ **Target inference time** — < 50ms per image

#### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | checkpoints/best_model.pth | Path to model weights |
| `--data_dir` | TEST_DIR | Path to test images |
| `--output_dir` | predictions | Output directory |
| `--use_tta` | True | Use test-time augmentation |
| `--no_tta` | - | Disable TTA |
| `--batch_size` | 8 | Batch size |
| `--num_vis` | 5 | Number of visualization samples |

---

## Model Architecture

### SegFormer-B2

```
┌─────────────────────────────────────────────┐
│           Input Image (512×512)             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│     Mix Transformer (MiT-B2) Backbone       │
│     - Hierarchical feature extraction       │
│     - Global self-attention                 │
│     - Pretrained: ImageNet-1K + ADE20K      │
│     - Parameters: ~25M                      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         MLP Decode Head                     │
│     - Feature fusion across scales          │
│     - Dense prediction (10 classes)         │
│     - Swapped from ADE20K (150 classes)     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│    Output: Segmentation Mask (512×512)      │
│    - 10 class channels (logits)             │
│    - Softmax → argmax for final prediction  │
└─────────────────────────────────────────────┘
```

### Training Hyperparameters (PRD-specified)

| Parameter | Value |
|-----------|-------|
| Model | nvidia/segformer-b2-finetuned-ade-512-512 |
| Image Size | 512 × 512 |
| Batch Size | 8 |
| Epochs | 60 |
| LR Head | 6e-5 |
| LR Backbone | 6e-6 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| AMP | Enabled |
| Grad Clip | 1.0 |

---

## Configuration

All hyperparameters are in `config.py`:

```python
# Model
PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

# Hardware
GPU_NAME = "RTX 3070 Laptop"
GPU_VRAM = "8GB GDDR6"
USE_AMP = True  # Mandatory on laptop GPU

# Training
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 6e-5       # Decode head
BACKBONE_LR = 6e-6         # Backbone (10x lower)

# Augmentation (PRD-specified)
AUG_COLOR_JITTER = {"brightness": 0.3, "contrast": 0.3, "hue": 0.1}
AUG_GAUSSIAN_BLUR = {"blur_limit": (3, 7), "p": 0.3}
AUG_GRID_DISTORTION = {"distort_limit": 0.2, "p": 0.3}

# TTA (adds +2-5 IoU)
USE_TTA = True
TTA_SCALES = [0.9, 1.0, 1.1]
```

---

## Project Structure

```
Duality/
├── config.py               # Configuration and hyperparameters
├── dataset.py              # Dataset class with ID remapping
├── model.py                # SegFormer-B2 model setup
├── augmentations.py        # Training and TTA augmentations
├── utils.py                # Metrics and visualization
├── train_segmentation.py   # Training script with AMP
├── test_segmentation.py    # Inference script with TTA
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── README.md               # This file
├── QUICKSTART.md           # Quick reference
│
├── Data/                   # Dataset (download separately)
│   ├── Offroad_Segmentation_Training_Dataset/
│   └── Offroad_Segmentation_testImages/
│
├── checkpoints/            # Model checkpoints
│   ├── best_model.pth      # Best validation IoU
│   ├── last_model.pth      # Last saved (every 5 epochs)
│   └── final_model.pth     # Final model
│
├── runs/                   # Training logs
│   ├── training_curves.png
│   ├── iou_curves.png
│   ├── dice_curves.png
│   ├── all_metrics_curves.png
│   └── evaluation_metrics.txt
│
└── predictions/            # Inference outputs
    ├── masks/              # Raw prediction masks (0-9)
    ├── masks_color/        # Colored visualizations
    ├── comparisons/        # Side-by-side comparisons
    ├── inference_summary.txt
    └── per_class_iou.png
```

---

## Expected Results

### IoU Benchmark Targets

| Stage | Expected Mean IoU |
|-------|-------------------|
| Baseline (sample train.py) | 0.25 – 0.35 |
| SegFormer-B2, no augmentation | 0.45 – 0.55 |
| SegFormer-B2, full augmentation + weighted loss | 0.55 – 0.70 |
| With TTA | +0.02 – 0.05 gain |

### Training Timeline

| Phase | Duration | Expected IoU |
|-------|----------|--------------|
| Epoch 1-10 | ~45 min | 0.20-0.40 |
| Epoch 11-30 | ~1.5 hr | 0.40-0.55 |
| Epoch 31-60 | ~2-3 hr | 0.55-0.70+ |

### Diagnosing Training Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Val loss increases, train loss decreases | Overfitting | Add more augmentation, reduce LR |
| Loss plateaus too high | Underfitting | Increase epochs, unfreeze backbone |
| Specific class IoU = 0 | Remapping bug or imbalance | Verify remapping, increase class weight |
| GPU utilization < 80% | DataLoader bottleneck | Increase num_workers, enable pin_memory |
| CUDA OOM | Batch too large | Reduce batch_size to 4, verify AMP is on |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_segmentation.py --batch_size 4

# Verify AMP is enabled (should be by default)
python train_segmentation.py --amp
```

### Slow Training

```bash
# Enable persistent workers (already enabled in config)
# Reduce num_workers if RAM is constrained
# Edit config.py: NUM_WORKERS = 2
```

### Class Imbalance Issues

The weighted CrossEntropyLoss is automatically computed. If rare classes (Logs, Flowers) still have 0 IoU:

1. Check that class weights are printed during training
2. Verify mask remapping with `dataset.verify_mask_remap(TRAIN_DIR)`
3. Consider oversampling rare-class images

### Loss Not Decreasing

- Verify learning rate is not too high (try `--lr 1e-5`)
- Check that masks are correctly remapped to [0, N-1]
- Ensure data augmentation is not too aggressive

---

## Report Structure (8 pages max)

The report is worth **20 points**. Document your process clearly.

### Page 1: Title
- Team name
- Project name
- One-line tagline

### Page 2: Methodology
- Model choice rationale (why SegFormer-B2)
- Architecture overview
- Training setup (LR, batch size, epochs)
- Augmentation pipeline

### Pages 3-4: Results & Metrics
- **Mean IoU** (prominently displayed)
- **Per-class IoU table** (all 10 classes)
- **Training/Validation loss curves**
- Before vs after comparison
- Sample segmentation outputs

### Pages 5-6: Challenges & Solutions

Use this format for each challenge:

```
Task:    [what you were trying to do]
Issue:   [what went wrong — include prediction image]
Why:     [root cause analysis]
Fix:     [what you changed]
Result:  [IoU before → IoU after]
```

**Cover minimum:**
- Class imbalance (Logs, Flowers, Ground Clutter)
- Domain shift behavior
- Misclassification patterns (Dry Bushes vs Rocks, Dry Grass vs Landscape)

### Page 7: Conclusion & Future Work
- Final mean IoU
- 3-5 things you would do with more time:
  - Better TTA configurations
  - Checkpoint ensembling
  - Copy-paste augmentation for rare classes
  - Domain adaptation techniques

---

## Submission Checklist

### Code Package

- [ ] `train_segmentation.py` runs without errors
- [ ] `test_segmentation.py` produces predictions on testImages
- [ ] `best_model.pth` checkpoint included
- [ ] `README.md` covers setup, training, inference
- [ ] `requirements.txt` included
- [ ] Loss graphs saved in `runs/`
- [ ] Prediction images saved in `predictions/`

### GitHub

- [ ] Repository is **private**
- [ ] Collaborators added:
  - [ ] `Maazsyedm`
  - [ ] `rebekah-bogdanoff`
  - [ ] `egold010`
- [ ] Everything zipped and uploaded

### Submission Form

- [ ] Final mean IoU score reported
- [ ] GitHub repository link provided

### Report

- [ ] PDF or DOCX format
- [ ] 8 pages maximum
- [ ] Per-class IoU table included
- [ ] Loss curves included
- [ ] At least 2 failure case images with analysis
- [ ] Methodology section covers model choice rationale

---

## Disqualification Risks

| Risk | Consequence | Prevention |
|------|-------------|------------|
| testImages used in training | **Immediate disqualification** | testImages never imported in train_segmentation.py |
| External image data used | Disqualification | Only dataset/ folder used |
| GitHub repo public before deadline | Integrity issue | Keep repo private until submission confirmed |

---

## Links

| Purpose | URL |
|---------|-----|
| Create Falcon account | https://falcon.duality.ai/auth/sign-up |
| Download dataset | https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert |
| Discord support | https://discord.com/invite/dualityfalconcommunity |
| SegFormer model | https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512 |

---

## Team

- Syed Muhammad Maaz (@Maazsyedm)
- Rebekah Bogdanoff (@rebekah-bogdanoff)
- Evan Goldman (@egold010)

---

*Model: SegFormer-B2 | Hardware: RTX 3070 Laptop 8GB | Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda*
