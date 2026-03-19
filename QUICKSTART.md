# Duality AI Segmentation - Quick Start Guide

## Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv
uv sync
```

## Training

```bash
# Default training (SegFormer-B2, 50 epochs)
python train_segmentation.py

# Quick test (fewer epochs, smaller batch)
python train_segmentation.py --epochs 10 --batch_size 4

# With custom settings
python train_segmentation.py --epochs 50 --batch_size 8 --lr 6e-5
```

## Inference

```bash
# Run on test images (with TTA)
python test_segmentation.py

# Without TTA (faster)
python test_segmentation.py --no_tta
```

## Check Results

- **Model checkpoints**: `checkpoints/best_model.pth`
- **Training metrics**: `runs/evaluation_metrics.txt`
- **Training plots**: `runs/training_curves.png`, `runs/iou_curves.png`
- **Predictions**: `predictions/masks_color/`

## Dataset Structure

Ensure your dataset is organized as:

```
Data/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    └── Color_Images/
```

## Troubleshooting

**Out of Memory?**
```bash
python train_segmentation.py --batch_size 4
# or
python train_segmentation.py --model_type deeplabv3
```

**Slow training?**
- Use GPU (CUDA)
- Reduce batch size
- Use fewer epochs for testing

## Files Overview

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters and settings |
| `dataset.py` | Data loading and augmentations |
| `model.py` | Model definitions (SegFormer, DeepLabV3+, UNet) |
| `utils.py` | Metrics, visualization, logging |
| `train_segmentation.py` | Training script |
| `test_segmentation.py` | Inference script |
