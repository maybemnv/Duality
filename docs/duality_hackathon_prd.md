# Duality AI — Offroad Semantic Scene Segmentation
## Product Requirements Document (PRD) · v1.0

---

## Table of Contents

1. [Overview](#1-overview)
2. [Objectives](#2-objectives)
3. [Dataset & Classes](#3-dataset--classes)
4. [Model Strategy & Recommendation](#4-model-strategy--recommendation)
5. [Technical Architecture](#5-technical-architecture)
6. [Execution Pipeline](#6-execution-pipeline)
7. [Augmentation Strategy](#7-augmentation-strategy)
8. [Evaluation & Metrics](#8-evaluation--metrics)
9. [Key Deliverables](#9-key-deliverables)
10. [Report Structure](#10-report-structure)
11. [Submission Checklist](#11-submission-checklist)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Glossary](#13-glossary)

---

## 1. Overview

**Competition:** Duality AI Offroad Autonomy Segmentation Challenge
**Platform:** Falcon / FalconCloud (Duality AI digital twin simulation)
**Primary Metric:** IoU Score (80 pts) + Report Clarity (20 pts)
**Total Score:** 100 points

### What We Are Building

A **semantic segmentation model** trained on synthetic desert imagery that can accurately classify every pixel in a scene across 10 environmental classes — and then **generalize** to a completely different desert location it has never seen during training.

### Why This Is Hard

The core challenge is **domain shift**: the train environment and the test environment are different desert locations. Lighting conditions, plant density, terrain texture, and vegetation distribution all change. A model that memorizes pixel patterns from the training environment will fail. We need a model that learns *semantic meaning* (what is a bush vs a rock) not location-specific appearance.

> **Key insight:** This is why Duality AI uses synthetic data — they can precisely control environment variables. Our job is to build a model robust enough to transfer across those variables.

---

## 2. Objectives

| Priority | Objective |
|----------|-----------|
| P0 | Train a semantic segmentation model on the provided synthetic dataset |
| P0 | Maximize mean IoU score on unseen testImages (different desert location) |
| P1 | Implement data augmentation to improve generalization |
| P1 | Handle class imbalance with weighted loss functions |
| P2 | Document methodology clearly for the 20-pt report score |
| P2 | Achieve inference speed < 50ms per image |

### Constraints

- **Must train exclusively on the provided dataset** — no external image data
- **testImages folder is strictly off-limits for training** — instant disqualification
- Custom models and training scripts are allowed
- Submission via private GitHub repository

---

## 3. Dataset & Classes

### Dataset Structure

```
dataset/
├── train/          # RGB color images + segmentation masks (paired)
├── val/            # RGB color images + segmentation masks (paired)
├── testImages/     # RGB only — NO masks — evaluation set (DO NOT use for training)
├── train.py        # Sample training script
├── test.py         # Sample inference script
└── ENV_SETUP/      # setup_env.bat (Windows) / create setup_env.sh for Mac/Linux
```

### Segmentation Classes

| Class ID | Class Name | Category | Expected Difficulty |
|----------|------------|----------|---------------------|
| 100 | Trees | Vegetation | Easy — tall distinctive silhouette |
| 200 | Lush Bushes | Vegetation | Medium — confused with Dry Bushes |
| 300 | Dry Grass | Ground Cover | Hard — blends with Landscape |
| 500 | Dry Bushes | Vegetation | Hard — similar color to Rocks |
| 550 | Ground Clutter | Ground Cover | Hard — low pixel count, rare class |
| 600 | Flowers | Vegetation | Hard — rare class, imbalance risk |
| 700 | Logs | Object | Hardest — occlusion, very rare |
| 800 | Rocks | Object | Hard — texture similar to ground |
| 7100 | Landscape | Ground | Easy pixel count but causes imbalance |
| 10000 | Sky | Environment | Easiest — clear distinct boundary |

### Critical Preprocessing: Class ID Remapping

> ⚠️ **This step is mandatory.** PyTorch `CrossEntropyLoss` requires class labels in range `[0, N-1]`. The provided IDs are non-sequential (100, 200... 10000) and will cause training to fail or produce garbage outputs if not remapped.

```python
# Class ID remapping — apply this during dataset loading
ID_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9    # Sky
}

def remap_mask(mask):
    remapped = torch.zeros_like(mask)
    for orig_id, new_id in ID_MAP.items():
        remapped[mask == orig_id] = new_id
    return remapped
```

---

## 4. Model Strategy & Recommendation

### TL;DR

**Use a pretrained semantic segmentation model and fine-tune it.** Do not train from scratch. Do not use an LLM or SLM.

### Why NOT an LLM/SLM?

LLMs and SLMs (Llama, Gemma, Phi, Mistral, etc.) are language models — they output text tokens, not pixel-level masks. Semantic segmentation requires a *dense prediction* output where every single pixel in a 512×512 image gets a class label. That is architecturally incompatible with how language models work.

Even vision-language models like LLaVA or Qwen-VL output text descriptions, not segmentation masks. You would need to bolt on a full decode head, at which point you're building a segmentation model anyway — just with unnecessary overhead.

**Exception:** SAM (Segment Anything Model) from Meta is vision-native, but it is prompt-based and not class-aware. Not suitable for 10-class semantic segmentation.

### Why Pretrained Weights?

- ImageNet pretrained backbones already understand low-level features (edges, textures, shapes)
- ADE20K pretrained segmentation models have seen outdoor scenes including vegetation and terrain
- Fine-tuning converges in hours, not days
- Dramatically better generalization than training from scratch on a small synthetic dataset

---

### Model Comparison

#### Option 1: SegFormer-B2 ✅ RECOMMENDED

| Property | Value |
|----------|-------|
| Architecture | Mix Transformer (MiT) backbone + lightweight MLP decode head |
| Pretrained on | ImageNet-1K + ADE20K semantic segmentation |
| Source | HuggingFace `transformers` |
| Model string | `nvidia/segformer-b2-finetuned-ade-512-512` |
| VRAM needed | ~6 GB |
| Training time | ~2–3 hrs on RTX 3060 |

**Pros:**
- Excellent texture and long-range context understanding (transformers capture global relationships)
- MLP decode head is simple to swap for 10 classes
- Best generalization across domain shifts in its class
- HuggingFace makes fine-tuning 10–15 lines of code

**Cons:**
- Needs ~6 GB VRAM for B2 (use B0/B1 if GPU-constrained)
- Slightly slower inference than CNN-based models

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=10,
    ignore_mismatched_sizes=True  # swaps the classification head
)
```

---

#### Option 2: DeepLabV3+ with ResNet50

| Property | Value |
|----------|-------|
| Architecture | ASPP + encoder-decoder with ResNet backbone |
| Pretrained on | COCO + VOC |
| Source | `torchvision.models.segmentation` |
| VRAM needed | ~4 GB |

**Pros:**
- Battle-tested, extremely well documented
- ASPP module captures multi-scale context (great for varying object sizes in desert scenes)
- Easy to load from torchvision

**Cons:**
- CNN struggles with long-range dependencies
- Less generalization capacity than transformers

**Use this if:** SegFormer is too heavy for your GPU or you want a faster baseline.

---

#### Option 3: UNet + ResNet34 (Prototype / Low-Resource)

| Property | Value |
|----------|-------|
| Architecture | Encoder-decoder with skip connections |
| Source | `segmentation_models_pytorch` |
| VRAM needed | ~3 GB |

**Pros:**
- Very fast training, lowest VRAM requirement
- Great for fine-grained boundary detail
- Good for getting a working baseline in under 2 hours

**Cons:**
- Skip connections can cause the model to overfit to training environment textures
- Lower ceiling on mean IoU vs transformer models

**Use this if:** You need a working prototype fast, or are on a CPU/low-VRAM machine.

---

#### Option 4: Mask2Former — NOT RECOMMENDED

Detectron2 dependency, complex setup, overkill for this dataset size. Setup time cost far outweighs performance gain.

---

### Final Recommendation

```
Primary:   SegFormer-B2   (best IoU ceiling, best generalization)
Fallback:  DeepLabV3+     (if VRAM < 6GB)
Prototype: UNet+ResNet34  (if you need something running in < 2 hours)
```

---

## 5. Technical Architecture

### Training Setup

```python
# Recommended hyperparameters (SegFormer-B2)

LEARNING_RATE = 6e-5          # Lower LR for pretrained transformer
BACKBONE_LR   = 6e-6          # 10x lower for backbone (differential LR)
BATCH_SIZE    = 8             # Adjust based on VRAM
EPOCHS        = 50            # With early stopping
IMAGE_SIZE    = (512, 512)
OPTIMIZER     = "AdamW"
SCHEDULER     = "CosineAnnealingLR"
NUM_CLASSES   = 10
```

### Loss Function

```python
import torch
import torch.nn as nn

# Compute class weights from training set pixel frequencies
# Invert frequencies so rare classes (Logs, Flowers) get higher weight
def compute_class_weights(dataloader, num_classes=10):
    counts = torch.zeros(num_classes)
    for _, masks in dataloader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

# Use weighted CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
```

### Differential Learning Rate

```python
# Different LR for backbone vs decode head
optimizer = torch.optim.AdamW([
    {"params": model.segformer.parameters(), "lr": 6e-6},   # backbone
    {"params": model.decode_head.parameters(), "lr": 6e-5}, # head
], weight_decay=1e-4)
```

---

## 6. Execution Pipeline

### Phase 1 — Environment Setup (Day 1 Morning, ~2 hrs)

1. Create free Falcon account at `falcon.duality.ai`
2. Download dataset — navigate to **"Segmentation Track"** section specifically
3. Set up environment:
   - Windows: run `setup_env.bat` in Anaconda Prompt
   - Mac/Linux: create equivalent `setup_env.sh` with conda commands
   - This creates a conda environment called `EDU`
4. Run the sample `train.py` as-is to get the **baseline IoU** — record this number

### Phase 2 — Baseline & Dataset Setup (Day 1 Midday, ~3 hrs)

1. Explore dataset — count images in train/val, inspect class distribution
2. Write custom `Dataset` class with ID remapping
3. Verify masks render correctly using the provided visualization script
4. Run baseline and record: predictions, loss metrics, per-class IoU

### Phase 3 — Model Fine-tuning (Day 1 Afternoon, ~4 hrs)

1. Install `transformers` and `segmentation-models-pytorch` in EDU env
2. Load SegFormer-B2 with swapped classification head
3. Add augmentation pipeline (see Section 7)
4. Train with weighted CrossEntropyLoss + AdamW + cosine LR scheduler
5. Save checkpoint every 5 epochs, keep best val IoU

### Phase 4 — Optimization Iterations (Day 2, ~6 hrs)

1. Evaluate per-class IoU on val set — identify worst classes
2. For low-IoU rare classes (Logs, Flowers): add targeted augmentations or oversample
3. Try test-time augmentation (TTA): horizontal flip + average predictions
4. Try ensemble of 2 model checkpoints if time permits
5. Run final `test.py` on testImages — this is your submission score

### Phase 5 — Package & Submit (Final Day, ~4 hrs)

1. Write README.md with step-by-step run instructions
2. Write the 8-page report (see Section 10)
3. Compress everything into `.zip`
4. Upload to private GitHub, add collaborators
5. Fill out submission form

---

## 7. Augmentation Strategy

### Must Have (Always Apply)

| Augmentation | Parameters | Why |
|-------------|------------|-----|
| `ColorJitter` | brightness=0.3, contrast=0.3, hue=0.1 | Desert lighting shifts heavily by time of day |
| `RandomHorizontalFlip` | p=0.5 | Free augmentation, always include |
| `RandomResizedCrop` | scale=(0.5, 1.0) | Simulates different camera distances, multi-scale objects |
| `Normalize` | ImageNet mean/std | Required for pretrained backbone |

### Recommended (Add After Baseline)

| Augmentation | Parameters | Why |
|-------------|------------|-----|
| `GaussianBlur` | kernel=(3,7) | Synthetic images are too crisp — blur bridges sim-to-real gap |
| `RandomGrayscale` | p=0.1 | Prevents overreliance on color cues |
| `RandomRotation` | degrees=10 | Desert terrain looks similar at slight angles |
| `GridDistortion` | from albumentations | Mimics lens distortion, improves robustness |

### For Rare Classes (Logs, Flowers, Ground Clutter)

| Augmentation | Why |
|-------------|-----|
| `CoarseDropout` / CutOut | Simulates occlusion — Logs often hidden behind other vegetation |
| Oversampling | Weight dataset sampler to show rare-class images more frequently |
| Copy-paste augmentation | Copy instances of rare classes into other images |

```python
import albumentations as A

train_transform = A.Compose([
    A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomGrayscale(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.pytorch.ToTensorV2(),
], additional_targets={"mask": "mask"})
```

---

## 8. Evaluation & Metrics

### Primary Metric: Mean IoU

IoU (Intersection over Union) per class:

```
IoU_class = TP / (TP + FP + FN)
Mean IoU  = average of IoU across all 10 classes
```

A class with 0 IoU drags the mean down significantly. Every class matters equally.

### What to Track

| Metric | When to Log | Tool |
|--------|-------------|------|
| Training loss | Every epoch | matplotlib / TensorBoard |
| Validation loss | Every epoch | matplotlib / TensorBoard |
| Per-class IoU | Every 5 epochs | custom eval loop |
| Mean IoU | Every epoch | custom eval loop |
| Inference time | Final evaluation | `time.time()` |

### Benchmark Targets

| Metric | Baseline (expected) | Good | Excellent |
|--------|---------------------|------|-----------|
| Mean IoU | 0.25 – 0.35 | 0.50 – 0.65 | > 0.70 |
| Inference speed | — | < 100ms | < 50ms |
| Training loss | — | Steadily decreasing | Converged smoothly |

### Signs of Problems

- **Loss plateaus high** → model underfitting → increase epochs, reduce regularization
- **Val loss increases while train loss decreases** → overfitting → add augmentation, reduce LR
- **Specific class IoU = 0** → class remapping bug, or weighted loss needed

---

## 9. Key Deliverables

### 1. Final Packaged Folder (GitHub)

```
submission/
├── train.py              # Full training script with your improvements
├── test.py               # Inference script for testImages
├── config.py             # All hyperparameters, class mappings
├── dataset.py            # Custom Dataset class with remapping
├── model.py              # Model definition / fine-tuning setup
├── utils.py              # IoU computation, visualization helpers
├── checkpoints/
│   └── best_model.pth    # Best validation IoU checkpoint
├── runs/                 # Training logs, loss graphs
└── README.md             # Run instructions (see below)
```

### 2. README.md Must Include

- Step-by-step instructions to reproduce training
- Step-by-step instructions to run inference on testImages
- Conda environment setup instructions
- Expected outputs and how to interpret them
- Hardware requirements (GPU VRAM, RAM)

### 3. Hackathon Report (PDF or DOCX, 8 pages max)

See Section 10 for full structure.

---

## 10. Report Structure

Follow this storytelling arc: **Problem → Fix → Results → Challenges → Future Work**

### Page 1: Title Page
- Team name
- Project name
- One-line tagline describing your approach
- Date

### Page 2: Methodology
- Model choice and rationale (why SegFormer-B2 over alternatives)
- Architecture overview with a simple diagram
- Training setup: optimizer, learning rate, batch size, epochs, scheduler
- Class remapping explanation
- Augmentation pipeline list

### Pages 3–4: Results & Performance Metrics
- **Mean IoU score** (prominently displayed)
- **Per-class IoU table** — all 10 classes with individual scores
- **Training loss curve** — screenshot from runs/
- **Validation loss curve** — screenshot from runs/
- Before vs after comparison (baseline IoU → final IoU)
- Sample segmentation output images (predicted vs ground truth)

### Pages 5–6: Challenges & Solutions

Use this format for each challenge:

```
Task:    [what you were trying to do]
Issue:   [what went wrong]
Fix:     [what you changed]
Result:  [IoU before → IoU after]
```

**Cover at minimum:**
- Class imbalance (Logs, Flowers, Ground Clutter)
- Domain shift (train env vs test env difference)
- Any misclassification patterns you observed

**Include failure case images** — show a prediction that went wrong and explain why.

### Page 7: Conclusion & Future Work

- Final mean IoU and what it means in context
- 3–5 things you would do with more time:
  - Test-time augmentation (TTA)
  - Ensemble of multiple model checkpoints
  - Domain adaptation techniques
  - Self-supervised pretraining on more desert imagery
  - Multi-scale inference

### Tips for Maximum Report Score

> The 20 report points are essentially free if you document your process well. Judges want to see that you *understood* what you did — not just that you ran a script.

- Show your **iteration loop** — a table of IoU scores across experiments is more impressive than one final number
- **Explain your reasoning** — why did you pick weighted loss? Why SegFormer and not UNet?
- **Failure analysis = free points** — honest failure cases with analysis score higher than a polished-looking report that hides problems
- Write for a smart reader who isn't familiar with your specific code

---

## 11. Submission Checklist

### Code Package

- [ ] `train.py` — complete training script
- [ ] `test.py` — inference on testImages
- [ ] `config.py` — hyperparams and class map
- [ ] `best_model.pth` — best checkpoint by val IoU
- [ ] `README.md` — step-by-step run instructions
- [ ] `requirements.txt` or `environment.yml`
- [ ] Loss graphs / training logs in `runs/`
- [ ] All scripts run without errors from README instructions

### GitHub

- [ ] Repo is **private**
- [ ] Everything zipped and uploaded
- [ ] Collaborator added: `Maazsyedm` (Syed Muhammad Maaz)
- [ ] Collaborator added: `rebekah-bogdanoff` (Rebekah Bogdanoff)
- [ ] Collaborator added: `egold010` (Evan Goldman)

### Submission Form

- [ ] Final IoU score reported
- [ ] GitHub repository link provided

### Report

- [ ] PDF or DOCX format
- [ ] 8 pages maximum
- [ ] Per-class IoU table included
- [ ] Loss graphs included
- [ ] Failure case images included
- [ ] Methodology clearly explained

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Accidentally using testImages in training | Low | Critical (DQ) | Keep a strict import barrier — testImages never touched by train.py |
| Class remapping bug (silent wrong labels) | Medium | High | Visualize 5 remapped masks before training starts |
| Class imbalance kills rare class IoU | High | High | Weighted CrossEntropyLoss from day 1 |
| Overfitting to train environment | High | High | Strong augmentation pipeline — ColorJitter, blur, crop |
| GPU OOM | Medium | Medium | Reduce batch size, switch to SegFormer-B0 or B1 |
| Training too slow | Medium | Medium | Reduce batch size, lower image resolution to 384×384 |
| Report too short / unstructured | Medium | Medium (20 pts) | Follow the 7-page template strictly |

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **Semantic Segmentation** | Task of assigning a class label to every pixel in an image |
| **IoU (Intersection over Union)** | Overlap between predicted mask and ground truth mask. 1.0 = perfect, 0.0 = no overlap |
| **Mean IoU** | Average IoU across all classes — the primary competition metric |
| **Domain Shift** | When the statistical distribution of test data differs from training data |
| **Class Imbalance** | When some classes appear far more often than others in the dataset |
| **Fine-tuning** | Starting from pretrained weights and training further on a specific dataset |
| **Digital Twin** | A virtual replica of a real-world environment used to generate synthetic data |
| **Backbone** | The feature extraction part of a neural network (e.g. ResNet, MiT) |
| **Decode Head** | The part of the segmentation model that converts features into per-pixel predictions |
| **Weighted Loss** | A loss function that penalizes errors on rare classes more heavily |
| **TTA (Test-Time Augmentation)** | Running inference multiple times with different augmentations and averaging results |
| **Training Loss** | Error on training data — should decrease steadily |
| **Validation Loss** | Error on held-out val data — monitors overfitting |
| **Inference Speed** | Time to predict segmentation for one image. Target: < 50ms |
| **Ground Truth** | The manually annotated correct segmentation mask |

---

*— PRD authored for Duality AI Offroad Segmentation Hackathon*
*Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda*
