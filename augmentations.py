"""
Augmentation pipeline for Duality AI Offroad Segmentation Challenge
PRD Version: RTX 3070 Laptop 8GB VRAM

Augmentation is the primary tool for improving generalization across
the domain shift between train and test environments.

Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMAGE_SIZE,
    AUG_RANDOM_RESIZED_CROP,
    AUG_HORIZONTAL_FLIP,
    AUG_COLOR_JITTER,
    AUG_GAUSSIAN_BLUR,
    AUG_RANDOM_GRAYSCALE,
    AUG_RANDOM_ROTATION,
    AUG_GRID_DISTORTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transform() -> A.Compose:
    """
    Get training augmentation pipeline (PRD-specified).
    
    Must-have augmentations (always applied):
    - RandomResizedCrop: Simulates different camera distances, handles multi-scale objects
    - HorizontalFlip: Free augmentation, zero cost
    - ColorJitter: Desert lighting shifts heavily by time of day and sun angle
    - Normalize: Required for ImageNet-pretrained backbone
    
    Recommended augmentations (add after baseline):
    - GaussianBlur: Synthetic images are too sharp — bridges sim-to-real gap
    - RandomGrayscale: Prevents overreliance on color cues
    - RandomRotation: Desert terrain looks similar at slight angles
    - GridDistortion: Mimics lens distortion, improves boundary robustness
    
    Returns:
        Albumentations composition for training
    """
    return A.Compose([
        # Must-have augmentations
        A.RandomResizedCrop(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1],
            scale=AUG_RANDOM_RESIZED_CROP["scale"],
            p=1.0
        ),
        A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP["p"]),
        A.ColorJitter(
            brightness=AUG_COLOR_JITTER["brightness"],
            contrast=AUG_COLOR_JITTER["contrast"],
            hue=AUG_COLOR_JITTER["hue"],
            p=AUG_COLOR_JITTER["p"]
        ),
        
        # Recommended augmentations
        A.GaussianBlur(
            blur_limit=AUG_GAUSSIAN_BLUR["blur_limit"],
            p=AUG_GAUSSIAN_BLUR["p"]
        ),
        A.RandomGrayscale(p=AUG_RANDOM_GRAYSCALE["p"]),
        A.Affine(
            rotate=(-AUG_RANDOM_ROTATION["degrees"], AUG_RANDOM_ROTATION["degrees"]),
            p=AUG_RANDOM_ROTATION["p"]
        ),
        A.GridDistortion(
            distort_limit=AUG_GRID_DISTORTION["distort_limit"],
            p=AUG_GRID_DISTORTION["p"]
        ),
        
        # Normalization (required for pretrained backbone)
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        
        # Convert to tensor
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_val_transform() -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    
    PRD: Validation set uses only Resize + Normalize.
    No random transforms — validation must be deterministic
    for consistent IoU tracking.
    
    Returns:
        Albumentations composition for validation
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_test_transform() -> A.Compose:
    """
    Get test augmentation pipeline (no mask).
    
    Returns:
        Albumentations composition for test inference
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """
    Get Test-Time Augmentation transforms (PRD optimization).
    
    TTA adds 2-5 IoU points with no additional training.
    Average the softmax outputs from multiple augmented versions,
    then argmax for final prediction.
    
    Returns:
        List of (transform, is_flipped) tuples for TTA
    """
    base_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    flip_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    # Multi-scale transforms
    scale_09 = A.Compose([
        A.Resize(height=int(IMAGE_SIZE[0] * 0.9), width=int(IMAGE_SIZE[1] * 0.9)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    scale_11 = A.Compose([
        A.Resize(height=int(IMAGE_SIZE[0] * 1.1), width=int(IMAGE_SIZE[1] * 1.1)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return [
        (base_transform, False, 1.0),      # Original
        (flip_transform, True, 1.0),       # Flipped
        (scale_09, False, 0.9),            # 0.9x scale
        (scale_11, False, 1.1),            # 1.1x scale
    ]


def apply_tta_to_image(image, model, device, use_flip=True, use_scales=True):
    """
    Apply Test-Time Augmentation to a single image.
    
    Args:
        image: Input image tensor [C, H, W]
        model: Trained SegFormer model
        device: Device to run inference on
        use_flip: Whether to use horizontal flip augmentation
        use_scales: Whether to use multi-scale augmentation
        
    Returns:
        Averaged prediction logits [NUM_CLASSES, H, W]
    """
    import torch
    import torch.nn.functional as F
    
    model.eval()
    predictions = []
    
    tta_transforms = get_tta_transforms()
    
    with torch.no_grad():
        for transform, is_flipped, scale_factor in tta_transforms:
            # Skip scales if disabled
            if not use_scales and scale_factor != 1.0:
                continue
            # Skip flip if disabled
            if not use_flip and is_flipped:
                continue
            
            # Apply transform
            augmented = transform(image=image.numpy().transpose(1, 2, 0))
            aug_tensor = augmented["image"].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(aug_tensor)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Resize to original size if scaled
            if scale_factor != 1.0:
                logits = F.interpolate(
                    logits,
                    size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Flip back if needed
            if is_flipped:
                logits = torch.flip(logits, dims=[3])
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
    
    # Average predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    
    return avg_prediction.squeeze(0)
