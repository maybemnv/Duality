"""
Custom Dataset class for Duality AI Offroad Segmentation Challenge
PRD Version: SegFormer-B2 | RTX 3070 Laptop 8GB VRAM

Handles loading, class ID remapping, and augmentations per PRD specification.

Critical: Class ID Remapping
PyTorch CrossEntropyLoss requires class labels in range [0, N-1].
The provided mask IDs are non-sequential (100, 200 ... 10000).
Remapping is mandatory — training will silently produce garbage outputs without it.

Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from config import (
    ORIGINAL_ID_MAP,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)
from augmentations import get_train_transform, get_val_transform, get_test_transform


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert raw mask pixel values to sequential class IDs [0, N-1].
    
    PRD Class Mapping:
        100      → 0   (Trees)
        200      → 1   (Lush Bushes)
        300      → 2   (Dry Grass)
        500      → 3   (Dry Bushes)
        550      → 4   (Ground Clutter)
        600      → 5   (Flowers)
        700      → 6   (Logs)
        800      → 7   (Rocks)
        7100     → 8   (Landscape)
        10000    → 9   (Sky)
    
    Verification step: After applying remapping, visualize 5 random masks
    and confirm labels match visible class boundaries before training.
    
    Args:
        mask: Raw mask as numpy array with original class IDs
        
    Returns:
        Remapped mask with class IDs in range [0, NUM_CLASSES-1]
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for orig_id, new_id in ORIGINAL_ID_MAP.items():
        remapped[mask == orig_id] = new_id
    return remapped


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for offroad semantic segmentation.
    
    Expects directory structure (PRD-specified):
        data_dir/
        ├── Color_Images/
        │   ├── image_001.png
        │   └── ...
        └── Segmentation/
            ├── image_001.png
            └── ...
    """
    
    def __init__(self, data_dir: str, transform=None, is_test: bool = False):
        """
        Args:
            data_dir: Path to dataset directory
            transform: Albumentations transform to apply
            is_test: If True, only load images (no masks)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        
        # Image directory
        self.image_dir = os.path.join(data_dir, "Color_Images")
        if not os.path.exists(self.image_dir):
            self.image_dir = data_dir  # Fallback if images are directly in data_dir
        
        # Mask directory (only for train/val)
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        
        # Get all image files
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f)) and
            os.path.splitext(f)[1].lower() in self.image_extensions
        ])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        image_id = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_id)
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if self.is_test:
            # Test mode: only return image (testImages has no masks)
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, image_id
        
        # Train/Val mode: load mask too
        mask_path = os.path.join(self.masks_dir, image_id)
        if not os.path.exists(mask_path):
            # Try with same stem but different extension
            mask_stem = os.path.splitext(image_id)[0]
            for ext in ['.png', '.jpg', '.tif']:
                alt_path = os.path.join(self.masks_dir, mask_stem + ext)
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    break
        
        mask = np.array(Image.open(mask_path))
        
        # CRITICAL: Remap mask values to [0, N-1]
        mask = remap_mask(mask)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask, image_id


def compute_class_weights(dataloader: DataLoader, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Compute class weights from training set pixel frequencies.
    
    PRD: The imbalance problem
    Landscape (class 8) and Sky (class 9) dominate pixel counts.
    Logs (class 6), Flowers (class 5), and Ground Clutter (class 4)
    appear in very few pixels. A naive CrossEntropyLoss optimizes for
    majority classes and ignores rare ones, producing IoU = 0.
    
    Since mean IoU averages all 10 classes equally, a class with IoU = 0
    heavily penalizes the score.
    
    Solution: weighted CrossEntropyLoss
    1. Count pixel frequency per class across entire training set
    2. Invert frequencies: rare class → high weight, dominant class → low weight
    3. Normalize weights so they sum to num_classes
    
    Computed weight formula:
        weight[c] = 1 / (pixel_count[c] + epsilon)
        weight    = weight / weight.sum() * num_classes
    
    Args:
        dataloader: DataLoader with training dataset
        num_classes: Number of segmentation classes
        
    Returns:
        Tensor of class weights for CrossEntropyLoss
    """
    counts = torch.zeros(num_classes)
    
    for _, masks, _ in dataloader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum()
    
    # Add epsilon to avoid division by zero
    weights = 1.0 / (counts + 1e-6)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights


def create_dataloaders(
    train_dir: str = TRAIN_DIR,
    val_dir: str = VAL_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    """
    Create training and validation DataLoaders.
    
    PRD Hardware Configuration (RTX 3070 Laptop 8GB VRAM):
    - BATCH_SIZE: 8 (safe ceiling with AMP enabled)
    - NUM_WORKERS: 4 (comfortable for 16GB system RAM)
    - PIN_MEMORY: True (faster CPU to GPU transfer)
    
    Args:
        train_dir: Path to training dataset directory
        val_dir: Path to validation dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster transfer
        
    Returns:
        train_loader, val_loader, class_weights
    """
    # Create datasets
    train_dataset = SegmentationDataset(
        train_dir,
        transform=get_train_transform(),
        is_test=False
    )
    
    val_dataset = SegmentationDataset(
        val_dir,
        transform=get_val_transform(),
        is_test=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    
    # Compute class weights from training data
    class_weights = compute_class_weights(train_loader, num_classes=NUM_CLASSES)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"\nClass Weights (for weighted CrossEntropyLoss):")
    for i, (name, weight) in enumerate(zip(
        ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter',
         'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'],
        class_weights.tolist()
    )):
        print(f"  {i}: {name:15} = {weight:.4f}")
    
    return train_loader, val_loader, class_weights


def create_test_dataloader(
    test_dir: str = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    """
    Create test DataLoader (no masks).
    
    PRD: testImages folder is strictly off-limits for training.
    Only used for final inference after training is complete.
    
    Args:
        test_dir: Path to test dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster transfer
        
    Returns:
        test_loader
    """
    test_dataset = SegmentationDataset(
        test_dir,
        transform=get_test_transform(),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nTest Dataset:")
    print(f"  Test samples: {len(test_dataset)}")
    
    return test_loader


def verify_mask_remap(data_dir: str, num_samples: int = 5):
    """
    Verify mask remapping is correct before training.
    
    PRD Verification step: After applying remapping, visualize 5 random
    masks and confirm labels match visible class boundaries.
    
    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to verify
    """
    import matplotlib.pyplot as plt
    from config import CLASS_NAMES, COLOR_PALETTE
    
    dataset = SegmentationDataset(data_dir, transform=None, is_test=False)
    
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        _, mask, image_id = dataset[i]
        
        # Create color visualization
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(COLOR_PALETTE):
            color_mask[mask == class_id] = color
        
        ax.imshow(color_mask)
        ax.set_title(f"{image_id}\nClasses: {np.unique(mask)}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved mask verification to mask_verification.png")
    print("Verify that class boundaries match expected segmentation.")
