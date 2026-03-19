"""
Custom Dataset class for Duality AI Offroad Segmentation Challenge
Handles loading, class ID remapping, and augmentations
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    ORIGINAL_ID_MAP,
    NUM_CLASSES,
    IMAGE_SIZE,
    AUG_COLOR_JITTER,
    AUG_HORIZONTAL_FLIP,
    AUG_RANDOM_RESIZED_CROP,
    AUG_GAUSSIAN_BLUR,
    AUG_RANDOM_GRAYSCALE,
    AUG_RANDOM_ROTATION,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert raw mask pixel values to sequential class IDs [0, N-1].
    
    Args:
        mask: Raw mask as numpy array with original class IDs
        
    Returns:
        Remapped mask with class IDs in range [0, NUM_CLASSES-1]
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for orig_id, new_id in ORIGINAL_ID_MAP.items():
        remapped[mask == orig_id] = new_id
    return remapped


def get_train_transform():
    """
    Get training augmentation pipeline.
    
    Includes:
    - RandomResizedCrop for multi-scale training
    - HorizontalFlip for invariance
    - ColorJitter for lighting robustness
    - GaussianBlur for sim-to-real gap
    - RandomGrayscale for color invariance
    - RandomRotation for angle robustness
    - Normalization for pretrained backbone
    """
    return A.Compose([
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
        A.GaussianBlur(
            blur_limit=AUG_GAUSSIAN_BLUR["blur_limit"],
            p=AUG_GAUSSIAN_BLUR["p"]
        ),
        A.RandomGrayscale(p=AUG_RANDOM_GRAYSCALE["p"]),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=AUG_RANDOM_ROTATION["degrees"],
            p=AUG_RANDOM_ROTATION["p"]
        ),
        # Rare-class augmentations
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_val_transform():
    """
    Get validation/test augmentation pipeline.
    
    Only includes:
    - Resize to fixed size
    - Normalization for pretrained backbone
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_test_transform():
    """
    Get test augmentation pipeline (no mask).
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        ToTensorV2(),
    ])


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for offroad semantic segmentation.
    
    Expects directory structure:
        data_dir/
        ├── Color_Images/
        │   ├── image1.png
        │   └── image2.png
        └── Segmentation/
            ├── image1.png
            └── image2.png
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
        
        # Filter out non-image files
        self.image_files = [
            f for f in self.image_files
            if os.path.splitext(f)[1].lower() in self.image_extensions
        ]
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        image_id = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_id)
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if self.is_test:
            # Test mode: only return image
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
        
        # Remap mask values to [0, N-1]
        mask = remap_mask(mask)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask, image_id


def build_rare_class_sampler(
    dataset: Dataset,
    rare_class_ids: list = None,
    rare_boost: float = 3.0,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples images containing rare classes.

    Args:
        dataset: SegmentationDataset instance
        rare_class_ids: Remapped class IDs to boost (default: Logs=6, Flowers=5, Ground Clutter=4)
        rare_boost: Multiplier applied to images containing any rare class

    Returns:
        WeightedRandomSampler
    """
    if rare_class_ids is None:
        rare_class_ids = {4, 5, 6}  # Ground Clutter, Flowers, Logs

    weights = []
    for idx in range(len(dataset)):
        _, mask, _ = dataset[idx]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        has_rare = any((mask_np == c).any() for c in rare_class_ids)
        weights.append(rare_boost if has_rare else 1.0)

    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def compute_class_weights(dataloader: DataLoader, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Compute class weights from training set pixel frequencies.
    
    Uses inverse frequency weighting to handle class imbalance.
    Rare classes (Logs, Flowers, Ground Clutter) get higher weights.
    
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
    
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (counts + 1e-6)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    use_rare_sampler: bool = True,
):
    """
    Create training and validation DataLoaders.

    Args:
        train_dir: Path to training dataset directory
        val_dir: Path to validation dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        use_rare_sampler: Oversample images with rare classes via WeightedRandomSampler

    Returns:
        train_loader, val_loader, class_weights
    """
    train_dataset = SegmentationDataset(train_dir, transform=get_train_transform())
    val_dataset = SegmentationDataset(val_dir, transform=get_val_transform())

    sampler = None
    shuffle = True
    if use_rare_sampler:
        print("Building rare-class sampler (Logs, Flowers, Ground Clutter)...")
        sampler = build_rare_class_sampler(train_dataset)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = compute_class_weights(train_loader, num_classes=NUM_CLASSES)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class weights: {class_weights}")

    return train_loader, val_loader, class_weights


def create_test_dataloader(
    test_dir: str,
    batch_size: int = 8,
    num_workers: int = 4
):
    """
    Create test DataLoader (no masks).
    
    Args:
        test_dir: Path to test dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        
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
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader
