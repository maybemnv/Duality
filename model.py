"""
Model definitions for Duality AI Offroad Segmentation Challenge
PRD Version: SegFormer-B2 Only

Model: nvidia/segformer-b2-finetuned-ade-512-512
Architecture: Mix Transformer (MiT-B2) encoder + lightweight MLP decode head
Pretrained on: ImageNet-1K + ADE20K semantic segmentation
Parameters: ~25M
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from config import (
    PRETRAINED_MODEL_NAME,
    NUM_CLASSES,
    LEARNING_RATE,
    BACKBONE_LR,
)


def create_segformer_b2(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create SegFormer-B2 model for semantic segmentation.
    
    Uses pretrained weights from ADE20K and swaps the classification head
    for our 10-class desert segmentation problem.
    
    Args:
        num_classes: Number of segmentation classes (default: 10)
        
    Returns:
        SegFormer-B2 model ready for fine-tuning
        
    PRD Rationale:
    - Dense prediction requirement: 512×512 = 262,144 pixel predictions
    - Pretrained weights converge in hours vs days for training from scratch
    - Transformer self-attention handles domain shift better than CNNs
    - ADE20K pretraining includes outdoor scenes, vegetation, terrain
    - Hardware fit: ~5.5-6GB VRAM at batch size 8 with AMP on RTX 3070
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Allows loading pretrained weights with different head size
        output_hidden_states=False,
    )
    
    return model


def get_differential_lr_params(model: nn.Module, 
                                lr: float = LEARNING_RATE, 
                                backbone_lr: float = BACKBONE_LR):
    """
    Set up differential learning rates for backbone vs decode head.
    
    PRD Rationale:
    The backbone (MiT-B2 encoder) already has good feature representations 
    from ADE20K pretraining. Applying a high learning rate destroys those 
    pretrained weights. The decode head is randomly initialized and needs 
    a higher learning rate to learn quickly.
    
    Args:
        model: SegFormer model
        lr: Learning rate for decode head (default: 6e-5)
        backbone_lr: Learning rate for backbone (default: 6e-6, 10x lower)
        
    Returns:
        Parameter groups for optimizer
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        # SegFormer parameter naming in transformers library
        if "segformer" in name or "encoder" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ]


def load_model(num_classes: int = NUM_CLASSES,
               device: torch.device = None) -> nn.Module:
    """
    Load SegFormer-B2 model.
    
    Args:
        num_classes: Number of segmentation classes
        device: Target device
        
    Returns:
        SegFormer-B2 model on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading SegFormer-B2 from: {PRETRAINED_MODEL_NAME}")
    model = create_segformer_b2(num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model loaded successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    val_iou: float,
                    val_loss: float,
                    filepath: str,
                    scaler: torch.cuda.amp.GradScaler = None):
    """
    Save model checkpoint with all training state.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        val_iou: Validation IoU score
        val_loss: Validation loss
        filepath: Path to save checkpoint
        scaler: AMP GradScaler state (optional)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_iou": val_iou,
        "val_loss": val_loss,
    }
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath} (epoch {epoch}, val_iou: {val_iou:.4f})")


def load_checkpoint(filepath: str, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    scaler: torch.cuda.amp.GradScaler = None,
                    device: torch.device = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
        device: Target device
        
    Returns:
        Checkpoint metadata (epoch, val_iou, val_loss, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    print(f"Loaded checkpoint from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary.
    
    Args:
        model: SegFormer model
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE: SegFormer-B2")
    print("=" * 60)
    print(f"Pretrained model: {PRETRAINED_MODEL_NAME}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Image size: 512x512")
    print(f"\nArchitecture:")
    print(f"  Encoder: Mix Transformer (MiT-B2)")
    print(f"  Decode Head: MLP")
    print(f"  Pretrained on: ImageNet-1K + ADE20K")
    print(f"\nWhy SegFormer-B2 (PRD rationale):")
    print(f"  1. Dense prediction: 262,144 pixel outputs per 512x512 image")
    print(f"  2. Pretrained weights: converges in hours, not days")
    print(f"  3. Transformer attention: handles domain shift better than CNNs")
    print(f"  4. ADE20K pretraining: includes outdoor/vegetation/terrain")
    print(f"  5. Hardware fit: ~6GB VRAM on RTX 3070 Laptop")
    print("=" * 60)
