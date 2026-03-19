"""
Model definitions for Duality AI Offroad Segmentation Challenge

Primary model: SegFormer-B2 (pretrained on ADE20K)
Fallback options: DeepLabV3+, UNet
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from config import NUM_CLASSES, PRETRAINED_MODEL_NAME


def create_segformer_b2(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create SegFormer-B2 model for semantic segmentation.
    
    Uses pretrained weights from ADE20K and swaps the classification head
    for our 10-class problem.
    
    Args:
        num_classes: Number of segmentation classes (default: 10)
        
    Returns:
        SegFormer model ready for fine-tuning
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Allows loading pretrained weights with different head size
        output_hidden_states=False,
    )
    
    return model


def create_deeplabv3_plus(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create DeepLabV3+ model with ResNet-50 backbone.
    
    Fallback option if SegFormer is too heavy for available GPU.
    
    Args:
        num_classes: Number of segmentation classes (default: 10)
        
    Returns:
        DeepLabV3+ model ready for fine-tuning
    """
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    
    # Load pretrained weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    
    # Replace classifier head for our number of classes
    model.classifier = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, kernel_size=1),
    )
    
    return model


def create_unet_resnet34(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create UNet with ResNet-34 encoder.
    
    Prototype/low-resource option for quick baseline.
    
    Args:
        num_classes: Number of segmentation classes (default: 10)
        
    Returns:
        UNet model ready for training
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("Please install segmentation-models-pytorch: pip install segmentation-models-pytorch")
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None,  # No activation - will use CrossEntropyLoss
    )
    
    return model


def get_differential_lr_params(model: nn.Module, 
                                lr: float = 6e-5, 
                                backbone_lr: float = 6e-6):
    """
    Set up differential learning rates for backbone vs decode head.
    
    The backbone (pretrained) uses a lower learning rate to preserve
    learned features, while the decode head uses a higher learning rate
    for faster adaptation to our task.
    
    Args:
        model: SegFormer model
        lr: Learning rate for decode head
        backbone_lr: Learning rate for backbone (typically 10x lower)
        
    Returns:
        Parameter groups for optimizer
    """
    # SegFormer parameter groups
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "segformer" in name or "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ]


def load_model(model_type: str = "segformer_b2", 
               num_classes: int = NUM_CLASSES,
               device: torch.device = None) -> nn.Module:
    """
    Factory function to create segmentation model.
    
    Args:
        model_type: One of "segformer_b2", "deeplabv3", "unet"
        num_classes: Number of segmentation classes
        device: Target device
        
    Returns:
        Model on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "segformer_b2":
        model = create_segformer_b2(num_classes)
    elif model_type == "deeplabv3":
        model = create_deeplabv3_plus(num_classes)
    elif model_type == "unet":
        model = create_unet_resnet34(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    print(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_iou: float,
                    filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_iou: Validation IoU score
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou": val_iou,
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_checkpoint(filepath: str, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer = None,
                    device: torch.device = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Target device
        
    Returns:
        Checkpoint metadata (epoch, val_iou, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']}, val_iou: {checkpoint['val_iou']:.4f})")
    
    return checkpoint
