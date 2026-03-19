"""
Training script for Duality AI Offroad Segmentation Challenge
PRD Version: SegFormer-B2 | RTX 3070 Laptop 8GB VRAM

Trains SegFormer-B2 with:
- Mixed Precision (AMP) for VRAM efficiency
- Weighted CrossEntropyLoss for class imbalance
- Data augmentation for domain generalization
- Differential learning rates (backbone vs head)
- Gradient clipping for stability
- Cosine annealing LR scheduler
- Early stopping

Usage:
    python train_segmentation.py [--epochs 60] [--batch_size 8]

Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    NUM_CLASSES,
    LEARNING_RATE,
    BACKBONE_LR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    GRAD_CLIP_MAX_NORM,
    T_MAX,
    ETA_MIN,
    USE_AMP,
    TRAIN_DIR,
    VAL_DIR,
    CHECKPOINTS_DIR,
    RUNS_DIR,
    DEVICE,
    setup_directories,
    validate_hardware,
    IOU_TARGETS,
)
from dataset import create_dataloaders
from model import load_model, get_differential_lr_params, save_checkpoint, print_model_summary
from utils import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    save_training_plots,
    save_history_to_file,
)


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scaler: GradScaler,
                device: torch.device,
                epoch: int,
                grad_clip: float = GRAD_CLIP_MAX_NORM) -> float:
    """
    Train for one epoch with Mixed Precision (AMP).
    
    PRD: AMP runs the forward pass in float16 instead of float32.
    This reduces VRAM usage by 30-40% with no accuracy loss.
    On a laptop GPU with power constraints, this is mandatory.
    
    Args:
        model: SegFormer model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP GradScaler
        device: Device to train on
        epoch: Current epoch number
        grad_clip: Maximum gradient norm for clipping
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False, unit="batch")
    
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass (AMP)
        with autocast(enabled=USE_AMP):
            outputs = model(imgs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Compute weighted CrossEntropyLoss
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling (AMP)
        scaler.scale(loss).backward()
        
        # Gradient clipping (PRD optimization)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step with scaled gradients (AMP)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / num_batches


def validate_epoch(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int) -> tuple:
    """
    Validate for one epoch.
    
    Args:
        model: SegFormer model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        (avg_val_loss, mean_iou, mean_dice, pixel_accuracy)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_ious = []
    all_dices = []
    all_accs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False, unit="batch")
    
    with torch.no_grad():
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            
            # Forward pass (no AMP needed for validation)
            outputs = model(imgs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Compute loss
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            iou, _ = compute_iou(logits, labels, num_classes=NUM_CLASSES)
            dice, _ = compute_dice(logits, labels, num_classes=NUM_CLASSES)
            acc = compute_pixel_accuracy(logits, labels)
            
            all_ious.append(iou)
            all_dices.append(dice)
            all_accs.append(acc)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.3f}")
    
    avg_loss = total_loss / num_batches
    mean_iou = sum(all_ious) / len(all_ious)
    mean_dice = sum(all_dices) / len(all_dices)
    mean_acc = sum(all_accs) / len(all_accs)
    
    return avg_loss, mean_iou, mean_dice, mean_acc


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train SegFormer-B2 for offroad segmentation")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate for decode head (default: {LEARNING_RATE})")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help=f"Learning rate for backbone (default: {BACKBONE_LR})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--amp", action="store_true", default=USE_AMP,
                        help="Use Mixed Precision (AMP) training")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable Mixed Precision (AMP)")
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Validate hardware
    validate_hardware()
    
    # Device setup
    device = torch.device(args.device) if args.device else DEVICE
    print(f"\nUsing device: {device}")
    print(f"Mixed Precision (AMP): {'Enabled' if args.amp else 'Disabled'}")
    
    # Print model summary
    print_model_summary(None)  # Will print in load_model
    
    # Create dataloaders
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    train_loader, val_loader, class_weights = create_dataloaders(
        TRAIN_DIR,
        VAL_DIR,
        batch_size=args.batch_size,
    )
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model = load_model(num_classes=NUM_CLASSES, device=device)
    print_model_summary(model)
    
    # Setup differential learning rates
    backbone_lr = args.backbone_lr if args.backbone_lr else BACKBONE_LR
    optimizer_params = get_differential_lr_params(model, lr=args.lr, backbone_lr=backbone_lr)
    print(f"\nDifferential Learning Rates:")
    print(f"  Backbone LR: {backbone_lr}")
    print(f"  Decode Head LR: {args.lr}")
    print(f"  Ratio: {args.lr / backbone_lr:.0f}x")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"\nOptimizer: AdamW")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Gradient clipping: {GRAD_CLIP_MAX_NORM}")
    
    # Create scheduler (Cosine Annealing per PRD)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
    print(f"\nScheduler: CosineAnnealingLR")
    print(f"  T_max: {T_MAX}")
    print(f"  ETA min: {ETA_MIN}")
    
    # Create loss function with class weights (PRD: weighted CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\nLoss: CrossEntropyLoss with class weights")
    
    # Initialize AMP GradScaler
    scaler = GradScaler(enabled=args.amp)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': []
    }
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Expected time: ~4-6 min/epoch on RTX 3070 Laptop")
    print("=" * 60 + "\n")
    
    best_val_iou = 0.0
    patience_counter = 0
    start_epoch = 0
    
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, grad_clip=GRAD_CLIP_MAX_NORM
        )
        
        # Validate
        val_loss, val_iou, val_dice, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Compute training metrics (sample)
        train_metrics = {
            'mean_iou': 0.0,
            'mean_dice': 0.0,
            'pixel_accuracy': 0.0,
        }
        # Quick train evaluation on subset
        model.eval()
        with torch.no_grad():
            for i, (imgs, labels, _) in enumerate(train_loader):
                if i >= 5:  # Sample first 5 batches
                    break
                imgs = imgs.to(device)
                labels = labels.to(device).long()
                outputs = model(imgs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                iou, _ = compute_iou(logits, labels, num_classes=NUM_CLASSES)
                dice, _ = compute_dice(logits, labels, num_classes=NUM_CLASSES)
                acc = compute_pixel_accuracy(logits, labels)
                train_metrics['mean_iou'] += iou
                train_metrics['mean_dice'] += dice
                train_metrics['pixel_accuracy'] += acc
            train_metrics['mean_iou'] /= 5
            train_metrics['mean_dice'] /= 5
            train_metrics['pixel_accuracy'] /= 5
        model.train()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['mean_iou'])
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_metrics['mean_dice'])
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_metrics['pixel_accuracy'])
        history['val_pixel_acc'].append(val_acc)
        
        # Update scheduler
        scheduler.step()
        
        # Update progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_acc:.3f}"
        )
        
        # Check for improvement
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            # Save best model
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_iou, val_loss,
                os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
                scaler=scaler if args.amp else None
            )
            print(f"  [NEW BEST] Val IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
        
        # Save last model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_iou, val_loss,
                os.path.join(CHECKPOINTS_DIR, "last_model.pth"),
                scaler=scaler if args.amp else None
            )
    
    # Save final plots and history
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_training_plots(history, RUNS_DIR)
    save_history_to_file(history, RUNS_DIR)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, len(history['train_loss']) - 1,
        history['val_iou'][-1], history['val_loss'][-1],
        os.path.join(CHECKPOINTS_DIR, "final_model.pth"),
        scaler=scaler if args.amp else None
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Final Validation IoU: {history['val_iou'][-1]:.4f}")
    print(f"Final Validation Dice: {history['val_dice'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    
    # IoU benchmark comparison
    print(f"\nIoU Benchmark Comparison:")
    print(f"  Baseline target: {IOU_TARGETS['baseline'][0]} - {IOU_TARGETS['baseline'][1]}")
    print(f"  SegFormer (no aug): {IOU_TARGETS['segformer_no_aug'][0]} - {IOU_TARGETS['segformer_no_aug'][1]}")
    print(f"  SegFormer (full aug): {IOU_TARGETS['segformer_full'][0]} - {IOU_TARGETS['segformer_full'][1]}")
    print(f"  Your best IoU: {best_val_iou:.4f}")
    
    if best_val_iou >= IOU_TARGETS['segformer_full'][1]:
        print(f"  Status: EXCELLENT - Above target!")
    elif best_val_iou >= IOU_TARGETS['segformer_full'][0]:
        print(f"  Status: GOOD - Within target range")
    elif best_val_iou >= IOU_TARGETS['segformer_no_aug'][0]:
        print(f"  Status: FAIR - Consider adding more augmentation")
    else:
        print(f"  Status: NEEDS IMPROVEMENT - Check class weights and augmentation")
    
    print(f"\nOutputs saved to:")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Training logs: {RUNS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
