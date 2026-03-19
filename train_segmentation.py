"""
Training script for Duality AI Offroad Segmentation Challenge

Trains a SegFormer-B2 model (or alternative) on synthetic desert imagery
with weighted loss and data augmentation for domain generalization.

Usage:
    python train_segmentation.py [--model_type segformer_b2] [--epochs 50] [--batch_size 8]
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    NUM_CLASSES,
    LEARNING_RATE,
    BACKBONE_LR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    TRAIN_DIR,
    VAL_DIR,
    CHECKPOINTS_DIR,
    RUNS_DIR,
    DEVICE,
    setup_directories,
)
from dataset import create_dataloaders
from model import load_model, get_differential_lr_params, save_checkpoint
from utils import evaluate_model, save_training_plots, save_history_to_file


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
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
        
        # Forward pass
        outputs = model(imgs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / num_batches


def validate_epoch(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int) -> float:
    """
    Validate for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False, unit="batch")
    
    with torch.no_grad():
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(imgs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Compute loss
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / num_batches


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument("--model_type", type=str, default="segformer_b2",
                        choices=["segformer_b2", "deeplabv3", "unet"],
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate for decode head")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Learning rate for backbone (default: lr/10)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    device = torch.device(args.device) if args.device else DEVICE
    
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, class_weights = create_dataloaders(
        TRAIN_DIR,
        VAL_DIR,
        batch_size=args.batch_size,
        num_workers=0  # Set to >0 for faster loading on Linux
    )
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    # Create model
    print(f"\nLoading {args.model_type} model...")
    model = load_model(args.model_type, num_classes=NUM_CLASSES, device=device)
    
    # Setup differential learning rates for SegFormer
    if args.model_type == "segformer_b2":
        backbone_lr = args.backbone_lr if args.backbone_lr else args.lr / 10
        optimizer_params = get_differential_lr_params(model, lr=args.lr, backbone_lr=backbone_lr)
        print(f"Using differential LR: backbone={backbone_lr}, head={args.lr}")
    else:
        optimizer_params = model.parameters()
    
    # Create optimizer
    optimizer = optim.AdamW(optimizer_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"Optimizer: AdamW (weight_decay={WEIGHT_DECAY})")
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print(f"Loss: CrossEntropyLoss with class weights and label smoothing")
    
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
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    best_val_iou = 0.0
    patience_counter = 0
    
    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Evaluate metrics
        train_metrics = evaluate_model(model, None, train_loader, device, 
                                       num_classes=NUM_CLASSES, show_progress=False)
        val_metrics = evaluate_model(model, None, val_loader, device, 
                                     num_classes=NUM_CLASSES, show_progress=False)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['mean_iou'])
        history['val_iou'].append(val_metrics['mean_iou'])
        history['train_dice'].append(train_metrics['mean_dice'])
        history['val_dice'].append(val_metrics['mean_dice'])
        history['train_pixel_acc'].append(train_metrics['pixel_accuracy'])
        history['val_pixel_acc'].append(val_metrics['pixel_accuracy'])
        
        # Update scheduler
        scheduler.step()
        
        # Update progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            val_iou=f"{val_metrics['mean_iou']:.3f}",
            val_acc=f"{val_metrics['pixel_accuracy']:.3f}"
        )
        
        # Check for improvement
        if val_metrics['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['mean_iou']
            patience_counter = 0
            
            # Save best model
            save_checkpoint(model, optimizer, epoch, val_metrics['mean_iou'],
                          os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
            print(f"  [NEW BEST] Val IoU: {val_metrics['mean_iou']:.4f}")
        else:
            patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
        
        # Save last model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics['mean_iou'],
                          os.path.join(CHECKPOINTS_DIR, "last_model.pth"))
    
    # Save final plots and history
    print("\nSaving training results...")
    save_training_plots(history, RUNS_DIR)
    save_history_to_file(history, RUNS_DIR)
    
    # Save final model
    save_checkpoint(model, optimizer, len(history['train_loss']) - 1, 
                   history['val_iou'][-1],
                   os.path.join(CHECKPOINTS_DIR, "final_model.pth"))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Final Validation IoU: {history['val_iou'][-1]:.4f}")
    print(f"Final Validation Dice: {history['val_dice'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print(f"\nOutputs saved to:")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Training logs: {RUNS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
