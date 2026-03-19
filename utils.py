"""
Utility functions for Duality AI Offroad Segmentation Challenge
Includes metrics computation, visualization, and logging helpers
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import CLASS_NAMES, COLOR_PALETTE, NUM_CLASSES, RUNS_DIR


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred: torch.Tensor, 
                target: torch.Tensor, 
                num_classes: int = NUM_CLASSES,
                ignore_index: int = 255) -> Tuple[float, List[float]]:
    """
    Compute Intersection over Union (IoU) for each class.
    
    Args:
        pred: Predicted logits or class indices [B, C, H, W] or [B, H, W]
        target: Ground truth masks [B, H, W]
        num_classes: Number of segmentation classes
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        mean_iou: Mean IoU across all classes
        class_ious: List of per-class IoU scores
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    class_ious = []
    
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
            
        pred_inds = pred == class_id
        target_inds = target == class_id
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            class_ious.append(float('nan'))
        else:
            class_ious.append((intersection / union).cpu().numpy())
    
    mean_iou = np.nanmean(class_ious)
    
    return mean_iou, class_ious


def compute_dice(pred: torch.Tensor, 
                 target: torch.Tensor, 
                 num_classes: int = NUM_CLASSES,
                 smooth: float = 1e-6) -> Tuple[float, List[float]]:
    """
    Compute Dice coefficient (F1 Score) for each class.
    
    Args:
        pred: Predicted logits or class indices
        target: Ground truth masks
        num_classes: Number of segmentation classes
        smooth: Smoothing factor for numerical stability
        
    Returns:
        mean_dice: Mean Dice score across all classes
        class_dices: List of per-class Dice scores
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    class_dices = []
    
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        
        class_dices.append(dice_score.cpu().numpy())
    
    mean_dice = np.mean(class_dices)
    
    return mean_dice, class_dices


def compute_pixel_accuracy(pred: torch.Tensor, 
                           target: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    
    Args:
        pred: Predicted logits or class indices
        target: Ground truth masks
        
    Returns:
        Pixel accuracy as a float
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    return (pred == target).float().mean().cpu().numpy()


def evaluate_model(model: torch.nn.Module,
                   backbone: Optional[torch.nn.Module],
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   num_classes: int = NUM_CLASSES,
                   show_progress: bool = True) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Segmentation model
        backbone: Optional backbone model (for DINOv2-style architectures)
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of segmentation classes
        show_progress: Show progress bar
        
    Returns:
        Dictionary with mean_iou, mean_dice, pixel_accuracy, and per-class metrics
    """
    model.eval()
    
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_ious = []
    all_class_dices = []
    
    loader = tqdm(dataloader, desc="Evaluating", leave=False, unit="batch") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if backbone is not None:
                # DINOv2-style: backbone + head
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            else:
                # End-to-end model (SegFormer, DeepLabV3+)
                outputs = model(imgs).logits if hasattr(model(imgs), 'logits') else model(imgs)
            
            # Compute metrics
            labels = labels.squeeze(dim=1).long() if labels.dim() == 4 else labels.long()
            
            iou, class_ious = compute_iou(outputs, labels, num_classes=num_classes)
            dice, class_dices = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)
            
            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_ious.append(class_ious)
            all_class_dices.append(class_dices)
    
    model.train()
    
    # Aggregate results
    results = {
        "mean_iou": np.nanmean(iou_scores),
        "mean_dice": np.nanmean(dice_scores),
        "pixel_accuracy": np.mean(pixel_accuracies),
        "class_ious": np.nanmean(all_class_ious, axis=0).tolist(),
        "class_dices": np.nanmean(all_class_dices, axis=0).tolist(),
    }
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def mask_to_color(mask: np.ndarray, 
                  color_palette: List[List[int]] = COLOR_PALETTE) -> np.ndarray:
    """
    Convert a class mask to a colored RGB image for visualization.
    
    Args:
        mask: Class mask as numpy array [H, W]
        color_palette: List of RGB colors for each class
        
    Returns:
        Colored RGB mask [H, W, 3]
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(color_palette):
        color_mask[mask == class_id] = color
    
    return color_mask


def denormalize_image(img_tensor: torch.Tensor,
                      mean: List[float] = [0.485, 0.456, 0.406],
                      std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        img_tensor: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image as numpy array [H, W, C] in range [0, 255]
    """
    img = img_tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_prediction_comparison(img_tensor: torch.Tensor,
                                gt_mask: torch.Tensor,
                                pred_mask: torch.Tensor,
                                output_path: str,
                                data_id: str,
                                class_names: List[str] = CLASS_NAMES):
    """
    Save a side-by-side comparison of input, ground truth, and prediction.
    
    Args:
        img_tensor: Input image tensor
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        output_path: Path to save the comparison image
        data_id: Image identifier for title
        class_names: List of class names
    """
    # Denormalize image
    img = denormalize_image(img_tensor)
    
    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_per_class_iou_bar(class_ious: List[float],
                           output_path: str,
                           class_names: List[str] = CLASS_NAMES,
                           color_palette: List[List[int]] = COLOR_PALETTE):
    """
    Save a bar chart showing per-class IoU scores.
    
    Args:
        class_ious: List of per-class IoU scores
        output_path: Path to save the chart
        class_names: List of class names
        color_palette: List of RGB colors for each bar
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_ious = [iou if not np.isnan(iou) else 0 for iou in class_ious]
    mean_iou = np.nanmean(class_ious)
    
    # Normalize colors to [0, 1]
    normalized_colors = [c / 255 for c in color_palette]
    
    ax.bar(range(len(class_names)), valid_ious, 
           color=normalized_colors, edgecolor='black')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {mean_iou:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=mean_iou, color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class IoU chart to {output_path}")


# ============================================================================
# Training Logging
# ============================================================================

def save_training_plots(history: Dict[str, List[float]], output_dir: str = RUNS_DIR):
    """
    Save all training metric plots to files.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    
    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    
    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    
    print(f"Saved training plots to {output_dir}")


def save_history_to_file(history: Dict[str, List[float]], 
                         output_dir: str = RUNS_DIR):
    """
    Save training history to a text file.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save the file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    
    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")
        
        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))
    
    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Inference Helpers
# ============================================================================

def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  output_dir: str,
                  save_color_masks: bool = True,
                  num_visualizations: int = 5) -> Dict[str, float]:
    """
    Run inference and save predictions.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for inference
        device: Device to run inference on
        output_dir: Directory to save predictions
        save_color_masks: Whether to save colored visualization masks
        num_visualizations: Number of comparison images to save
        
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    model.eval()
    
    all_class_ious = []
    sample_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Running inference", unit="batch")
        for batch in pbar:
            if len(batch) == 3:
                imgs, labels, data_ids = batch
                has_labels = True
            else:
                imgs, data_ids = batch
                has_labels = False
            
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = model(imgs).logits if hasattr(model(imgs), 'logits') else model(imgs)
            predicted_masks = torch.argmax(outputs, dim=1)
            
            # Calculate metrics if labels available
            if has_labels:
                labels = labels.to(device)
                labels = labels.squeeze(dim=1).long() if labels.dim() == 4 else labels.long()
                iou, class_ious = compute_iou(outputs, labels)
                all_class_ious.append(class_ious)
                pbar.set_postfix(iou=f"{iou:.3f}")
            
            # Save predictions
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]
                
                # Save raw prediction mask
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(output_dir, "masks", f'{base_name}_pred.png'))
                
                # Save colored prediction mask
                if save_color_masks:
                    pred_color = mask_to_color(pred_mask)
                    from PIL import Image
                    Image.fromarray(pred_color).save(
                        os.path.join(output_dir, "masks_color", f'{base_name}_pred_color.png')
                    )
                
                # Save comparison visualization
                if has_labels and sample_count < num_visualizations:
                    save_prediction_comparison(
                        imgs[i], labels[i], predicted_masks[i],
                        os.path.join(output_dir, "comparisons", f'sample_{sample_count}_comparison.png'),
                        data_id
                    )
                
                sample_count += 1
    
    inference_time = (time.time() - start_time) / sample_count
    
    results = {
        "mean_iou": np.nanmean(all_class_ious) if all_class_ious else 0,
        "class_ious": np.nanmean(all_class_ious, axis=0).tolist() if all_class_ious else [],
        "inference_time_ms": inference_time * 1000,
        "num_samples": sample_count,
    }
    
    print(f"\nInference complete! Processed {sample_count} images.")
    print(f"Average inference time: {inference_time * 1000:.2f} ms per image")
    
    return results
