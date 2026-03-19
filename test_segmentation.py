"""
Test/Inference script for Duality AI Offroad Segmentation Challenge
PRD Version: SegFormer-B2 | RTX 3070 Laptop 8GB VRAM

Runs inference on test images with:
- Test-Time Augmentation (TTA) for +2-5 IoU points
- Multi-scale averaging
- Horizontal flip averaging

PRD: testImages folder is strictly off-limits for training.
Only used for final inference after training is complete.

Usage:
    python test_segmentation.py [--model_path checkpoints/best_model.pth] [--use_tta]

Stack: Python · PyTorch · HuggingFace Transformers · albumentations · Conda
"""

import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    NUM_CLASSES,
    CLASS_NAMES,
    COLOR_PALETTE,
    TEST_DIR,
    PREDICTIONS_DIR,
    DEVICE,
    setup_directories,
    IMAGE_SIZE,
    USE_TTA,
    INFERENCE_TIME_TARGET,
    validate_hardware,
)
from dataset import create_test_dataloader
from model import load_model, load_checkpoint
from augmentations import apply_tta_to_image, get_tta_transforms
from utils import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    mask_to_color,
    save_prediction_comparison,
    save_per_class_iou_bar,
)


def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  output_dir: str,
                  use_tta: bool = True,
                  num_visualizations: int = 5) -> dict:
    """
    Run inference with optional TTA and save predictions.
    
    PRD: Test-Time Augmentation (TTA) consistently adds 2-5 IoU points
    with no additional training. Average predictions from:
    - Original image
    - Horizontally flipped image (flip prediction back)
    - Slightly scaled images (0.9x, 1.1x)
    
    Args:
        model: Trained SegFormer model
        dataloader: Test dataloader
        device: Device to run inference on
        output_dir: Directory to save predictions
        use_tta: Whether to use test-time augmentation
        num_visualizations: Number of comparison images to save
        
    Returns:
        Dictionary with inference statistics
    """
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    model.eval()
    
    sample_count = 0
    start_time = time.time()
    all_class_ious = []
    
    print(f"\nRunning inference{' with TTA' if use_tta else ''}...")
    print(f"Target inference time: < {INFERENCE_TIME_TARGET}ms per image")
    
    pbar = tqdm(dataloader, desc="Processing", unit="batch")
    
    for batch in pbar:
        if len(batch) == 3:
            imgs, labels, data_ids = batch
            has_labels = True
        else:
            imgs, data_ids = batch
            has_labels = False
        
        # Process each image in batch
        for i in range(imgs.shape[0]):
            data_id = data_ids[i]
            base_name = os.path.splitext(data_id)[0]
            batch_start = time.time()
            
            # Get prediction
            if use_tta:
                # TTA: average multiple augmented predictions
                pred_logits = apply_tta_to_image(
                    imgs[i], model, device,
                    use_flip=True, use_scales=True
                )
                pred_mask = torch.argmax(pred_logits, dim=0)
            else:
                # Single forward pass
                with torch.no_grad():
                    img_batch = imgs[i:i+1].to(device)
                    outputs = model(img_batch)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    pred_mask = torch.argmax(logits, dim=1)[0]
            
            inference_time = time.time() - batch_start
            
            # Convert to numpy
            pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            
            # Save raw prediction mask (class IDs 0-9)
            pred_img = Image.fromarray(pred_mask_np)
            pred_img.save(os.path.join(output_dir, "masks", f'{base_name}_pred.png'))
            
            # Save colored prediction mask (RGB visualization)
            pred_color = mask_to_color(pred_mask_np)
            Image.fromarray(pred_color).save(
                os.path.join(output_dir, "masks_color", f'{base_name}_pred_color.png')
            )
            
            # Calculate metrics if labels available (for validation)
            if has_labels:
                with torch.no_grad():
                    img_batch = imgs[i:i+1].to(device)
                    outputs = model(img_batch)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    label_batch = labels[i:i+1].to(device).long()
                    iou, class_ious = compute_iou(logits, label_batch.squeeze(1), num_classes=NUM_CLASSES)
                    all_class_ious.append(class_ious)
                pbar.set_postfix(iou=f"{iou:.3f}", time=f"{inference_time*1000:.1f}ms")
            else:
                pbar.set_postfix(time=f"{inference_time*1000:.1f}ms")
            
            sample_count += 1
    
    # Calculate average inference time
    total_time = time.time() - start_time
    avg_inference_time = total_time / sample_count
    
    results = {
        "num_samples": sample_count,
        "total_time_sec": total_time,
        "avg_inference_time_ms": avg_inference_time * 1000,
        "use_tta": use_tta,
        "target_met": avg_inference_time * 1000 < INFERENCE_TIME_TARGET,
    }
    
    # Calculate mean IoU if labels available
    if all_class_ious:
        results["mean_iou"] = np.nanmean([np.nanmean(c) for c in all_class_ious])
        results["class_ious"] = np.nanmean(all_class_ious, axis=0).tolist()
    else:
        results["mean_iou"] = None
        results["class_ious"] = None
    
    return results


def save_inference_summary(results: dict, output_dir: str):
    """
    Save inference summary file (PRD submission requirement).
    
    Args:
        results: Inference results dictionary
        output_dir: Directory to save summary
    """
    filepath = os.path.join(output_dir, "inference_summary.txt")
    
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("INFERENCE RESULTS — Duality AI Offroad Segmentation\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Model: SegFormer-B2\n")
        f.write(f"  Pretrained: nvidia/segformer-b2-finetuned-ade-512-512\n")
        f.write(f"  Number of classes: {NUM_CLASSES}\n")
        f.write(f"  Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}\n")
        f.write(f"  Test-time augmentation: {results['use_tta']}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("PERFORMANCE\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total images processed: {results['num_samples']}\n")
        f.write(f"Total time: {results['total_time_sec']:.2f} seconds\n")
        f.write(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms per image\n")
        f.write(f"Target (< {INFERENCE_TIME_TARGET}ms): {'MET' if results['target_met'] else 'NOT MET'}\n")
        f.write("\n")
        
        if results['mean_iou'] is not None:
            f.write("=" * 60 + "\n")
            f.write("IOU METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
            f.write("\n")
            f.write("Per-Class IoU:\n")
            f.write("-" * 40 + "\n")
            for i, (name, iou) in enumerate(zip(CLASS_NAMES, results['class_ious'])):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
                f.write(f"  {i}: {name:15} = {iou_str}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("CLASS MAPPING\n")
        f.write("=" * 60 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")
    
    print(f"\nSaved inference summary to {filepath}")


def save_per_class_results(class_ious: list, output_dir: str):
    """
    Save per-class IoU bar chart.
    
    Args:
        class_ious: List of per-class IoU scores
        output_dir: Directory to save chart
    """
    save_per_class_iou_bar(class_ious, os.path.join(output_dir, "per_class_iou.png"))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference on test images")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth"),
                        help="Path to trained model weights")
    parser.add_argument("--data_dir", type=str, default=TEST_DIR,
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default=PREDICTIONS_DIR,
                        help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--use_tta", action="store_true", default=USE_TTA,
                        help="Use test-time augmentation")
    parser.add_argument("--no_tta", action="store_false", dest="use_tta",
                        help="Disable test-time augmentation")
    parser.add_argument("--num_vis", type=int, default=5,
                        help="Number of visualization samples to save")
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Validate hardware
    validate_hardware()
    
    # Device
    device = DEVICE
    print(f"\nUsing device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"TTA enabled: {args.use_tta}")
    
    # Create test dataloader
    print("\n" + "=" * 60)
    print("LOADING TEST DATASET")
    print("=" * 60)
    test_loader = create_test_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
    )
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model = load_model(num_classes=NUM_CLASSES, device=device)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        print(f"\nLoading weights from {args.model_path}...")
        load_checkpoint(args.model_path, model, device=device)
    else:
        print(f"\nWARNING: Model path {args.model_path} not found.")
        print("Using random initialization (for testing only).")
    
    # Run inference
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    results = run_inference_with_tta(
        model,
        test_loader,
        device,
        args.output_dir,
        use_tta=args.use_tta,
        num_visualizations=args.num_vis,
    )
    
    # Save summary
    save_inference_summary(results, args.output_dir)
    
    # Save per-class IoU chart if available
    if results.get('class_ious') is not None:
        save_per_class_results(results['class_ious'], args.output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("OUTPUTS")
    print("=" * 60)
    print(f"Predictions saved to {args.output_dir}/")
    print(f"  - masks/              : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/        : Colored prediction masks (RGB)")
    print(f"  - comparisons/        : Side-by-side comparisons ({args.num_vis} samples)")
    print(f"  - inference_summary.txt")
    print(f"  - per_class_iou.png   : Per-class IoU bar chart")
    
    if results['mean_iou'] is not None:
        print(f"\nMean IoU: {results['mean_iou']:.4f}")
    
    print(f"\nInference time: {results['avg_inference_time_ms']:.2f} ms per image")
    print(f"Target (< {INFERENCE_TIME_TARGET}ms): {'✓ MET' if results['target_met'] else '✗ NOT MET'}")
    print("=" * 60)


def run_inference_with_tta(model, test_loader, device, output_dir, use_tta, num_visualizations):
    """Wrapper for run_inference to maintain compatibility."""
    return run_inference(model, test_loader, device, output_dir, use_tta, num_visualizations)


if __name__ == "__main__":
    main()
