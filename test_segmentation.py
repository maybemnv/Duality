"""
Test/Inference script for Duality AI Offroad Segmentation Challenge

Runs inference on test images and saves predictions in the required format.
Supports test-time augmentation (TTA) for improved accuracy.

Usage:
    python test_segmentation.py [--model_path checkpoints/best_model.pth] [--data_dir Data/Offroad_Segmentation_testImages]
"""

import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from config import (
    NUM_CLASSES,
    CLASS_NAMES,
    COLOR_PALETTE,
    TEST_DIR,
    PREDICTIONS_DIR,
    DEVICE,
    setup_directories,
    IMAGE_SIZE,
)
from dataset import create_test_dataloader, get_test_transform, get_val_transform
from model import load_model, load_checkpoint
from utils import (
    compute_iou,
    mask_to_color,
    denormalize_image,
    save_prediction_comparison,
    save_per_class_iou_bar,
    run_inference,
)


def apply_tta(model: torch.nn.Module,
              img: torch.Tensor,
              device: torch.device,
              flip: bool = True) -> torch.Tensor:
    """
    Apply Test-Time Augmentation for improved predictions.
    
    Args:
        model: Trained segmentation model
        img: Input image tensor [C, H, W]
        device: Device to run inference on
        flip: Whether to use horizontal flip augmentation
        
    Returns:
        Averaged prediction logits [NUM_CLASSES, H, W]
    """
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(device)
        outputs = model(img_batch)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        pred = F.softmax(logits, dim=1)
    
    if flip:
        # Flipped prediction
        flipped_img = torch.flip(img, dims=[2])
        with torch.no_grad():
            flipped_batch = flipped_img.unsqueeze(0).to(device)
            outputs = model(flipped_batch)
            flipped_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            flipped_pred = F.softmax(flipped_logits, dim=1)
            
            # Flip back and average
            flipped_pred = torch.flip(flipped_pred, dims=[3])
            pred = (pred + flipped_pred) / 2
    
    return pred.squeeze(0)


def run_inference_with_tta(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: torch.device,
                           output_dir: str,
                           use_tta: bool = True,
                           num_visualizations: int = 5) -> dict:
    """
    Run inference with optional TTA and save predictions.
    
    Args:
        model: Trained segmentation model
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
    
    print(f"\nRunning inference{' with TTA' if use_tta else ''}...")
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
            
            # Get prediction
            if use_tta:
                pred_logits = apply_tta(model, imgs[i], device, flip=True)
                pred_mask = torch.argmax(pred_logits, dim=0)
            else:
                with torch.no_grad():
                    img_batch = imgs[i:i+1].to(device)
                    outputs = model(img_batch)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    pred_mask = torch.argmax(logits, dim=1)[0]
            
            # Convert to numpy
            pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            
            # Save raw prediction mask (class IDs 0-9)
            pred_img = Image.fromarray(pred_mask_np)
            pred_img.save(os.path.join(output_dir, "masks", f'{base_name}_pred.png'))
            
            # Save colored prediction mask
            pred_color = mask_to_color(pred_mask_np)
            Image.fromarray(pred_color).save(
                os.path.join(output_dir, "masks_color", f'{base_name}_pred_color.png')
            )
            
            # Save comparison if ground truth available
            if has_labels and sample_count < num_visualizations:
                save_prediction_comparison(
                    imgs[i], 
                    labels[i] if labels.dim() == 3 else labels[i, 0],
                    pred_mask,
                    os.path.join(output_dir, "comparisons", f'sample_{sample_count}_comparison.png'),
                    data_id
                )
            
            sample_count += 1
            pbar.set_description(f"Processing: {base_name[:30]}")
    
    # Calculate inference time
    inference_time = (time.time() - start_time) / sample_count
    
    results = {
        "num_samples": sample_count,
        "inference_time_ms": inference_time * 1000,
        "use_tta": use_tta,
    }
    
    print(f"\nInference complete!")
    print(f"  Processed: {sample_count} images")
    print(f"  Average time: {inference_time * 1000:.2f} ms per image")
    print(f"  TTA enabled: {use_tta}")
    
    return results


def save_submission_file(predictions_dir: str, results: dict):
    """
    Save a summary file for submission.
    
    Args:
        predictions_dir: Directory containing predictions
        results: Inference results dictionary
    """
    filepath = os.path.join(predictions_dir, "inference_summary.txt")
    
    with open(filepath, 'w') as f:
        f.write("INFERENCE RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: SegFormer-B2\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n")
        f.write(f"Test-time augmentation: {results['use_tta']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images processed: {results['num_samples']}\n")
        f.write(f"Average inference time: {results['inference_time_ms']:.2f} ms\n")
        f.write("=" * 50 + "\n\n")
        f.write("Class mapping:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")
    
    print(f"Saved inference summary to {filepath}")


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
    parser.add_argument("--use_tta", action="store_true", default=True,
                        help="Use test-time augmentation")
    parser.add_argument("--no_tta", action="store_false", dest="use_tta",
                        help="Disable test-time augmentation")
    parser.add_argument("--model_type", type=str, default="segformer_b2",
                        choices=["segformer_b2", "deeplabv3", "unet"],
                        help="Model architecture")
    parser.add_argument("--num_vis", type=int, default=5,
                        help="Number of visualization samples to save")
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    device = DEVICE
    
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create test dataloader
    print("\nLoading test dataset...")
    test_loader = create_test_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Load model
    print(f"\nLoading {args.model_type} model...")
    model = load_model(args.model_type, num_classes=NUM_CLASSES, device=device)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, val_iou: {checkpoint['val_iou']:.4f})")
    else:
        print(f"Warning: Model path {args.model_path} not found. Using random initialization.")
    
    # Run inference
    results = run_inference_with_tta(
        model, 
        test_loader, 
        device, 
        args.output_dir,
        use_tta=args.use_tta,
        num_visualizations=args.num_vis
    )
    
    # Save submission file
    save_submission_file(args.output_dir, results)
    
    print(f"\n" + "=" * 80)
    print("OUTPUTS")
    print("=" * 80)
    print(f"Predictions saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB visualization)")
    print(f"  - comparisons/     : Side-by-side comparisons ({args.num_vis} samples)")
    print(f"  - inference_summary.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
