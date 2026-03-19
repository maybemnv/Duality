"""
Visualization script for segmentation masks.

Colorizes raw segmentation masks using the class palette defined in config.py.

Usage:
    python visualize.py --input_dir <path_to_masks> [--output_dir <path>]
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from config import COLOR_PALETTE, CLASS_NAMES


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(COLOR_PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def main():
    parser = argparse.ArgumentParser(description="Colorize segmentation masks")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: input_dir/colorized)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "colorized"
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions)

    print(f"Found {len(files)} mask files")
    print(f"Output: {output_dir}")

    for f in files:
        mask = np.array(Image.open(f))
        colored = colorize_mask(mask)
        Image.fromarray(colored).save(output_dir / f"{f.stem}_color.png")
        print(f"  {f.name} -> {f.stem}_color.png")

    print(f"\nDone. Class legend:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name} -> RGB{tuple(COLOR_PALETTE[i])}")


if __name__ == "__main__":
    main()