#!/usr/bin/env python3
"""Run Gabor-based rain streak detection on a single image or a directory of
images, saving the binary rain mask (and an optional overlay) for each.

Usage:
    python scripts/detect.py --input data/raw --output results/raw
    python scripts/detect.py --input path/to/one_image.jpg --output results/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gabor_rain_detector import GaborRainDetectorConfig, detect_rain_mask
from src.utils import list_image_files, load_image, overlay_mask_on_image, save_mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", required=True, help="Directory to write masks/overlays to")
    parser.add_argument(
        "--overlay", action="store_true", help="Also save the mask overlaid on the input image"
    )
    parser.add_argument(
        "--save-response",
        action="store_true",
        help="Also save the raw Gabor response map (before thresholding)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.input):
        image_paths = list_image_files(args.input)
    else:
        image_paths = [args.input]

    if not image_paths:
        print(f"No images found at {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    cfg = GaborRainDetectorConfig()

    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = load_image(path)
        mask, response = detect_rain_mask(bgr, cfg)

        save_mask(os.path.join(args.output, f"{name}_mask.png"), mask)
        if args.overlay:
            overlay = overlay_mask_on_image(bgr, mask)
            save_mask(os.path.join(args.output, f"{name}_overlay.png"), overlay)
        if args.save_response:
            save_mask(os.path.join(args.output, f"{name}_response.png"), response)

        print(f"[done] {path} -> {name}_mask.png")


if __name__ == "__main__":
    main()
