#!/usr/bin/env python3
"""Evaluate the Gabor rain streak detector against a validation set with
ground-truth rain masks, reporting per-image and mean IoU (and P/R/F1).

Usage:
    python scripts/evaluate.py \
        --images data/validation/images \
        --gt-masks data/validation/gt_masks \
        --output results/validation
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gabor_rain_detector import GaborRainDetectorConfig, detect_rain_mask
from src.metrics import iou_score, precision_recall_f1
from src.utils import (
    find_matching_mask,
    list_image_files,
    load_image,
    load_mask,
    overlay_mask_on_image,
    save_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory of validation rain images")
    parser.add_argument("--gt-masks", required=True, help="Directory of ground-truth rain masks")
    parser.add_argument("--output", default=None, help="Optional directory to save predicted masks")
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list_image_files(args.images)

    if not image_paths:
        print(f"No images found at {args.images}")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    cfg = GaborRainDetectorConfig()
    rows = []

    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        gt_path = find_matching_mask(path, args.gt_masks)
        if gt_path is None:
            print(f"[skip] no ground-truth mask found for {name}")
            continue

        bgr = load_image(path)
        gt_mask = load_mask(gt_path)
        pred_mask, _ = detect_rain_mask(bgr, cfg)

        if pred_mask.shape != gt_mask.shape:
            gt_mask_resized = load_mask(gt_path)
            import cv2

            gt_mask = cv2.resize(
                gt_mask_resized,
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        iou = iou_score(pred_mask, gt_mask)
        precision, recall, f1 = precision_recall_f1(pred_mask, gt_mask)
        rows.append((name, iou, precision, recall, f1))
        print(f"{name:30s} IoU={iou:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

        if args.output:
            save_mask(os.path.join(args.output, f"{name}_mask.png"), pred_mask)
            overlay = overlay_mask_on_image(bgr, pred_mask)
            save_mask(os.path.join(args.output, f"{name}_overlay.png"), overlay)

    if not rows:
        print("No image/ground-truth pairs were evaluated.")
        return

    mean_iou = sum(r[1] for r in rows) / len(rows)
    mean_f1 = sum(r[4] for r in rows) / len(rows)
    print("-" * 60)
    print(f"Evaluated {len(rows)} images | mean IoU = {mean_iou:.4f} | mean F1 = {mean_f1:.4f}")

    if args.output:
        csv_path = os.path.join(args.output, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "iou", "precision", "recall", "f1"])
            writer.writerows(rows)
            writer.writerow(["MEAN", mean_iou, "", "", mean_f1])
        print(f"Saved per-image metrics to {csv_path}")


if __name__ == "__main__":
    main()
