#!/usr/bin/env python3
"""Generates a synthetic rainy image with a known ground-truth rain mask, so
the detection pipeline can be exercised end-to-end without needing to
download the course dataset first.

This is only a sanity check / smoke test -- it is NOT a substitute for
running the detector on the real "Single Rain Images" / validation set
supplied in the assignment (see data/README.md).

Usage:
    python scripts/generate_synthetic_demo.py --output data/synthetic_demo
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_background(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Builds a simple, textured, colourful "scene" (gradient sky + noisy
    ground + a few solid shapes) so the background is not just flat colour."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient (top) to ground gradient (bottom).
    for y in range(height):
        t = y / height
        sky_color = np.array([180, 140, 90], dtype=np.float32)   # BGR light blue-ish
        ground_color = np.array([60, 90, 70], dtype=np.float32)  # BGR muted green
        img[y, :, :] = (1 - t) * sky_color + t * ground_color

    # Textured noise to emulate foliage/ground texture (isotropic, not streaky).
    noise = rng.normal(0, 12, size=(height, width, 3)).astype(np.float32)
    img += noise

    # A few solid rectangles/circles to emulate buildings/objects with edges.
    cv2.rectangle(img, (int(0.1 * width), int(0.55 * height)), (int(0.35 * width), int(0.8 * height)), (40, 40, 40), -1)
    cv2.rectangle(img, (int(0.6 * width), int(0.45 * height)), (int(0.8 * width), int(0.8 * height)), (70, 60, 50), -1)
    cv2.circle(img, (int(0.5 * width), int(0.3 * height)), int(0.08 * height), (200, 220, 230), -1)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def add_synthetic_rain(
    background: np.ndarray,
    num_streaks: int = 900,
    angle_deg_spread: float = 12.0,
    seed: int = 1,
):
    """Overlays semi-transparent near-vertical line segments on top of the
    background to emulate rain streaks, and returns the composited rainy
    image together with the exact binary ground-truth mask used to draw
    them."""
    height, width = background.shape[:2]
    rng = np.random.default_rng(seed)

    rain_layer = np.zeros((height, width), dtype=np.float32)
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_streaks):
        x0 = rng.uniform(0, width)
        y0 = rng.uniform(-0.2 * height, height)
        length = rng.uniform(0.03, 0.09) * height
        angle_deg = 90 + rng.uniform(-angle_deg_spread, angle_deg_spread)
        angle_rad = np.deg2rad(angle_deg)
        dx = np.cos(angle_rad) * length
        dy = np.sin(angle_rad) * length
        x1, y1 = x0 + dx, y0 + dy

        thickness = rng.choice([1, 1, 2])
        intensity = rng.uniform(90, 170)

        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)), int(round(y1)))
        cv2.line(rain_layer, pt0, pt1, float(intensity), thickness=thickness, lineType=cv2.LINE_AA)
        cv2.line(gt_mask, pt0, pt1, 255, thickness=thickness, lineType=8)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

    bg_f = background.astype(np.float32)
    rain_f = np.repeat(rain_layer[:, :, None], 3, axis=2)
    rainy = np.clip(bg_f + rain_f, 0, 255).astype(np.uint8)

    return rainy, gt_mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic_demo")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-streaks", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = os.path.join(args.output, "images")
    masks_dir = os.path.join(args.output, "gt_masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    background = make_background(args.height, args.width, seed=args.seed)
    rainy, gt_mask = add_synthetic_rain(
        background, num_streaks=args.num_streaks, seed=args.seed + 1
    )

    cv2.imwrite(os.path.join(images_dir, "synthetic_rain_01.png"), rainy)
    cv2.imwrite(os.path.join(masks_dir, "synthetic_rain_01.png"), gt_mask)
    cv2.imwrite(os.path.join(args.output, "clean_background.png"), background)

    print(f"Wrote synthetic rainy image + ground-truth mask to {args.output}/")


if __name__ == "__main__":
    main()
