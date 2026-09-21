"""I/O and visualization helpers."""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_image_files(directory: str) -> List[str]:
    """Returns a sorted list of image file paths inside `directory`."""
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(files)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def save_mask(path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, mask)


def overlay_mask_on_image(
    bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.6
) -> np.ndarray:
    """Overlays a binary mask on top of an image in the given BGR colour."""
    overlay = bgr.copy()
    color_layer = np.zeros_like(bgr)
    color_layer[:] = color
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(bgr, 1 - alpha, color_layer, alpha, 0)[mask_bool]
    return overlay


def find_matching_mask(image_path: str, masks_dir: str) -> str | None:
    """Finds the ground-truth mask file for a given image, matching by
    filename stem (extension-agnostic)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for candidate in list_image_files(masks_dir):
        if os.path.splitext(os.path.basename(candidate))[0] == stem:
            return candidate
    return None
