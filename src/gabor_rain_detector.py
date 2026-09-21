"""
Gabor-filter-bank based rain streak detection.

Rain streaks in a single image are thin, elongated, roughly-vertical bright
structures superimposed on the underlying scene. The pipeline below detects
them without any learning-based component:

1.  High-frequency ("detail") layer extraction
    The scene content (buildings, sky, foliage, ...) is mostly low frequency
    compared to rain streaks, so a large-kernel median filter is used to
    estimate a smooth base layer. Subtracting it from the image isolates the
    high-frequency detail layer, which contains rain streaks together with
    other high-frequency content such as object edges and texture.

2.  Oriented Gabor filter bank
    A bank of Gabor kernels tuned to orientations close to vertical (rain
    rarely falls perfectly straight down, so a small angular spread around
    the vertical is used) and to several wavelengths (to cover streaks of
    different apparent widths/densities) is convolved with the detail layer.
    Taking the maximum response across the bank at every pixel produces a
    "streak-likelihood" map: pixels that align with one of the near-vertical
    oriented filters respond strongly, while isotropic texture/noise does not.

3.  Chroma suppression
    Rain streaks are close to achromatic (low saturation) because they are
    formed by scattered/reflected light. The streak-likelihood map is
    down-weighted in highly saturated regions to suppress colourful texture
    edges (leaves, painted surfaces, etc.) that would otherwise survive the
    orientation filter.

4.  Thresholding + shape-based cleanup
    The weighted response map is binarised with Otsu's method, then cleaned
    with morphology and a connected-component elongation/orientation filter
    that discards blobs that are not thin and near-vertical, which further
    reduces false positives from scene edges that happen to share the
    dominant orientation.

The result is a binary rain mask the same size as the input image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class GaborRainDetectorConfig:
    # --- detail layer extraction ---
    median_ksize: int = 25          # base-layer median filter kernel size (odd)

    # --- Gabor filter bank ---
    kernel_size: int = 15           # spatial support of each Gabor kernel (odd)
    orientation_spread_deg: float = 30.0   # max deviation from vertical (+/-)
    orientation_step_deg: float = 5.0      # angular resolution of the bank
    wavelengths: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)  # streak "widths"
    sigma: float = 0.9              # Gaussian envelope std (keeps streaks thin)
    gamma: float = 0.7              # spatial aspect ratio (< 1 => elongated)
    psi: float = 0.0                # phase offset

    # --- chroma suppression ---
    use_saturation_suppression: bool = True
    saturation_weight: float = 1.0  # 0 = no suppression, 1 = full suppression

    # --- thresholding ---
    threshold_offset: int = 0       # added to the Otsu threshold (can tune)

    # --- morphology / shape filtering ---
    morph_open_ksize: int = 0
    morph_close_ksize: Tuple[int, int] = (1, 3)  # vertical-biased closing
    min_area: int = 6               # discard components smaller than this
    min_elongation: float = 1.8     # bbox height / width must exceed this
    max_orientation_dev_deg: float = 35.0  # component angle vs. vertical


def _vertical_orientations_deg(cfg: GaborRainDetectorConfig) -> List[float]:
    n_steps = int(cfg.orientation_spread_deg / cfg.orientation_step_deg)
    offsets = [i * cfg.orientation_step_deg for i in range(-n_steps, n_steps + 1)]
    return [90.0 + off for off in offsets]  # 90 deg = vertical in OpenCV convention


def build_gabor_filter_bank(cfg: GaborRainDetectorConfig) -> List[np.ndarray]:
    """Builds a bank of Gabor kernels covering near-vertical orientations and
    a small set of wavelengths, so both the streak angle and apparent width
    of rain are covered."""
    ksize = (cfg.kernel_size, cfg.kernel_size)
    kernels = []
    for theta_deg in _vertical_orientations_deg(cfg):
        theta = np.deg2rad(theta_deg)
        for lambd in cfg.wavelengths:
            kernel = cv2.getGaborKernel(
                ksize, cfg.sigma, theta, lambd, cfg.gamma, cfg.psi, ktype=cv2.CV_32F
            )
            kernel -= kernel.mean()  # zero-DC response -> ignore flat regions
            kernels.append(kernel)
    return kernels


def extract_detail_layer(gray: np.ndarray, cfg: GaborRainDetectorConfig) -> np.ndarray:
    """Removes the smooth scene content, keeping high-frequency detail
    (rain streaks + edges/texture)."""
    gray_f = gray.astype(np.float32)
    base = cv2.medianBlur(gray, cfg.median_ksize).astype(np.float32)
    detail = gray_f - base
    return detail


def gabor_response_map(detail: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
    """Convolves the detail layer with every kernel in the bank and returns
    the per-pixel maximum absolute response (i.e. the best-matching
    orientation/wavelength wins at each pixel)."""
    response = np.zeros_like(detail, dtype=np.float32)
    for kernel in kernels:
        filtered = cv2.filter2D(detail, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        response = np.maximum(response, np.abs(filtered))
    return response


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)


def apply_saturation_suppression(
    response: np.ndarray, bgr: np.ndarray, cfg: GaborRainDetectorConfig
) -> np.ndarray:
    """Down-weights the response map wherever the pixel is strongly
    saturated/colourful, since rain streaks are close to achromatic."""
    if not cfg.use_saturation_suppression:
        return response
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    weight = 1.0 - cfg.saturation_weight * sat
    weight = np.clip(weight, 0.0, 1.0)
    return response * weight


def _filter_components_by_shape(
    mask: np.ndarray, cfg: GaborRainDetectorConfig
) -> np.ndarray:
    """Removes connected components that are not thin and roughly vertical,
    which suppresses blob-like false positives (scene edges, texture) that
    survived thresholding."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < cfg.min_area:
            continue
        elongation = h / max(w, 1)
        component_mask = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
        if component_mask.sum() >= 5:
            ys, xs = np.nonzero(component_mask)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            (_, _), (_, _), angle = cv2.minAreaRect(pts)
            # cv2.minAreaRect angle is in [-90, 0); convert to deviation from vertical
            dev_from_vertical = min(abs(angle), abs(angle + 90))
        else:
            dev_from_vertical = 0.0

        if elongation >= cfg.min_elongation or dev_from_vertical <= cfg.max_orientation_dev_deg:
            cleaned[y : y + h, x : x + w][component_mask == 1] = 255
    return cleaned


def detect_rain_mask(
    bgr: np.ndarray, cfg: GaborRainDetectorConfig | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the full Gabor-based rain streak detection pipeline.

    Parameters
    ----------
    bgr : np.ndarray
        Input image in BGR order (as read by cv2.imread), uint8.
    cfg : GaborRainDetectorConfig, optional
        Pipeline hyperparameters. Defaults are used if omitted.

    Returns
    -------
    mask : np.ndarray
        Binary rain mask (uint8, values in {0, 255}), same H x W as input.
    response_map : np.ndarray
        The (uint8, normalized) Gabor streak-likelihood map before
        thresholding -- useful for visualization/debugging.
    """
    if cfg is None:
        cfg = GaborRainDetectorConfig()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detail = extract_detail_layer(gray, cfg)

    kernels = build_gabor_filter_bank(cfg)
    response = gabor_response_map(detail, kernels)
    response = apply_saturation_suppression(response, bgr, cfg)

    response_u8 = _normalize_to_uint8(response)

    otsu_thresh, mask = cv2.threshold(
        response_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if cfg.threshold_offset:
        _, mask = cv2.threshold(
            response_u8,
            min(255, max(0, otsu_thresh + cfg.threshold_offset)),
            255,
            cv2.THRESH_BINARY,
        )

    if cfg.morph_open_ksize > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open_ksize, cfg.morph_open_ksize)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    close_w, close_h = cfg.morph_close_ksize
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    mask = _filter_components_by_shape(mask, cfg)

    return mask, response_u8
