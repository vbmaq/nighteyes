from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PupilRoiResult:
    """Result for pupil ROI cropping."""
    roi_img: np.ndarray
    roi_mask: Optional[np.ndarray]
    offset_x: int
    offset_y: int
    pad_left: int
    pad_top: int
    valid: bool


@dataclass
class PupilRoiDecision:
    """Decision for pupil ROI center resolution under fail policies."""
    center: Optional[Tuple[float, float]]
    radius: Optional[float]
    action: str
    used_last_good: bool


def _border_mode(pad_mode: str) -> int:
    if pad_mode == "reflect":
        return cv2.BORDER_REFLECT_101
    if pad_mode == "edge":
        return cv2.BORDER_REPLICATE
    if pad_mode == "constant":
        return cv2.BORDER_CONSTANT
    raise ValueError(f"Unsupported pad_mode: {pad_mode}")


def compute_pupil_roi(
    img: np.ndarray,
    center: Tuple[float, float],
    size: int = 80,
    pad_mode: str = "reflect",
    pad_value: int = 0,
    mask: Optional[np.ndarray] = None,
) -> PupilRoiResult:
    """
    Crop a square ROI centered at center (cx, cy). Returns ROI and padding metadata.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if img is None:
        raise ValueError("img cannot be None")

    cx, cy = center
    cx_i = int(round(float(cx)))
    cy_i = int(round(float(cy)))
    half = int(size) // 2

    x0 = cx_i - half
    y0 = cy_i - half
    x1 = x0 + int(size)
    y1 = y0 + int(size)

    H, W = img.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    border_mode = _border_mode(pad_mode)
    if pad_left or pad_right or pad_top or pad_bottom:
        if border_mode == cv2.BORDER_CONSTANT:
            img_pad = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right, border_mode, value=pad_value
            )
        else:
            img_pad = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right, border_mode
            )
    else:
        img_pad = img

    mask_pad = None
    if mask is not None:
        if border_mode == cv2.BORDER_CONSTANT:
            mask_pad = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right, border_mode, value=pad_value
            )
        else:
            mask_pad = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right, border_mode
            )

    x0p = x0 + pad_left
    y0p = y0 + pad_top
    roi_img = img_pad[y0p:y0p + int(size), x0p:x0p + int(size)]
    roi_mask = None if mask_pad is None else mask_pad[y0p:y0p + int(size), x0p:x0p + int(size)]

    return PupilRoiResult(
        roi_img=roi_img,
        roi_mask=roi_mask,
        offset_x=int(x0),
        offset_y=int(y0),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        valid=True,
    )


def map_points_to_full(points: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    if points is None:
        return points
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape((-1, 2))
    arr = arr.copy()
    arr[:, 0] += float(offset_x)
    arr[:, 1] += float(offset_y)
    return arr


def resolve_pupil_roi_center(
    center: Optional[Tuple[float, float]],
    radius: Optional[float],
    W: int,
    H: int,
    policy: str,
    last_good: Optional[Tuple[float, float, float]],
) -> PupilRoiDecision:
    def _valid(c):
        if c is None:
            return False
        cx, cy = c
        if not np.isfinite(cx) or not np.isfinite(cy):
            return False
        if cx < 0 or cy < 0 or cx >= W or cy >= H:
            return False
        return True

    if _valid(center):
        return PupilRoiDecision(center=center, radius=radius, action="use", used_last_good=False)

    if policy == "full_frame":
        return PupilRoiDecision(center=None, radius=None, action="full_frame", used_last_good=False)
    if policy == "skip":
        return PupilRoiDecision(center=None, radius=None, action="skip", used_last_good=False)
    if policy == "last_good":
        if last_good is not None and _valid((last_good[0], last_good[1])):
            return PupilRoiDecision(
                center=(float(last_good[0]), float(last_good[1])),
                radius=float(last_good[2]) if last_good[2] is not None else None,
                action="use",
                used_last_good=True,
            )
        return PupilRoiDecision(center=None, radius=None, action="skip", used_last_good=False)
    raise ValueError(f"Unknown fail policy: {policy}")
