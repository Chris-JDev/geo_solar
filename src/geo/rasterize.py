"""Rasterization helpers.

Rasterization means converting vector geometry (polygons defined by vertices) into a
pixel/grid mask. Here we create fixed-size masks (e.g. 256×256) that can be used as
conditioning tensors for ML models.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def normalize_to_canvas(
    points_xy: Sequence[Tuple[float, float]],
    width: int,
    height: int,
) -> List[Tuple[float, float]]:
    """Normalize pixel coordinates into [0,1] range.

    Args:
        points_xy: list of (x, y) in original image pixels.
        width, height: original image dimensions.

    Returns:
        List of (x_norm, y_norm) where each coordinate is in [0,1].
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    out = []
    for x, y in points_xy:
        out.append((float(x) / float(width), float(y) / float(height)))
    return out


def _points_to_out_coords(
    points_xy: Sequence[Tuple[float, float]],
    img_w: int,
    img_h: int,
    out_size: int,
) -> np.ndarray:
    """Convert image pixel coords to output-grid integer coordinates."""

    if out_size <= 0:
        raise ValueError("out_size must be positive")
    if img_w <= 0 or img_h <= 0:
        raise ValueError("img_w and img_h must be positive")

    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be a list of (x, y)")

    # Map x in [0..img_w] to [0..out_size-1], same for y.
    sx = (out_size - 1) / float(img_w)
    sy = (out_size - 1) / float(img_h)
    pts[:, 0] = np.clip(pts[:, 0] * sx, 0, out_size - 1)
    pts[:, 1] = np.clip(pts[:, 1] * sy, 0, out_size - 1)
    return np.round(pts).astype(np.int32)


def polygon_to_mask(
    points_xy: Sequence[Tuple[float, float]],
    img_w: int,
    img_h: int,
    out_size: int = 256,
) -> np.ndarray:
    """Rasterize a polygon into an (out_size, out_size) mask.

    Returns:
        np.uint8 mask where 1 indicates inside the polygon, 0 outside.
    """

    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    if not points_xy or len(points_xy) < 3:
        return mask
    pts = _points_to_out_coords(points_xy, img_w, img_h, out_size)
    # cv2.fillPoly expects shape (n_points, 1, 2)
    cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 1)
    return mask


def multi_polygon_to_mask(
    list_of_polys: Iterable[Sequence[Tuple[float, float]]],
    img_w: int,
    img_h: int,
    out_size: int = 256,
) -> np.ndarray:
    """Rasterize multiple polygons into one combined mask."""

    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    for poly in list_of_polys or []:
        if not poly or len(poly) < 3:
            continue
        mask = np.maximum(mask, polygon_to_mask(poly, img_w, img_h, out_size=out_size))
    return mask
