"""Synthetic dataset generator for cVAE training.

We generate training pairs:
  - conditioning masks: boundary_mask + keepout_mask  -> cond.npy (2, H, W)
  - target layout mask: row stripes inside boundary avoiding keepouts -> target.npy (1, H, W)

Everything is created in *grid space* (0..size-1). The goal is to give a cVAE fast,
cheap data that teaches: "place rows inside boundary but not in keepouts".

Output folder structure:
  data/synthetic/sample_000001/cond.npy
  data/synthetic/sample_000001/target.npy
  data/synthetic/sample_000001/meta.json

Deterministic: controlled by --seed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _fill_poly(mask: np.ndarray, pts: List[Tuple[float, float]], value: int) -> None:
    if len(pts) < 3:
        return
    arr = np.asarray(pts, dtype=np.float32)
    arr = np.round(arr).astype(np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, mask.shape[1] - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, mask.shape[0] - 1)
    cv2.fillPoly(mask, [arr.reshape((-1, 1, 2))], int(value))


def _random_boundary_polygon(rng: np.random.Generator, size: int) -> List[Tuple[float, float]]:
    """Generate a convex-ish polygon by starting from a rectangle and cutting corners."""

    pad = int(rng.integers(low=int(size * 0.05), high=int(size * 0.15) + 1))
    x0, y0 = pad, pad
    x1, y1 = size - 1 - pad, size - 1 - pad

    # Random cut amounts for corners (0..~25% of rectangle side)
    cut = int(rng.integers(low=int(size * 0.02), high=int(size * 0.12) + 1))
    c1 = int(rng.integers(0, cut + 1))
    c2 = int(rng.integers(0, cut + 1))
    c3 = int(rng.integers(0, cut + 1))
    c4 = int(rng.integers(0, cut + 1))

    pts = [
        (x0 + c1, y0),
        (x1 - c2, y0),
        (x1, y0 + c2),
        (x1, y1 - c3),
        (x1 - c3, y1),
        (x0 + c4, y1),
        (x0, y1 - c4),
        (x0, y0 + c1),
    ]

    # Remove near-duplicate consecutive points
    cleaned = []
    for p in pts:
        if not cleaned:
            cleaned.append(p)
        else:
            if (abs(cleaned[-1][0] - p[0]) + abs(cleaned[-1][1] - p[1])) >= 1:
                cleaned.append(p)
    return cleaned


def _random_keepout_polygon(rng: np.random.Generator, size: int) -> List[Tuple[float, float]]:
    """Small random quadrilateral (rough rectangle) in grid space."""

    w = int(rng.integers(low=int(size * 0.05), high=int(size * 0.15) + 1))
    h = int(rng.integers(low=int(size * 0.05), high=int(size * 0.15) + 1))
    cx = int(rng.integers(low=int(size * 0.2), high=int(size * 0.8) + 1))
    cy = int(rng.integers(low=int(size * 0.2), high=int(size * 0.8) + 1))
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = cx + w // 2, cy + h // 2

    # Add small jitter to corners.
    j = int(max(1, min(w, h) * 0.1))
    pts = [
        (x0 + int(rng.integers(-j, j + 1)), y0 + int(rng.integers(-j, j + 1))),
        (x1 + int(rng.integers(-j, j + 1)), y0 + int(rng.integers(-j, j + 1))),
        (x1 + int(rng.integers(-j, j + 1)), y1 + int(rng.integers(-j, j + 1))),
        (x0 + int(rng.integers(-j, j + 1)), y1 + int(rng.integers(-j, j + 1))),
    ]
    return pts


def _generate_row_layout(
    rng: np.random.Generator,
    boundary_mask: np.ndarray,
    keepout_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Generate a simple baseline layout mask with horizontal stripes.

    We draw parallel row bands every `pitch` pixels.
    Then clip: rows = stripes AND boundary AND (NOT keepouts).
    """

    size = boundary_mask.shape[0]
    pitch = int(rng.integers(6, 15))  # 6..14 inclusive
    band = int(max(1, pitch // 2))

    stripes = np.zeros((size, size), dtype=np.uint8)
    offset = int(rng.integers(0, pitch))
    for y in range(offset, size, pitch):
        y0 = y
        y1 = min(size, y + band)
        stripes[y0:y1, :] = 1

    target = (stripes & boundary_mask & (1 - keepout_mask)).astype(np.uint8)
    return target, pitch


def generate_one(rng: np.random.Generator, size: int):
    boundary_poly = _random_boundary_polygon(rng, size)

    boundary_mask = np.zeros((size, size), dtype=np.uint8)
    _fill_poly(boundary_mask, boundary_poly, 1)

    # Keepouts: 0..5 polygons. We'll accept only those that overlap the boundary.
    keepout_mask = np.zeros((size, size), dtype=np.uint8)
    keepout_count = int(rng.integers(0, 6))
    placed = 0
    for _ in range(keepout_count):
        poly = _random_keepout_polygon(rng, size)
        tmp = np.zeros((size, size), dtype=np.uint8)
        _fill_poly(tmp, poly, 1)
        # Keep only keepouts that land inside boundary a bit
        if int((tmp & boundary_mask).sum()) < 20:
            continue
        keepout_mask = np.maximum(keepout_mask, tmp)
        placed += 1

    target_mask, pitch = _generate_row_layout(rng, boundary_mask, keepout_mask)

    cond = np.stack([boundary_mask, keepout_mask], axis=0).astype(np.float32)
    target = target_mask[None, :, :].astype(np.float32)
    meta = {
        "size": int(size),
        "pitch": int(pitch),
        "keepouts_requested": int(keepout_count),
        "keepouts_placed": int(placed),
        "boundary_area_px": int(boundary_mask.sum()),
        "target_area_px": int(target_mask.sum()),
    }
    return cond, target, meta


def _save_sample(out_dir: Path, idx: int, cond: np.ndarray, target: np.ndarray, meta: dict) -> None:
    sample_dir = out_dir / f"sample_{idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "cond.npy", cond)
    np.save(sample_dir / "target.npy", target)
    with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _save_preview_grid(out_path: Path, conds: List[np.ndarray], targets: List[np.ndarray]) -> None:
    """Save a 3x3 preview grid PNG showing boundary/keepouts/target."""

    # Each cell is a color image:
    #  - boundary in red
    #  - keepouts in blue
    #  - target rows in green
    tiles = []
    for cond, target in zip(conds, targets):
        b = (cond[0] > 0.5).astype(np.uint8)
        k = (cond[1] > 0.5).astype(np.uint8)
        t = (target[0] > 0.5).astype(np.uint8)
        rgb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = b * 255
        rgb[:, :, 1] = t * 255
        rgb[:, :, 2] = k * 255
        tiles.append(rgb)

    n = 3
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.zeros((n * tile_h, n * tile_w, 3), dtype=np.uint8)
    for i in range(n * n):
        r = i // n
        c = i % n
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tiles[i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 expects BGR
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic conditioning/target masks for cVAE training")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="data/synthetic")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate 9 samples and save a preview grid PNG to <out_dir>/preview.png",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.preview:
        conds, targets = [], []
        for i in range(9):
            cond, target, meta = generate_one(rng, args.size)
            _save_sample(out_dir, i + 1, cond, target, meta)
            conds.append(cond)
            targets.append(target)
        _save_preview_grid(out_dir / "preview.png", conds, targets)
        print(f"Wrote 9 samples + preview to {out_dir}")
        return

    for i in range(args.n_samples):
        cond, target, meta = generate_one(rng, args.size)
        _save_sample(out_dir, i + 1, cond, target, meta)
        if (i + 1) % 250 == 0:
            print(f"Generated {i + 1}/{args.n_samples}")

    print(f"Done. Wrote {args.n_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
