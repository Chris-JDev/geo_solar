"""Convert binary layout masks into row-like polylines.

Input masks are 256x256 with values {0,1}.
We skeletonize to get 1-pixel centerlines, then trace connected pixels into
polyline point sequences.

This is designed to be beginner-friendly and CPU-only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from skimage.morphology import skeletonize


Point = Tuple[float, float]


def _polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    length = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        length += (dx * dx + dy * dy) ** 0.5
    return float(length)


def _neighbors8(x: int, y: int) -> List[Tuple[int, int]]:
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out.append((x + dx, y + dy))
    return out


def _build_graph(skel_u8: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Build an 8-neighborhood adjacency list for skeleton pixels."""

    h, w = skel_u8.shape
    ys, xs = np.where(skel_u8 > 0)
    nodes = set(zip(xs.tolist(), ys.tolist()))

    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for x, y in nodes:
        nbrs = []
        for nx, ny in _neighbors8(x, y):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) in nodes:
                nbrs.append((nx, ny))
        graph[(x, y)] = nbrs
    return graph


def _trace_from(graph: Dict[Tuple[int, int], List[Tuple[int, int]]], start: Tuple[int, int], visited: set) -> List[Tuple[int, int]]:
    """Trace a polyline starting from 'start' until it terminates.

    This is a simple walker: it tries to continue forward (avoid going back to the previous
    pixel). Branches create multiple segments (handled by starting from multiple endpoints).
    """

    line = [start]
    visited.add(start)

    prev = None
    cur = start

    while True:
        nbrs = graph.get(cur, [])
        # Prefer an unvisited neighbor that's not the previous node.
        candidates = [n for n in nbrs if n != prev and n not in visited]
        if not candidates:
            # If we're at a branch/loop, allow moving to an unvisited neighbor even if it equals prev
            candidates = [n for n in nbrs if n not in visited]
        if not candidates:
            break

        nxt = candidates[0]
        prev, cur = cur, nxt
        line.append(cur)
        visited.add(cur)

    return line


def mask_to_row_lines(mask_u8: np.ndarray, min_len: int = 30) -> List[List[Point]]:
    """Convert a 256x256 binary mask into a list of row-like polylines.

    Steps:
    1) skeletonize() to reduce thick stripes into 1-pixel centerlines
    2) trace connected pixels into ordered polylines
    3) filter short polylines by length

    Args:
        mask_u8: uint8/bool array shape (256,256) with values 0/1
        min_len: minimum polyline length (in grid pixels)

    Returns:
        list of polylines, each polyline is a list of (x,y) points in grid coordinates.
    """

    if mask_u8 is None:
        return []
    if mask_u8.shape != (256, 256):
        raise ValueError(f"mask_u8 must be shape (256,256), got {mask_u8.shape}")

    mask_bool = (mask_u8 > 0)
    if not mask_bool.any():
        return []

    skel = skeletonize(mask_bool).astype(np.uint8)
    if skel.sum() == 0:
        return []

    graph = _build_graph(skel)
    degrees = {node: len(nbrs) for node, nbrs in graph.items()}

    endpoints = [n for n, deg in degrees.items() if deg == 1]

    lines_xy: List[List[Point]] = []
    visited: set = set()

    # Trace from endpoints first (these produce the cleanest segments).
    for ep in endpoints:
        if ep in visited:
            continue
        nodes = _trace_from(graph, ep, visited)
        pts = [(float(x), float(y)) for x, y in nodes]
        if _polyline_length(pts) >= float(min_len):
            lines_xy.append(pts)

    # Trace any remaining pixels (loops or leftover fragments)
    for node in graph.keys():
        if node in visited:
            continue
        nodes = _trace_from(graph, node, visited)
        pts = [(float(x), float(y)) for x, y in nodes]
        if _polyline_length(pts) >= float(min_len):
            lines_xy.append(pts)

    return lines_xy
