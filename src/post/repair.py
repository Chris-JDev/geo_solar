"""Clip and clean row polylines against boundary/keepouts (grid coords).

Everything here operates in the 256x256 grid coordinate system.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def _flatten_lines(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "LineString":
        return [geom]
    if gtype == "MultiLineString":
        return [g for g in geom.geoms if (g is not None and not g.is_empty)]
    if gtype == "GeometryCollection":
        out: List[LineString] = []
        for g in geom.geoms:
            out.extend(_flatten_lines(g))
        return out
    # ignore polygons/points
    return []


def clip_and_clean_rows(
    row_lines: Sequence[Sequence[tuple[float, float]]],
    boundary_poly_grid: Polygon,
    keepouts_polys_grid: Sequence[Polygon],
    buffer_px: int = 2,
) -> List[LineString]:
    """Clip row lines to boundary and subtract buffered keepouts.

    Args:
        row_lines: list of polylines (each list of (x,y) grid points)
        boundary_poly_grid: shapely Polygon in grid coords
        keepouts_polys_grid: list of shapely Polygons in grid coords
        buffer_px: keepout buffer in grid pixels

    Returns:
        list of cleaned shapely LineStrings in grid coords
    """

    if boundary_poly_grid is None or boundary_poly_grid.is_empty:
        return []

    keepout_union = None
    if keepouts_polys_grid:
        buffered = [p.buffer(float(buffer_px)) for p in keepouts_polys_grid if p is not None and not p.is_empty]
        if buffered:
            keepout_union = unary_union(buffered)

    cleaned: List[LineString] = []

    for pts in row_lines:
        if pts is None or len(pts) < 2:
            continue
        line = LineString(pts)
        if line.is_empty or line.length == 0:
            continue

        # Clip to boundary
        clipped = line.intersection(boundary_poly_grid)
        for seg in _flatten_lines(clipped):
            if keepout_union is not None and not keepout_union.is_empty:
                seg2 = seg.difference(keepout_union)
                cleaned.extend([s for s in _flatten_lines(seg2) if s.length > 0])
            else:
                if seg.length > 0:
                    cleaned.append(seg)

    return cleaned
