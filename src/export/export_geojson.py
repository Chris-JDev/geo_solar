"""GeoJSON export helpers.

Beginner-friendly: accepts a single shapely geometry or a list of geometries and writes a
valid GeoJSON FeatureCollection in EPSG:4326.
"""

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform


def export_geojson(
    geometries: Union[Any, Sequence[Any]],
    out_path: str,
    src_crs: Any,
    properties: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
) -> None:
    """Write GeoJSON FeatureCollection in EPSG:4326.

    Args:
        geometries: A single shapely geometry or a list of geometries.
        out_path: Where to write the GeoJSON.
        src_crs: CRS of the input geometries (e.g., "EPSG:4326" or a pyproj CRS).
        properties:
            - None: each feature gets {}.
            - dict: applied to all features.
            - list of dicts: one per geometry (e.g. {"type": "keepout", "id": 0}).
    """

    if geometries is None:
        raise ValueError("geometries cannot be None")

    geoms: List[Any]
    if isinstance(geometries, (list, tuple)):
        geoms = list(geometries)
    else:
        geoms = [geometries]

    if properties is None:
        props_list: List[Dict[str, Any]] = [{} for _ in geoms]
    elif isinstance(properties, dict):
        props_list = [properties for _ in geoms]
    else:
        props_list = list(properties)
        if len(props_list) != len(geoms):
            raise ValueError("If properties is a list, it must match geometries length")

    tx = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def _proj(x, y, z=None):
        return tx.transform(x, y)

    fc = {"type": "FeatureCollection", "features": []}
    for geom, props in zip(geoms, props_list):
        if geom is None:
            continue
        geom_wgs = transform(_proj, geom)
        fc["features"].append(
            {"type": "Feature", "properties": dict(props), "geometry": mapping(geom_wgs)}
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
