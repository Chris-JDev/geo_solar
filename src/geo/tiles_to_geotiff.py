"""Leafmap wrapper: create a GeoTIFF from a bbox by downloading map tiles.

This is intentionally a tiny adapter so the Streamlit app stays readable.

Requires:
- leafmap (pip install leafmap)

Notes:
- bbox is EPSG:4326 in the format [min_lon, min_lat, max_lon, max_lat]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def tiles_to_geotiff(
    output: str,
    bbox: list[float],
    zoom: int = 18,
    source: str = "OpenStreetMap",
    **kwargs: Any,
) -> str:
    """Download map tiles and write a georeferenced GeoTIFF.

    Args:
        output: output .tif path
        bbox: [min_lon, min_lat, max_lon, max_lat] (EPSG:4326)
        zoom: tile zoom
        source: leafmap tile source name or URL template

    Returns:
        Absolute path to the written GeoTIFF.
    """

    def _tiles_to_geotiff_fallback() -> str:
        """Fallback path that does NOT require python GDAL bindings.

        Uses:
        - mercantile (tile math)
        - requests + Pillow (download/decode tiles)
        - rasterio (write GeoTIFF in EPSG:3857)
        """
        try:
            import mercantile  # type: ignore
        except Exception as e:
            raise ImportError("mercantile is required for the fallback. Install it with: pip install mercantile") from e

        try:
            import rasterio  # type: ignore
            from rasterio.transform import from_origin  # type: ignore
        except Exception as e:
            raise ImportError(
                "GeoTIFF creation fallback requires rasterio. Install it with: pip install rasterio"
            ) from e

        try:
            import requests
            from PIL import Image
            import io
        except Exception as e:
            raise ImportError("requests and pillow are required. Install requirements.txt") from e

        if len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
        z = int(zoom)

        # Resolve tile URL template.
        if source == "OpenStreetMap":
            url_tmpl = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        elif source == "Esri.WorldImagery":
            url_tmpl = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        else:
            url_tmpl = str(source)
        if "{x}" not in url_tmpl or "{y}" not in url_tmpl or "{z}" not in url_tmpl:
            raise ValueError("source must be a known preset or a URL template containing {z}/{x}/{y}")

        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, [z]))
        if not tiles:
            raise ValueError("No tiles found for bbox at this zoom")

        x_min = min(t.x for t in tiles)
        x_max = max(t.x for t in tiles)
        y_min = min(t.y for t in tiles)
        y_max = max(t.y for t in tiles)

        tile_size = 256
        width = (x_max - x_min + 1) * tile_size
        height = (y_max - y_min + 1) * tile_size
        mosaic = np.zeros((height, width, 3), dtype=np.uint8)

        headers = {"User-Agent": "pvx-genai/0.1"}

        for t in tiles:
            url = url_tmpl.format(z=t.z, x=t.x, y=t.y)
            try:
                resp = requests.get(url, timeout=20, headers=headers)
                resp.raise_for_status()
                im = Image.open(io.BytesIO(resp.content)).convert("RGB")
                arr = np.array(im, dtype=np.uint8)
            except Exception:
                # Leave tile blank if download fails.
                arr = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

            ox = (t.x - x_min) * tile_size
            oy = (t.y - y_min) * tile_size
            mosaic[oy : oy + tile_size, ox : ox + tile_size] = arr

        # Bounds in WebMercator (EPSG:3857)
        tl = mercantile.xy_bounds(x_min, y_min, z)
        br = mercantile.xy_bounds(x_max, y_max, z)
        left = float(tl.left)
        top = float(tl.top)
        right = float(br.right)
        bottom = float(br.bottom)

        res_x = (right - left) / float(width)
        res_y = (top - bottom) / float(height)
        transform = from_origin(left, top, res_x, res_y)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            str(out_path),
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=mosaic.dtype,
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            dst.write(np.transpose(mosaic, (2, 0, 1)))

        return str(out_path.resolve())

    try:
        import leafmap  # type: ignore
    except Exception as e:
        raise ImportError("leafmap is not installed. Install it with: pip install leafmap") from e

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Preferred path: leafmap API.
    try:
        leafmap.map_tiles_to_geotiff(str(out_path), bbox=bbox, zoom=int(zoom), source=source, **kwargs)
        return str(out_path.resolve())
    except Exception as e:
        msg = str(e)
        # On Windows, some leafmap paths require Python GDAL bindings.
        # Fall back to a rasterio-based writer (usually easier to install).
        if "GDAL" in msg or "osgeo" in msg:
            return _tiles_to_geotiff_fallback()
        raise
