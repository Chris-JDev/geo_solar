"""Georeferencing helpers.

Supports:
- Simple 2-point north-up linear mapping (TwoPointGeoRef)
- Corner-based mapping (auto from provided corners)
- Best-effort auto-detection for uploaded images (GeoTIFF, world file, EXIF, OCR)

All georefs returned by this module expose:
    - pixel_to_wgs84(px, py) -> (lat, lon)
    - wgs84_to_pixel(lat, lon) -> (px, py)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyproj import Transformer

from .crs import utm_crs_from_latlon


@dataclass
class AffineGeoRef:
    """General affine georef (pixel -> map coords), with optional CRS->WGS84 conversion."""

    # Affine transform in "world file" convention:
    # x = a*px + b*py + c
    # y = d*px + e*py + f
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    # Transformer from source CRS -> WGS84 (lon,lat) if needed.
    # If None, we assume x=lon and y=lat already.
    to_wgs84: Optional[Transformer] = None
    from_wgs84: Optional[Transformer] = None

    def pixel_to_wgs84(self, px: float, py: float) -> Tuple[float, float]:
        x = self.a * px + self.b * py + self.c
        y = self.d * px + self.e * py + self.f
        if self.to_wgs84 is not None:
            lon, lat = self.to_wgs84.transform(x, y)
            return float(lat), float(lon)
        # Assume already lon/lat
        return float(y), float(x)

    def wgs84_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        # Convert to source CRS if needed.
        x = float(lon)
        y = float(lat)
        if self.from_wgs84 is not None:
            x, y = self.from_wgs84.transform(lon, lat)

        # Invert 2x2 affine part.
        det = (self.a * self.e - self.b * self.d)
        if abs(det) < 1e-12:
            # Degenerate; return something safe.
            return 0.0, 0.0

        inv_a = self.e / det
        inv_b = -self.b / det
        inv_d = -self.d / det
        inv_e = self.a / det

        px = inv_a * (x - self.c) + inv_b * (y - self.f)
        py = inv_d * (x - self.c) + inv_e * (y - self.f)
        return float(px), float(py)


@dataclass
class TwoPointGeoRef:
    px1: float
    py1: float
    lat1: float
    lon1: float
    px2: float
    py2: float
    lat2: float
    lon2: float

    def __post_init__(self) -> None:
        dx_px = self.px2 - self.px1
        dy_px = self.py2 - self.py1
        dx_lon = self.lon2 - self.lon1
        dy_lat = self.lat2 - self.lat1

        self.x_scale = dx_lon / dx_px if dx_px != 0 else 0.0
        self.y_scale = dy_lat / dy_px if dy_px != 0 else 0.0
        self.x_offset = self.lon1 - self.px1 * self.x_scale
        self.y_offset = self.lat1 - self.py1 * self.y_scale

        mean_lat = (self.lat1 + self.lat2) / 2.0
        mean_lon = (self.lon1 + self.lon2) / 2.0
        utm = utm_crs_from_latlon(mean_lat, mean_lon)
        self._to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        self._to_wgs = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)

    def pixel_to_wgs84(self, px: float, py: float) -> Tuple[float, float]:
        lon = px * self.x_scale + self.x_offset
        lat = py * self.y_scale + self.y_offset
        return lat, lon

    def wgs84_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        px = (lon - self.x_offset) / (self.x_scale if self.x_scale != 0 else 1e-9)
        py = (lat - self.y_offset) / (self.y_scale if self.y_scale != 0 else 1e-9)
        return px, py

    def pixel_to_utm(self, px: float, py: float) -> Tuple[float, float]:
        lat, lon = self.pixel_to_wgs84(px, py)
        x, y = self._to_utm.transform(lon, lat)
        return x, y

    def utm_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        lon, lat = self._to_wgs.transform(x, y)
        return self.wgs84_to_pixel(lat, lon)


def make_georef(
        px1: float,
        py1: float,
        lat1: float,
        lon1: float,
        px2: float,
        py2: float,
        lat2: float,
        lon2: float,
) -> TwoPointGeoRef:
        """Factory function required by the MVP spec.

        Returns an object with:
            - pixel_to_wgs84(x, y) -> (lat, lon)
            - wgs84_to_pixel(lat, lon) -> (x, y)

        Notes for beginners:
        - This is a simple linear mapping from 2 points.
        - It assumes the image is north-up (no rotation).
        - Good enough for a demo; not a survey-grade georeference.
        """

        return TwoPointGeoRef(px1, py1, lat1, lon1, px2, py2, lat2, lon2)


def make_georef_from_corners(
    img_w: int,
    img_h: int,
    lat_top_left: float,
    lon_top_left: float,
    lat_bottom_right: float,
    lon_bottom_right: float,
) -> TwoPointGeoRef:
    """Create a georef using image corner coordinates.

    Maps:
      - P1 pixel = (0,0)               <-> (lat_top_left, lon_top_left)
      - P2 pixel = (img_w-1, img_h-1)  <-> (lat_bottom_right, lon_bottom_right)

    Notes:
    - Assumes north-up and no rotation (same assumption as make_georef).
    - Works well when the provided corners represent the image extent.
    """

    if img_w is None or img_h is None:
        raise ValueError("img_w and img_h are required")
    if int(img_w) < 2 or int(img_h) < 2:
        raise ValueError("Image must be at least 2x2 pixels for corner georeferencing")

    px1, py1 = 0.0, 0.0
    px2, py2 = float(int(img_w) - 1), float(int(img_h) - 1)
    return TwoPointGeoRef(
        px1,
        py1,
        float(lat_top_left),
        float(lon_top_left),
        px2,
        py2,
        float(lat_bottom_right),
        float(lon_bottom_right),
    )


def try_make_georef_auto(uploaded_file: Any, image_np: np.ndarray) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Try multiple georef detection methods for a Streamlit-uploaded image.

    Returns:
        (georef_or_none, info_dict)

    The info_dict contains per-method status messages.
    """

    info: Dict[str, Any] = {
        "geotiff": {"attempted": False, "ok": False, "message": "not attempted"},
        "world_file": {"attempted": False, "ok": False, "message": "not attempted"},
        "exif_gps": {"attempted": False, "ok": False, "message": "not attempted"},
        "ocr": {"attempted": False, "ok": False, "message": "not attempted"},
        "warning": None,
    }

    # Helpers
    def _uploaded_name() -> str:
        name = getattr(uploaded_file, "name", "") or ""
        return str(name)

    def _get_bytes() -> Optional[bytes]:
        try:
            return uploaded_file.getvalue()  # streamlit UploadedFile
        except Exception:
            try:
                return bytes(uploaded_file)
            except Exception:
                return None

    fname = _uploaded_name().lower()
    img_h, img_w = image_np.shape[:2]

    # 1) GeoTIFF
    if fname.endswith(".tif") or fname.endswith(".tiff"):
        info["geotiff"]["attempted"] = True
        try:
            import rasterio  # type: ignore
            from rasterio.io import MemoryFile  # type: ignore

            data = _get_bytes()
            if not data:
                raise ValueError("could not read uploaded bytes")

            with MemoryFile(data) as mem:
                with mem.open() as ds:
                    if ds.crs is None or ds.transform is None:
                        raise ValueError("missing CRS/transform")

                    # Pixel (col,row) -> CRS x/y using affine.
                    t = ds.transform
                    src_crs = ds.crs
                    to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
                    from_wgs = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

                    georef = AffineGeoRef(
                        a=float(t.a),
                        b=float(t.b),
                        c=float(t.c),
                        d=float(t.d),
                        e=float(t.e),
                        f=float(t.f),
                        to_wgs84=to_wgs,
                        from_wgs84=from_wgs,
                    )
                    info["geotiff"]["ok"] = True
                    info["geotiff"]["message"] = f"GeoTIFF CRS={src_crs.to_string()}"
                    return georef, info
        except ImportError:
            info["geotiff"]["message"] = "rasterio not installed; skipping GeoTIFF"
        except Exception as e:
            info["geotiff"]["message"] = f"failed: {e}"

    # 2) World file (.pgw/.jgw/.tfw etc)
    info["world_file"]["attempted"] = True
    try:
        # Best-effort: look for a sidecar world file by basename.
        # In Streamlit uploads we usually *don't* have a folder, so this often fails.
        base = Path(_uploaded_name()).stem
        candidates = [f"{base}.pgw", f"{base}.jgw", f"{base}.tfw", f"{base}.wld"]
        search_dirs = [Path.cwd(), Path(__file__).resolve().parents[2]]

        wf_path = None
        for d in search_dirs:
            for c in candidates:
                p = d / c
                if p.exists():
                    wf_path = p
                    break
            if wf_path is not None:
                break

        if wf_path is None:
            raise FileNotFoundError("no matching world file found (upload sidecar not provided)")

        lines = [ln.strip() for ln in wf_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if len(lines) < 6:
            raise ValueError("world file must have 6 lines")
        a, d, b, e, c, f = [float(lines[i]) for i in range(6)]

        # IMPORTANT: Without CRS info, we assume x=lon and y=lat (EPSG:4326).
        georef = AffineGeoRef(a=a, b=b, c=c, d=d, e=e, f=f, to_wgs84=None, from_wgs84=None)
        info["world_file"]["ok"] = True
        info["world_file"]["message"] = f"world file: {wf_path.name} (assumed EPSG:4326)"
        info["warning"] = "World file CRS unknown; assuming EPSG:4326 (lon/lat)."
        return georef, info
    except Exception as e:
        info["world_file"]["message"] = f"failed: {e}"

    # 3) EXIF GPS (approximate)
    info["exif_gps"]["attempted"] = True
    try:
        import io
        from PIL import Image  # type: ignore
        from PIL.ExifTags import GPSTAGS, TAGS  # type: ignore

        data = _get_bytes()
        if not data:
            raise ValueError("could not read uploaded bytes")

        im = Image.open(io.BytesIO(data))
        exif = getattr(im, "_getexif", lambda: None)()
        if not exif:
            raise ValueError("no EXIF")

        gps_info = None
        for k, v in exif.items():
            tag = TAGS.get(k, k)
            if tag == "GPSInfo":
                gps_info = v
                break
        if not gps_info:
            raise ValueError("no GPSInfo")

        gps_parsed = {}
        for k, v in gps_info.items():
            gps_parsed[GPSTAGS.get(k, k)] = v

        def _to_deg(values):
            d = float(values[0][0]) / float(values[0][1])
            m = float(values[1][0]) / float(values[1][1])
            s = float(values[2][0]) / float(values[2][1])
            return d + m / 60.0 + s / 3600.0

        lat = _to_deg(gps_parsed["GPSLatitude"])  # type: ignore
        if str(gps_parsed.get("GPSLatitudeRef", "N")).upper().startswith("S"):
            lat = -lat
        lon = _to_deg(gps_parsed["GPSLongitude"])  # type: ignore
        if str(gps_parsed.get("GPSLongitudeRef", "E")).upper().startswith("W"):
            lon = -lon

        # Approximate mapping: treat EXIF GPS as image center, assume a default ground span.
        span_m = 200.0
        deg_per_m_lat = 1.0 / 111_320.0
        deg_per_m_lon = 1.0 / (111_320.0 * max(0.1, np.cos(np.deg2rad(lat))))
        deg_span_lat = span_m * deg_per_m_lat
        deg_span_lon = span_m * deg_per_m_lon

        lat_tl = lat + 0.5 * deg_span_lat
        lon_tl = lon - 0.5 * deg_span_lon
        lat_br = lat - 0.5 * deg_span_lat
        lon_br = lon + 0.5 * deg_span_lon

        georef = make_georef_from_corners(img_w, img_h, lat_tl, lon_tl, lat_br, lon_br)
        info["exif_gps"]["ok"] = True
        info["exif_gps"]["message"] = f"EXIF GPS found (approx, assumed {span_m:.0f}m span)"
        info["warning"] = "EXIF GPS georef is approximate (uses a default image ground span)."
        return georef, info
    except Exception as e:
        info["exif_gps"]["message"] = f"failed: {e}"

    # 4) OCR (best effort; optional)
    info["ocr"]["attempted"] = True
    try:
        from .ocr_georef import try_ocr_corner_coords

        pair = try_ocr_corner_coords(image_np)
        if pair is None:
            raise ValueError("no coordinates detected")

        a, b = pair

        # Need two distinct points with different x and different y to solve full 2D scale.
        # If y is identical, lat scale is undefined; in that case we still try to use
        # the image corners with a small inferred lat span to avoid degeneracy.
        if abs(a.py - b.py) < 1e-6 and abs(a.px - b.px) > 1e-6:
            # Infer lon scale from left/right; infer a small default lat span.
            lon_left, lon_right = (a.lon, b.lon) if a.px < b.px else (b.lon, a.lon)
            lat_base = a.lat
            lat_tl = lat_base + 0.0005
            lat_br = lat_base - 0.0005
            georef = make_georef_from_corners(img_w, img_h, lat_tl, lon_left, lat_br, lon_right)
            info["ocr"]["ok"] = True
            info["ocr"]["message"] = "OCR found bottom coords; used small inferred lat span (approx)"
            info["warning"] = "OCR georef is approximate (inferred latitude span)."
            return georef, info

        # General case: use the two OCR anchors as the calibration points.
        georef = make_georef(a.px, a.py, a.lat, a.lon, b.px, b.py, b.lat, b.lon)
        info["ocr"]["ok"] = True
        info["ocr"]["message"] = "OCR extracted two coordinate anchors (approx)"
        info["warning"] = "OCR georef is approximate; verify using the sanity table."
        return georef, info
    except ImportError:
        info["ocr"]["message"] = "pytesseract not installed; skipping OCR"
    except Exception as e:
        info["ocr"]["message"] = f"failed: {e}"

    return None, info


def make_georef_from_geotiff(tif_path: str) -> Any:
    """Create a georef from a GeoTIFF's CRS + affine transform.

    Args:
        tif_path: path to a .tif/.tiff file

    Returns:
        An object with pixel_to_wgs84(px, py) and wgs84_to_pixel(lat, lon).

    Notes:
    - Uses rasterio to read CRS and dataset transform.
    - Always outputs WGS84 lat/lon (EPSG:4326).
    """

    try:
        import rasterio  # type: ignore
    except Exception as e:
        raise ImportError(
            "rasterio is required to read GeoTIFF georeferencing. Install it with: pip install rasterio"
        ) from e

    path = Path(tif_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(str(path)) as ds:
        if ds.crs is None or ds.transform is None:
            raise ValueError("GeoTIFF missing CRS and/or transform")

        t = ds.transform
        src_crs = ds.crs

        # Convert dataset CRS <-> WGS84
        to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        from_wgs = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

        # rasterio Affine: x = a*col + b*row + c ; y = d*col + e*row + f
        return AffineGeoRef(
            a=float(t.a),
            b=float(t.b),
            c=float(t.c),
            d=float(t.d),
            e=float(t.e),
            f=float(t.f),
            to_wgs84=to_wgs,
            from_wgs84=from_wgs,
        )
