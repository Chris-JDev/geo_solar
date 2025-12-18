"""Best-effort OCR-based georeference hints.

This is intentionally optional: if pytesseract isn't installed (or tesseract
binary isn't available), we skip OCR and return None.

We try to read coordinate labels that sometimes appear around map imagery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OCRResult:
    # Pixel anchor where the OCR text was read.
    px: float
    py: float
    lat: float
    lon: float
    raw_text: str


def _parse_decimal_latlon(text: str) -> Optional[Tuple[float, float]]:
    """Parse decimal degrees like '25.2048, 55.2708' or '25.2048 55.2708'."""
    t = (text or "").replace(";", ",")
    nums = re.findall(r"[-+]?\d{1,3}\.\d+", t)
    if len(nums) >= 2:
        lat = float(nums[0])
        lon = float(nums[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None


def _parse_dms(text: str) -> Optional[Tuple[float, float]]:
    """Parse a simple DMS pattern like 25°12'17"N 55°16'15"E."""

    def dms_to_deg(deg: float, minutes: float, seconds: float, hemi: str) -> float:
        value = abs(deg) + minutes / 60.0 + seconds / 3600.0
        if hemi.upper() in ("S", "W"):
            value = -value
        return value

    # Example: 25°12'17"N 55°16'15"E (allow a lot of OCR noise)
    pat = re.compile(
        r"(\d{1,3})\s*[°o]\s*(\d{1,2})\s*['’]\s*(\d{1,2})\s*[\"”]?\s*([NnSs])"
        r"[^0-9A-Za-z]+"
        r"(\d{1,3})\s*[°o]\s*(\d{1,2})\s*['’]\s*(\d{1,2})\s*[\"”]?\s*([EeWw])"
    )
    m = pat.search(text or "")
    if not m:
        return None

    lat = dms_to_deg(float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
    lon = dms_to_deg(float(m.group(5)), float(m.group(6)), float(m.group(7)), m.group(8))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def _try_parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    return _parse_decimal_latlon(text) or _parse_dms(text)


def _crop_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def try_ocr_corner_coords(image_np: np.ndarray) -> Optional[Tuple[OCRResult, OCRResult]]:
    """Try OCR on a few corner regions to extract two coordinate readouts.

    Returns two OCRResult entries if it finds at least two *different* anchors
    (preferably with different x and/or y), else None.

    NOTE: We attempt bottom-left/bottom-right per spec, and also try top-left/top-right
    as a best-effort extension.
    """

    try:
        import pytesseract  # type: ignore
    except Exception:
        return None

    img = image_np
    if img is None:
        return None

    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return None

    # Crop bands ~20% wide and ~18% tall.
    cw = int(0.22 * w)
    ch = int(0.18 * h)

    regions = [
        ("bottom_left", 0, h - ch, cw, h, 0.0, float(h - 1)),
        ("bottom_right", w - cw, h - ch, w, h, float(w - 1), float(h - 1)),
        ("top_left", 0, 0, cw, ch, 0.0, 0.0),
        ("top_right", w - cw, 0, w, ch, float(w - 1), 0.0),
    ]

    results: list[OCRResult] = []
    for name, x0, y0, x1, y1, px, py in regions:
        crop = _crop_region(img, x0, y0, x1, y1)
        try:
            text = pytesseract.image_to_string(crop)
        except Exception:
            continue

        parsed = _try_parse_latlon(text)
        if parsed is None:
            continue
        lat, lon = parsed
        results.append(OCRResult(px=px, py=py, lat=float(lat), lon=float(lon), raw_text=text.strip()))

    if len(results) < 2:
        return None

    # Pick two distinct anchors.
    a = results[0]
    for b in results[1:]:
        if (abs(a.px - b.px) + abs(a.py - b.py)) > 0.0:
            return a, b

    return None
