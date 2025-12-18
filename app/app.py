"""PVX-GenAI MVP: Georeference + boundary export.

Steps:
1) Upload an image
2) Enter 2 calibration points (pixel ↔ lat/lon)
3) Paste boundary vertices (pixel x,y per line)
4) Export as GeoJSON (EPSG:4326)

This is intentionally simple for beginners.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import io
import tempfile
import json
from datetime import datetime, timezone

import streamlit as st
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon

import cv2

# Make imports work even if Streamlit is launched from a different CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geo.georef import (
    make_georef,
    make_georef_from_corners,
    try_make_georef_auto,
    make_georef_from_geotiff,
)  # noqa: E402
from src.export.export_geojson import export_geojson  # noqa: E402
from src.geo.rasterize import polygon_to_mask, multi_polygon_to_mask  # noqa: E402
from src.gen.sample import load_cvae, generate_candidates  # noqa: E402
from src.geo.tiles_to_geotiff import tiles_to_geotiff  # noqa: E402
from src.post.vectorize import mask_to_row_lines  # noqa: E402
from src.post.repair import clip_and_clean_rows  # noqa: E402
from src.score.score import rank_candidates, score_candidate  # noqa: E402


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # mean Earth radius (meters)
    r = 6371008.8
    import math

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(r * c)


def _approx_pixel_area_m2(georef, img_w: int, img_h: int) -> float | None:
    """Approximate square meters per image pixel using georef at image center."""

    if georef is None or img_w is None or img_h is None:
        return None
    try:
        cx = float(img_w) * 0.5
        cy = float(img_h) * 0.5
        lat0, lon0 = georef.pixel_to_wgs84(cx, cy)
        latx, lonx = georef.pixel_to_wgs84(cx + 1.0, cy)
        laty, lony = georef.pixel_to_wgs84(cx, cy + 1.0)
        mx = _haversine_m(float(lat0), float(lon0), float(latx), float(lonx))
        my = _haversine_m(float(lat0), float(lon0), float(laty), float(lony))
        if mx <= 0 or my <= 0:
            return None
        return float(mx * my)
    except Exception:
        return None


def _mask_to_polygons_u8(mask_u8: np.ndarray):
    """Convert a 256x256 binary mask into shapely Polygons (grid coords)."""

    if mask_u8 is None:
        return []
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if m.sum() == 0:
        return []

    contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        polys.append(poly)
    return polys


def _overlay_with_rows(boundary_u8: np.ndarray, keepout_u8: np.ndarray, cand_u8: np.ndarray, row_lines_grid):
    boundary = (boundary_u8 > 0).astype(np.uint8)
    keepout = (keepout_u8 > 0).astype(np.uint8)
    cand = (cand_u8 > 0).astype(np.uint8)

    base = (boundary * 120).astype(np.uint8)
    overlay = np.stack([base, base, base], axis=-1)
    overlay[..., 0] = np.clip(overlay[..., 0] + keepout * 200, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] + cand * 200, 0, 255)

    if row_lines_grid:
        for ls in row_lines_grid:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            pts = np.array([[int(round(x)), int(round(y))] for x, y in coords], dtype=np.int32)
            pts = np.clip(pts, 0, 255)
            cv2.polylines(overlay, [pts.reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255), thickness=1)

    return overlay


def _parse_vertices(text: str):
    """Parse boundary vertices into a list of (x, y) floats.

    Accepted formats:
    - One vertex per line: `x,y`
    - One vertex per line: `x y`
    - Multiple vertices separated by whitespace/newlines: `x1,y1 x2,y2 ...`
    - Fallback: extract all numbers from the text and pair sequentially
    """

    import re

    raw = (text or "").strip()
    if not raw:
        return []

    pts = []

    # First pass: line-based parsing (most user-friendly).
    lines = [ln.strip() for ln in raw.replace(";", "\n").splitlines() if ln.strip()]
    for line in lines:
        # If the user pasted many pairs on one line, split them.
        chunks = line.split() if ("," in line and " " in line) else [line]
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if "," in chunk:
                xs, ys = chunk.split(",", 1)
                pts.append((float(xs.strip()), float(ys.strip())))
            else:
                parts = chunk.split()
                if len(parts) == 2:
                    pts.append((float(parts[0]), float(parts[1])))
                else:
                    # We'll fall back to regex parsing below.
                    pts = []
                    lines = []
                    break
        if not lines:
            break

    if pts:
        return pts

    # Fallback: extract numbers and pair them.
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if len(nums) % 2 != 0:
        raise ValueError(
            "Could not parse vertices. Provide `x,y` per line (or `x y`). "
            "Also OK: paste all numbers as x1 y1 x2 y2 ..."
        )
    floats = [float(n) for n in nums]
    return list(zip(floats[0::2], floats[1::2]))


def _parse_keepout_polygons(text: str):
    """Parse multiple polygons from a textarea.

    Format:
    - Polygons separated by a blank line
    - Each polygon is lines of x,y (or x y)

    Returns: list of list[(x, y)]
    """

    raw = (text or "").strip()
    if not raw:
        return []

    blocks = []
    current = []
    for line in raw.splitlines():
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))

    polys = []
    for block in blocks:
        pts = _parse_vertices(block)
        polys.append(pts)
    return polys


def _is_georef_configured(px1, py1, lat1, lon1, px2, py2, lat2, lon2) -> bool:
    """Return True if calibration points are non-degenerate.

    We require the two pixel points to be meaningfully separated AND the two lat/lon
    points to be meaningfully separated (otherwise the mapping is undefined).
    """

    dx_px = float(px2) - float(px1)
    dy_px = float(py2) - float(py1)
    dpx = (dx_px * dx_px + dy_px * dy_px) ** 0.5

    dlat = abs(float(lat2) - float(lat1))
    dlon = abs(float(lon2) - float(lon1))

    # Tolerances: keep them simple and beginner-friendly.
    # - pixel points must be at least ~1px apart
    # - lat/lon points must differ by at least ~1e-7 degrees (~1 cm-ish at equator)
    if dpx < 1.0:
        return False
    if (dlat < 1e-7) and (dlon < 1e-7):
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="pvx-genai MVP", layout="wide")
    st.title("pvx-genai — Georef + Boundary Export MVP")

    st.header("Create GeoTIFF from Bounding Box (leafmap)")
    st.caption(
        "Optional: generate a georeferenced GeoTIFF by downloading map tiles for a bounding box. "
        "You can then upload the GeoTIFF above and use Auto-detect georeferencing."
    )

    bb1, bb2, bb3, bb4 = st.columns(4)
    with bb1:
        min_lat = st.number_input("min_lat", value=0.0, format="%.6f")
    with bb2:
        min_lon = st.number_input("min_lon", value=0.0, format="%.6f")
    with bb3:
        max_lat = st.number_input("max_lat", value=0.001, format="%.6f")
    with bb4:
        max_lon = st.number_input("max_lon", value=0.001, format="%.6f")

    cA, cB, cC = st.columns(3)
    with cA:
        zoom = st.number_input("zoom", min_value=0, max_value=22, value=18, step=1)
    with cB:
        out_tif = st.text_input("output filename", value="data/site.tif")
    with cC:
        source = st.selectbox("basemap source", options=["OpenStreetMap", "Esri.WorldImagery"], index=0)

    if "created_geotiff" not in st.session_state:
        st.session_state.created_geotiff = None

    if st.button("Create GeoTIFF"):
        try:
            bbox = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                st.error("Invalid bbox: ensure max_lon>min_lon and max_lat>min_lat")
            else:
                out_path = tiles_to_geotiff(out_tif, bbox=bbox, zoom=int(zoom), source=str(source))
                st.session_state.created_geotiff = out_path
                st.success(f"GeoTIFF created at {out_path}")
        except Exception as e:
            msg = str(e)
            st.error(f"GeoTIFF creation failed: {msg}")
            if "GDAL" in msg or "osgeo" in msg:
                st.info(
                    "Tip (Windows): instead of installing python GDAL directly, try `pip install rasterio mercantile` "
                    "and re-run. The app will fall back to a rasterio-based GeoTIFF writer."
                )

    if st.session_state.created_geotiff:
        try:
            tif_path = Path(st.session_state.created_geotiff)
            if tif_path.exists():
                with open(tif_path, "rb") as f:
                    st.download_button(
                        "Download GeoTIFF",
                        data=f,
                        file_name=tif_path.name,
                        mime="image/tiff",
                    )
            else:
                st.warning("Created GeoTIFF path does not exist (was it moved/deleted?)")
        except Exception as e:
            st.error(f"Could not read GeoTIFF for download: {e}")

    st.header("1) Upload image")
    uploaded = st.file_uploader(
        "Upload a site image (JPG/PNG/GeoTIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    img_w = None
    img_h = None
    img = None
    uploaded_is_tif = False
    if uploaded is not None:
        name = getattr(uploaded, "name", "") or ""
        uploaded_is_tif = str(name).lower().endswith((".tif", ".tiff"))
        try:
            img = Image.open(uploaded).convert("RGB")
            img_w, img_h = img.size
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            # GeoTIFFs may not be readable by Pillow depending on environment.
            # We still want to allow auto-detect (GeoTIFF metadata) to run.
            st.warning(f"Preview unavailable for this upload ({e}). Auto-detect may still work.")
            img_w, img_h = None, None

    st.header("2) Enter 2-point calibration")
    st.caption("Assumes north-up (no rotation). For demo use only.")

    # If the working image is a GeoTIFF, prefer its embedded georeferencing.
    # This hides manual calibration fields, but keeps corner/manual modes available for JPG/PNG.
    georef_mode = None
    if not uploaded_is_tif:
        georef_mode = st.selectbox(
            "Georeference mode",
            options=["Auto-detect (recommended)", "Corner coordinates (user provides)", "Two calibration points (manual)"],
            index=0,
        )

    # Default values (used later by export section).
    px1 = py1 = lat1 = lon1 = px2 = py2 = lat2 = lon2 = 0.0
    georef = None

    def _show_sanity_table(g):
        if g is None or img_w is None or img_h is None:
            return
        st.subheader("Sanity overlay (corner lat/lon)")
        corners = [
            ("(0,0)", 0.0, 0.0),
            ("(W-1,0)", float(int(img_w) - 1), 0.0),
            ("(0,H-1)", 0.0, float(int(img_h) - 1)),
            ("(W-1,H-1)", float(int(img_w) - 1), float(int(img_h) - 1)),
        ]
        rows = []
        for label, x, y in corners:
            lat, lon = g.pixel_to_wgs84(x, y)
            rows.append({"pixel": label, "lat": float(lat), "lon": float(lon)})
        st.table(rows)

    # GeoTIFF direct path: use geotransform/CRS automatically.
    if uploaded_is_tif and uploaded is not None:
        st.subheader("GeoTIFF georeferencing")
        try:
            # Streamlit uploads are not file paths; write to a temp file for rasterio.
            data = uploaded.getvalue()
            if not data:
                raise ValueError("Uploaded GeoTIFF is empty")

            if "_uploaded_tif_path" not in st.session_state:
                st.session_state._uploaded_tif_path = None
            if st.session_state._uploaded_tif_path is None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                tmp.write(data)
                tmp.flush()
                tmp.close()
                st.session_state._uploaded_tif_path = tmp.name

            georef = make_georef_from_geotiff(st.session_state._uploaded_tif_path)
            st.success("Using embedded GeoTIFF georeferencing (transform + CRS)")

            # If Pillow couldn't read the TIFF, try to get width/height via rasterio.
            if img_w is None or img_h is None:
                try:
                    import rasterio  # type: ignore

                    with rasterio.open(st.session_state._uploaded_tif_path) as ds:
                        img_w, img_h = int(ds.width), int(ds.height)
                except Exception:
                    pass

            _show_sanity_table(georef)
        except ImportError as e:
            st.error(str(e))
            st.info("Install rasterio to enable GeoTIFF auto-georef: `pip install rasterio`")
        except Exception as e:
            st.error(f"Failed to read GeoTIFF georeferencing: {e}")

    elif georef_mode == "Auto-detect (recommended)":
        if uploaded is None:
            st.info("Upload an image first, then auto-detect will run.")
        else:
            # If we could not preview the image (e.g., some GeoTIFFs), fall back to a dummy
            # array so OCR/EXIF steps can be skipped safely while GeoTIFF metadata still works.
            image_np = np.array(img) if img is not None else np.zeros((10, 10, 3), dtype=np.uint8)
            g, info = try_make_georef_auto(uploaded, image_np=image_np)

            st.subheader("Auto-detect status")
            for key in ["geotiff", "world_file", "exif_gps", "ocr"]:
                entry = info.get(key, {})
                attempted = bool(entry.get("attempted"))
                ok = bool(entry.get("ok"))
                msg = str(entry.get("message", ""))
                prefix = "✅" if ok else ("⏳" if attempted else "•")
                st.write(f"{prefix} {key}: {msg}")

            if info.get("warning"):
                st.warning(str(info["warning"]))

            if g is None:
                st.error(
                    "Could not auto-detect georeferencing. "
                    "Please use Corner Coordinates or Manual mode."
                )
            else:
                georef = g
                _show_sanity_table(georef)

    elif georef_mode == "Two calibration points (manual)":
        c1, c2 = st.columns(2)
        with c1:
            px1 = st.number_input("px1", value=100.0)
            py1 = st.number_input("py1", value=100.0)
            lat1 = st.number_input("lat1", value=0.0, format="%.6f")
            lon1 = st.number_input("lon1", value=0.0, format="%.6f")
        with c2:
            px2 = st.number_input("px2", value=200.0)
            py2 = st.number_input("py2", value=200.0)
            lat2 = st.number_input("lat2", value=0.001, format="%.6f")
            lon2 = st.number_input("lon2", value=0.001, format="%.6f")

        # Show calibration deltas so users can quickly spot degenerate inputs.
        dx_px = float(px2) - float(px1)
        dy_px = float(py2) - float(py1)
        dpx = (dx_px * dx_px + dy_px * dy_px) ** 0.5
        dlat = abs(float(lat2) - float(lat1))
        dlon = abs(float(lon2) - float(lon1))
        st.caption(f"Calibration deltas: pixel distance={dpx:.3f}px, Δlat={dlat:.8f}, Δlon={dlon:.8f}")

        if _is_georef_configured(px1, py1, lat1, lon1, px2, py2, lat2, lon2):
            georef = make_georef(px1, py1, lat1, lon1, px2, py2, lat2, lon2)

    else:
        st.caption(
            "Enter the WGS84 lat/lon for the image corners. "
            "The app will automatically map (0,0) and (W-1,H-1)."
        )

        c1, c2 = st.columns(2)
        with c1:
            lat_tl = st.number_input("lat_top_left", value=0.0, format="%.6f")
            lon_tl = st.number_input("lon_top_left", value=0.0, format="%.6f")
        with c2:
            lat_br = st.number_input("lat_bottom_right", value=-0.001, format="%.6f")
            lon_br = st.number_input("lon_bottom_right", value=0.001, format="%.6f")

        if uploaded is None or img_w is None or img_h is None:
            st.warning("Upload an image first so we know its width/height.")
        else:
            try:
                georef = make_georef_from_corners(int(img_w), int(img_h), lat_tl, lon_tl, lat_br, lon_br)
                # Mirror into px/py/lat/lon vars so the rest of the app can keep using the same pattern.
                px1, py1, lat1, lon1 = 0.0, 0.0, float(lat_tl), float(lon_tl)
                px2, py2, lat2, lon2 = float(int(img_w) - 1), float(int(img_h) - 1), float(lat_br), float(lon_br)
            except Exception as e:
                st.error(f"Corner georef failed: {e}")

        # Sanity overlay: show what the georef thinks each corner is.
        _show_sanity_table(georef)

    st.header("3) Boundary vertices (pixel coordinates)")
    st.caption("Paste vertices as 'x,y' per line. Example:\n100,100\n300,120\n280,260\n110,240")
    vertices_text = st.text_area("Boundary vertices", height=160)

    st.header("Keepouts (optional)")
    st.caption(
        "Paste one or more polygons. Separate polygons with a blank line. "
        "Each polygon is lines of `x,y` (or `x y`)."
    )
    keepouts_text = st.text_area(
        "Keepout polygons",
        height=200,
        placeholder=(
            "120,200\n180,210\n160,260\n\n"
            "400,500\n520,510\n510,620\n390,610"
        ),
    )

    # Parse keepouts as the user types, so we can show a count.
    keepout_polys_px = []
    keepout_parse_error = None
    if keepouts_text.strip():
        try:
            keepout_polys_px = _parse_keepout_polygons(keepouts_text)
        except Exception as e:
            keepout_parse_error = str(e)

    if keepout_parse_error:
        st.warning(f"Keepouts not parsed yet: {keepout_parse_error}")
    else:
        if keepout_polys_px:
            st.info(f"Parsed keepouts: {len(keepout_polys_px)}")
        else:
            st.info("Parsed keepouts: 0")

    st.header("Build conditioning masks")
    st.caption(
        "Rasterization converts your polygons (vertices) into fixed-size pixel masks. "
        "We build 256×256 masks for boundary and keepouts, then stack them into a conditioning tensor." 
    )

    if "boundary_mask" not in st.session_state:
        st.session_state.boundary_mask = None
    if "keepout_mask" not in st.session_state:
        st.session_state.keepout_mask = None
    if "cond" not in st.session_state:
        st.session_state.cond = None
    if "candidates" not in st.session_state:
        st.session_state.candidates = None
    if "ranking" not in st.session_state:
        st.session_state.ranking = None
    if "selected_candidate_idx" not in st.session_state:
        st.session_state.selected_candidate_idx = 0

    if st.button("Build conditioning masks (256×256)"):
        if uploaded is None or img_w is None or img_h is None:
            st.warning("Upload an image first (needed for image width/height).")
        else:
            try:
                boundary_pts_px = _parse_vertices(vertices_text)
                if len(boundary_pts_px) < 3:
                    st.warning("Boundary needs at least 3 vertices.")
                else:
                    boundary_mask = polygon_to_mask(boundary_pts_px, img_w, img_h, out_size=256)

                    # Keepouts are optional.
                    if keepout_polys_px and not keepout_parse_error:
                        keepout_mask = multi_polygon_to_mask(keepout_polys_px, img_w, img_h, out_size=256)
                    else:
                        keepout_mask = np.zeros((256, 256), dtype=np.uint8)

                    cond = np.stack([boundary_mask, keepout_mask], axis=0).astype(np.float32)

                    st.session_state.boundary_mask = boundary_mask
                    st.session_state.keepout_mask = keepout_mask
                    st.session_state.cond = cond
                    st.success("Conditioning masks built")
            except Exception as e:
                st.error(f"Failed to build masks: {e}")

    if st.session_state.boundary_mask is not None:
        cA, cB = st.columns(2)
        with cA:
            st.subheader("Boundary mask (256×256)")
            st.image((st.session_state.boundary_mask * 255).astype(np.uint8), clamp=True)
        with cB:
            st.subheader("Keepout mask (256×256)")
            st.image((st.session_state.keepout_mask * 255).astype(np.uint8), clamp=True)

        buf = io.BytesIO()
        np.savez(
            buf,
            boundary_mask=st.session_state.boundary_mask,
            keepout_mask=st.session_state.keepout_mask,
            cond=st.session_state.cond,
        )
        st.download_button(
            "Download cond.npz",
            data=buf.getvalue(),
            file_name="cond.npz",
            mime="application/octet-stream",
        )

    st.header("Generate layouts (GenAI)")
    st.caption(
        "Uses the trained conditional VAE (cVAE) to sample multiple candidate layout masks "
        "from your 256×256 conditioning tensor (boundary + keepouts)."
    )

    model_path = Path("models") / "layout_cvae.pt"
    if st.session_state.cond is None:
        st.info("Build conditioning masks first (section above).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            n_candidates = st.number_input("Number of candidates", min_value=1, max_value=64, value=8, step=1)
        with c2:
            thr = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        with c3:
            device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

        if not model_path.exists():
            st.warning(
                "Model checkpoint missing. Train model first:\n\n"
                "`python -m train.train_cvae --data_dir data/synthetic --epochs 10 --batch_size 16 --lr 1e-3`\n"
            )
        else:
            if st.button("Generate candidates"):
                try:
                    # Load model.
                    model, _cfg = load_cvae(str(model_path), device=device_choice)

                    # Generate binary candidate masks.
                    cond_np = st.session_state.cond.astype(np.float32, copy=False)
                    masks = generate_candidates(model, cond_np=cond_np, n=int(n_candidates), thr=float(thr), device=device_choice)
                    st.session_state.candidates = masks
                    st.success(f"Generated {int(masks.shape[0])} candidates")
                except Exception as e:
                    st.error(f"Failed to generate candidates: {e}")

            masks = st.session_state.candidates
            if masks is not None:
                boundary = (st.session_state.boundary_mask > 0).astype(np.uint8)
                keepout = (st.session_state.keepout_mask > 0).astype(np.uint8)
                boundary_sum = float(boundary.sum())

                n = int(masks.shape[0])
                ncols = 4 if n >= 4 else 2
                rows = int(np.ceil(n / ncols))

                idx = 0
                for _r in range(rows):
                    cols = st.columns(ncols)
                    for c in cols:
                        if idx >= n:
                            break
                        m = (masks[idx] > 0).astype(np.uint8)

                        # Metrics
                        coverage = float(m.sum()) / (boundary_sum + 1e-6)
                        keepout_violation = int((m & keepout).sum())

                        # Candidate mask as an image
                        mask_img = (m * 255).astype(np.uint8)

                        # Simple overlay: boundary as gray, keepout as red, generated as green.
                        base = (boundary * 120).astype(np.uint8)
                        overlay = np.stack([base, base, base], axis=-1)
                        overlay[..., 0] = np.clip(overlay[..., 0] + keepout * 200, 0, 255)
                        overlay[..., 1] = np.clip(overlay[..., 1] + m * 200, 0, 255)

                        with c:
                            st.caption(f"cand {idx} | coverage={coverage:.3f} | keepout_violation={keepout_violation}")
                            st.image(mask_img, caption="mask", clamp=True)
                            st.image(overlay, caption="overlay", clamp=True)

                        idx += 1

                st.subheader("Auto-rank + Pick Best")
                st.caption(
                    "Scores candidates using simple PVX-like rules: maximize coverage, "
                    "and heavily penalize any keepout/outside-boundary violations."
                )

                r1, r2 = st.columns(2)
                with r1:
                    if st.button("Rank candidates"):
                        try:
                            ranking = rank_candidates(masks, boundary, keepout, compute_compactness=True)
                            st.session_state.ranking = ranking
                            st.success(f"Ranked {len(ranking)} candidates")
                        except Exception as e:
                            st.error(f"Ranking failed: {e}")
                with r2:
                    if st.button("Use Best Candidate"):
                        ranking = st.session_state.ranking
                        if not ranking:
                            st.warning("Run 'Rank candidates' first.")
                        else:
                            st.session_state.selected_candidate_idx = int(ranking[0]["candidate_id"])
                            st.success(f"Selected candidate {st.session_state.selected_candidate_idx}")

                if st.session_state.ranking:
                    st.subheader("Candidate ranking")
                    st.dataframe(st.session_state.ranking, use_container_width=True)

    st.header("Select + Export Layout (rows)")
    st.caption(
        "Select a generated candidate and vectorize it into row polylines. "
        "Rows are clipped to the boundary and removed from keepouts, then exported to GeoJSON (EPSG:4326)."
    )

    can_export_layout = (
        (st.session_state.candidates is not None)
        and (st.session_state.boundary_mask is not None)
        and (st.session_state.keepout_mask is not None)
    )
    if not can_export_layout:
        st.info("Generate candidates first (and ensure masks exist).")
    else:
        masks = st.session_state.candidates
        n = int(masks.shape[0])
        sel = st.number_input(
            "Candidate index",
            min_value=0,
            max_value=max(0, n - 1),
            value=int(st.session_state.selected_candidate_idx),
            step=1,
            key="selected_candidate_idx",
        )
        min_len = st.number_input("Min row length (grid px)", min_value=5, max_value=500, value=30, step=5)
        buffer_px = st.number_input("Keepout buffer (grid px)", min_value=0, max_value=50, value=2, step=1)

        if st.button("Vectorize + Export layout.geojson"):
            try:
                if georef is None:
                    st.warning("Georef not configured. Configure georeferencing first.")
                    return
                if img_w is None or img_h is None:
                    st.warning("Image width/height unknown. Upload an image that can be read for preview/export.")
                    return

                boundary_u8 = st.session_state.boundary_mask.astype(np.uint8, copy=False)
                keepout_u8 = st.session_state.keepout_mask.astype(np.uint8, copy=False)
                cand_u8 = (masks[int(sel)] > 0).astype(np.uint8)

                boundary_polys = _mask_to_polygons_u8(boundary_u8)
                if not boundary_polys:
                    st.error("Boundary mask produced no polygon. Rebuild masks.")
                    return
                boundary_poly = max(boundary_polys, key=lambda p: p.area)

                keepout_polys = _mask_to_polygons_u8(keepout_u8)

                row_lines = mask_to_row_lines(cand_u8, min_len=int(min_len))
                row_lines_grid = clip_and_clean_rows(
                    row_lines,
                    boundary_poly_grid=boundary_poly,
                    keepouts_polys_grid=keepout_polys,
                    buffer_px=int(buffer_px),
                )

                if not row_lines_grid:
                    st.warning("No row lines extracted after clipping/cleaning.")
                    overlay = _overlay_with_rows(boundary_u8, keepout_u8, cand_u8, [])
                    st.image(overlay, caption="overlay (no rows)", clamp=True)
                    return

                # Convert grid -> pixel -> WGS84
                geometries_ll = []
                properties = []
                for row_id, ls_grid in enumerate(row_lines_grid):
                    coords_ll = []
                    for gx, gy in ls_grid.coords:
                        # Use cell centers for stability.
                        px = (float(gx) + 0.5) * float(img_w) / 256.0
                        py = (float(gy) + 0.5) * float(img_h) / 256.0
                        lat, lon = georef.pixel_to_wgs84(px, py)
                        coords_ll.append((float(lon), float(lat)))

                    if len(coords_ll) >= 2:
                        geometries_ll.append(LineString(coords_ll))
                        properties.append({"type": "row", "candidate_id": int(sel), "row_id": int(row_id)})

                if not geometries_ll:
                    st.warning("Row extraction produced no valid LineStrings.")
                    return

                os.makedirs("exports", exist_ok=True)
                layout_path = os.path.join("exports", "layout.geojson")
                export_geojson(geometries_ll, layout_path, src_crs="EPSG:4326", properties=properties)

                overlay = _overlay_with_rows(boundary_u8, keepout_u8, cand_u8, row_lines_grid)
                st.image(overlay, caption="overlay (rows in blue)", clamp=True)

                with open(layout_path, "rb") as f:
                    st.download_button(
                        "Download layout.geojson",
                        data=f,
                        file_name="layout.geojson",
                        mime="application/geo+json",
                    )

                st.success(f"Exported {len(geometries_ll)} row LineStrings")
            except Exception as e:
                st.error(f"Layout export failed: {e}")

    st.header("4) Export")
    if st.button("Build + Export GeoJSON"):
        try:
            if georef is None:
                st.warning(
                    "Georef not configured (calibration is missing/degenerate). "
                    "Fix calibration to enable exports."
                )
                return

            pts_px = _parse_vertices(vertices_text)
            if len(pts_px) < 3:
                st.error("Need at least 3 vertices.")
                return

            # Convert each pixel vertex to WGS84.
            # pixel_to_wgs84 returns (lat, lon). Shapely expects (x, y) = (lon, lat).
            pts_ll = []
            for x, y in pts_px:
                lat, lon = georef.pixel_to_wgs84(x, y)
                pts_ll.append((lon, lat))

            boundary = Polygon(pts_ll)
            if not boundary.is_valid:
                boundary = boundary.buffer(0)
            if boundary.is_empty or boundary.area == 0:
                st.error("Boundary polygon is invalid/empty. Check your vertices.")
                return

            os.makedirs("exports", exist_ok=True)
            boundary_path = os.path.join("exports", "site_boundary.geojson")

            # Input is already WGS84, so src_crs is EPSG:4326.
            export_geojson(boundary, boundary_path, src_crs="EPSG:4326", properties={"type": "site_boundary"})

            keepouts_path = os.path.join("exports", "keepouts.geojson")
            keepout_geoms = []
            keepout_props = []
            if keepout_polys_px and not keepout_parse_error:
                for idx, pts in enumerate(keepout_polys_px):
                    # Validate polygon
                    if len(pts) < 3:
                        continue

                    pts_ll = []
                    for x, y in pts:
                        lat, lon = georef.pixel_to_wgs84(x, y)
                        pts_ll.append((lon, lat))

                    poly = Polygon(pts_ll)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty or poly.area == 0:
                        continue

                    keepout_geoms.append(poly)
                    keepout_props.append({"type": "keepout", "id": int(idx)})

            if keepout_geoms:
                export_geojson(keepout_geoms, keepouts_path, src_crs="EPSG:4326", properties=keepout_props)
            else:
                # If no keepouts were provided, we don't export a keepouts file.
                keepouts_path = None

            with open(boundary_path, "rb") as f:
                st.download_button(
                    "Download site_boundary.geojson",
                    data=f,
                    file_name="site_boundary.geojson",
                    mime="application/geo+json",
                )

            if keepouts_path is not None:
                with open(keepouts_path, "rb") as f:
                    st.download_button(
                        "Download keepouts.geojson",
                        data=f,
                        file_name="keepouts.geojson",
                        mime="application/geo+json",
                    )
            else:
                st.info("No keepouts exported (none provided or none valid).")

            st.success("Exported GeoJSON")
        except Exception as e:
            st.error(f"Failed: {e}")


if __name__ == "__main__":
    main()
