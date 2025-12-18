"""CRS helpers (placeholder)."""

from pyproj import CRS


def utm_crs_from_latlon(lat: float, lon: float) -> CRS:
    """Return an appropriate UTM CRS for a lat/lon (minimal placeholder)."""
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south})
