import json
from shapely.geometry import shape
from shapely.ops import transform, unary_union
import pyproj

GEOJSON_PATH = "dubai.geojson"

def load_dubai_geometry(geojson_path: str):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feature = gj["features"][0]
    geom_lonlat = shape(feature["geometry"])  # EPSG:4326
    return geom_lonlat

proj_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32640", always_xy=True).transform

dubai_lonlat = load_dubai_geometry(GEOJSON_PATH)
dubai_utm = transform(proj_to_utm, dubai_lonlat)
print('Initial type:', dubai_utm.geom_type)
if dubai_utm.geom_type == 'MultiPolygon':
    dubai_utm = unary_union(dubai_utm)
print('After unary_union type:', dubai_utm.geom_type)
if hasattr(dubai_utm, 'exterior'):
    print('Has exterior')
else:
    geoms = list(getattr(dubai_utm, 'geoms', []))
    print('No exterior; geoms length:', len(geoms))
