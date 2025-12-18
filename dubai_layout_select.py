import json
import math

import matplotlib.pyplot as plt
from shapely.geometry import shape, Polygon
from shapely.ops import transform, unary_union
import pyproj

# ================================
# 1) CONFIG – EDIT IF YOU WANT
# ================================

GEOJSON_PATH = "dubai.geojson"  # make sure this file is in the same folder

# Module & table parameters
module_width = 1.1           # m (east-west)
module_height = 2.2          # m (north-south)
modules_per_table_vertical = 2
modules_per_table_horizontal = 1
module_power_kw = 0.55       # 550 W per module

# Layout parameters
row_pitch = 6.0              # m row spacing
tilt_deg = 25.0              # panel tilt
edge_setback = 3.0           # m from site boundary

# Simple energy model
specific_yield_kwh_per_kwp = 1700.0  # adjust per location if you like


# ================================
# 2) LOAD & PROJECT DUBAI GEOMETRY
# ================================

def load_dubai_geometry(geojson_path: str):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # assume one feature for Dubai
    feature = gj["features"][0]
    geom_lonlat = shape(feature["geometry"])  # EPSG:4326 (lon, lat)
    return geom_lonlat

# Dubai is around lon 55E, lat 25N -> UTM zone 40N is reasonable
proj_to_utm = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:32640", always_xy=True
).transform

proj_to_lonlat = pyproj.Transformer.from_crs(
    "EPSG:32640", "EPSG:4326", always_xy=True
).transform

dubai_lonlat = load_dubai_geometry(GEOJSON_PATH)
dubai_utm = transform(proj_to_utm, dubai_lonlat)

# unify multiparts if needed
if dubai_utm.geom_type == "MultiPolygon":
    dubai_utm = unary_union(dubai_utm)


def plot_outline(ax, geom, **kwargs):
    """Plot the exterior outline of a Polygon or MultiPolygon onto ax."""
    # Polygon case
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, **kwargs)
    # MultiPolygon or collection of polygons
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            if part.geom_type == 'Polygon':
                xs, ys = part.exterior.xy
                ax.plot(xs, ys, **kwargs)
    else:
        # fallback: try to access exterior if present
        try:
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, **kwargs)
        except Exception:
            pass


# ================================
# 3) ASK USER TO SELECT A RECTANGLE
# ================================

print("Showing Dubai boundary. Click TWO points to define your site rectangle.")
print("Tip: click bottom-left and top-right (roughly).")

fig, ax = plt.subplots(figsize=(8, 6))

# plot Dubai outline in meters
plot_outline(ax, dubai_utm, linewidth=2, color='black')

ax.set_aspect("equal", "box")
ax.set_title("Dubai (UTM meters). Click 2 points for site area.")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

plt.tight_layout()
# blocking call: waits for 2 mouse clicks on the figure
pts = plt.ginput(2, timeout=-1)
plt.close(fig)

if len(pts) < 2:
    raise RuntimeError("You must click 2 points to define the site area.")

(x1, y1), (x2, y2) = pts
minx, maxx = sorted([x1, x2])
miny, maxy = sorted([y1, y2])

# Build rectangular site polygon in UTM
selected_rect = Polygon([
    (minx, miny),
    (maxx, miny),
    (maxx, maxy),
    (minx, maxy)
])

# Clip with Dubai polygon so we don't go outside
site_utm = selected_rect.intersection(dubai_utm)

if site_utm.is_empty:
    raise RuntimeError("Selected rectangle does not intersect Dubai polygon.")

print("Site selected successfully.")


# ================================
# 4) LAYOUT ALGORITHM
# ================================

def generate_layout(site_polygon: Polygon):
    from shapely.ops import unary_union

    # shrink for setback
    usable = site_polygon.buffer(-edge_setback)
    if usable.is_empty:
        raise ValueError("Setback removed all usable area – site too small.")

    if usable.geom_type == "MultiPolygon":
        usable_geom = unary_union(usable)
    else:
        usable_geom = usable

    table_width = module_width * modules_per_table_horizontal
    table_height = module_height * modules_per_table_vertical

    panel_tables = []

    minx, miny, maxx, maxy = usable_geom.bounds

    y = miny + row_pitch
    while y + table_height <= maxy:
        x = minx + table_width
        while x + table_width <= maxx:
            table = Polygon([
                (x, y),
                (x + table_width, y),
                (x + table_width, y + table_height),
                (x, y + table_height)
            ])
            if table.centroid.within(usable_geom):
                panel_tables.append(table)
            x += table_width + 0.1  # small horizontal gap
        y += row_pitch

    return usable_geom, panel_tables

usable_utm, panel_tables = generate_layout(site_utm)

# ================================
# 5) METRICS
# ================================

n_tables = len(panel_tables)
n_modules = n_tables * modules_per_table_vertical * modules_per_table_horizontal
dc_capacity_kwp = n_modules * module_power_kw

site_area_m2 = site_utm.area
site_area_ha = site_area_m2 / 10_000.0
kwp_per_ha = dc_capacity_kwp / site_area_ha if site_area_ha > 0 else 0.0

tilt_rad = math.radians(tilt_deg)
table_height = module_height * modules_per_table_vertical
projected_height = table_height * math.cos(tilt_rad)
gcr = projected_height / row_pitch

annual_kwh = dc_capacity_kwp * specific_yield_kwh_per_kwp
capacity_factor = (
    annual_kwh / (dc_capacity_kwp * 8760)
    if dc_capacity_kwp > 0 else 0.0
)

print("===== LAYOUT METRICS =====")
print(f"Tables:            {n_tables}")
print(f"Modules:           {n_modules}")
print(f"DC size:           {dc_capacity_kwp:.1f} kWp")
print(f"Area:              {site_area_ha:.2f} ha")
print(f"kWp/ha:            {kwp_per_ha:.1f}")
print(f"GCR:               {gcr:.2f} (tilt {tilt_deg}°)")
print()
print("===== ENERGY (SIMPLE MODEL) =====")
print(f"Specific yield:    {specific_yield_kwh_per_kwp:.0f} kWh/kWp/yr")
print(f"Annual energy:     {annual_kwh:,.0f} kWh/yr")
print(f"Capacity factor:   {capacity_factor*100:.1f} %")


# ================================
# 6) PLOT LAYOUT
# ================================

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Dubai outline
plot_outline(ax2, dubai_utm, linewidth=1.0, color="grey", label="Dubai boundary")

# Selected site (may be MultiPolygon) — plot all parts
plot_outline(ax2, site_utm, linewidth=2, color="black")

# Usable area (after setback)
plot_outline(ax2, usable_utm, linestyle="--", color="orange")

# Panel tables
for t in panel_tables:
    x, y = t.exterior.xy
    ax2.fill(x, y, alpha=0.6, linewidth=0.1)

ax2.set_aspect("equal", "box")
ax2.set_xlabel("Easting (m)")
ax2.set_ylabel("Northing (m)")
ax2.set_title("Auto-generated solar layout – selected area in Dubai")

metrics_text = (
    f"DC: {dc_capacity_kwp:.1f} kWp\n"
    f"Modules: {n_modules}\n"
    f"Area: {site_area_ha:.2f} ha\n"
    f"kWp/ha: {kwp_per_ha:.1f}\n"
    f"GCR: {gcr:.2f} (tilt {tilt_deg}°)\n"
    f"Yield: {specific_yield_kwh_per_kwp:.0f} kWh/kWp/yr\n"
    f"Annual: {annual_kwh/1000:.1f} MWh/yr\n"
    f"CF: {capacity_factor*100:.1f} %"
)

ax2.text(
    0.01, 0.99,
    metrics_text,
    transform=ax2.transAxes,
    va="top",
    fontsize=8,
    bbox=dict(boxstyle="round", alpha=0.7)
)

ax2.legend(loc="lower right")
plt.tight_layout()
plt.show()
