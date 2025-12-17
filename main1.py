import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# =========================
# 1) SITE & PARAMETERS
# =========================

# Example non-rectangular site polygon (meters)
site = Polygon([
    (0, 0),
    (220, 0),
    (210, 110),
    (10, 100)
])

# Module & table parameters
module_width = 1.1          # m (east-west)
module_height = 2.2         # m (north-south)
modules_per_table_vertical = 2
modules_per_table_horizontal = 1
module_power_kw = 0.55      # 550 W per module

# Layout parameters
row_pitch = 6.0             # m (front-to-front row spacing)
tilt_deg = 25.0             # panel tilt
edge_setback = 3.0          # m from site boundary

# Derived dimensions
table_width = module_width * modules_per_table_horizontal
table_height = module_height * modules_per_table_vertical

# =========================
# 2) BUILD USABLE AREA
# =========================

# Shrink polygon by setback to get usable area
usable = site.buffer(-edge_setback)
if usable.is_empty:
    raise ValueError("Setback too large – usable area disappeared!")

# Make sure we work with polygons (union multiparts)
if usable.geom_type == "MultiPolygon":
    usable_geom = unary_union(usable)
else:
    usable_geom = usable

# =========================
# 3) LAYOUT ALGORITHM
# =========================

panel_tables = []

minx, miny, maxx, maxy = usable_geom.bounds

# Sweep rows from bottom to top
y = miny + row_pitch
while y + table_height <= maxy:
    # Row strip for this y (we'll clip panels to usable area)
    x = minx + table_width
    while x + table_width <= maxx:
        # Build a table polygon
        table = Polygon([
            (x, y),
            (x + table_width, y),
            (x + table_width, y + table_height),
            (x, y + table_height)
        ])
        # Only keep if the centroid is inside usable area
        if table.centroid.within(usable_geom):
            panel_tables.append(table)
        x += table_width + 0.1  # small horizontal gap
    y += row_pitch

# =========================
# 4) METRICS
# =========================

n_tables = len(panel_tables)
n_modules = n_tables * modules_per_table_vertical * modules_per_table_horizontal
dc_capacity_kwp = n_modules * module_power_kw

site_area_m2 = site.area
site_area_ha = site_area_m2 / 10_000.0
kwp_per_ha = dc_capacity_kwp / site_area_ha if site_area_ha > 0 else 0.0

# Simple shading proxy: Ground Coverage Ratio (GCR)
tilt_rad = math.radians(tilt_deg)
projected_height = table_height * math.cos(tilt_rad)
gcr = projected_height / row_pitch

print(f"Tables:   {n_tables}")
print(f"Modules:  {n_modules}")
print(f"DC size:  {dc_capacity_kwp:.1f} kWp")
print(f"Area:     {site_area_ha:.2f} ha")
print(f"kWp/ha:   {kwp_per_ha:.1f}")
print(f"GCR:      {gcr:.2f}")

# =========================
# 5) PLOT
# =========================

fig, ax = plt.subplots(figsize=(8, 4))

# Plot site boundary
xs, ys = site.exterior.xy
ax.plot(xs, ys, linewidth=2)

# Plot usable area
ux, uy = usable_geom.exterior.xy
ax.plot(ux, uy, linestyle="--")

# Plot all panel tables
for t in panel_tables:
    x, y = t.exterior.xy
    ax.fill(x, y, alpha=0.6, linewidth=0.2)

ax.set_aspect("equal", "box")
ax.set_xlabel("Meters (E-W)")
ax.set_ylabel("Meters (N-S)")
ax.set_title("Auto-generated solar layout (irregular site demo)")

# Add metrics as a text box
metrics_text = (
    f"DC: {dc_capacity_kwp:.1f} kWp\n"
    f"Modules: {n_modules}\n"
    f"Area: {site_area_ha:.2f} ha\n"
    f"kWp/ha: {kwp_per_ha:.1f}\n"
    f"GCR: {gcr:.2f} (tilt {tilt_deg}°)"
)

ax.text(
    0.01, 0.99,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox=dict(boxstyle="round", alpha=0.5)
)

plt.tight_layout()
plt.show()
