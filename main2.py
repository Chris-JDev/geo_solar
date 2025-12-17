import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# =========================================
# 1) INPUTS – EDIT THESE FOR SCENARIOS
# =========================================

# Example non-rectangular site polygon (meters)
# You can change these coordinates to any shape you like
site_coords = [
    (0, 0),
    (220, 0),
    (210, 110),
    (10, 100)
]
site = Polygon(site_coords)

# Module & table parameters
module_width = 1.1           # m (east-west)
module_height = 2.2          # m (north-south)
modules_per_table_vertical = 2
modules_per_table_horizontal = 1
module_power_kw = 0.55       # 550 W per module

# Layout parameters
row_pitch = 6.0              # m (front-to-front row spacing)
tilt_deg = 25.0              # panel tilt
edge_setback = 3.0           # m from site boundary

# Very simple energy model:
# specific_yield = kWh per kWp per year
#   ~1200–1400 kWh/kWp in Germany/UK
#   ~1600–1900 kWh/kWp in good sun locations
specific_yield_kwh_per_kwp = 1700.0


# =========================================
# 2) BUILD USABLE AREA
# =========================================

usable = site.buffer(-edge_setback)
if usable.is_empty:
    raise ValueError("Setback too large – usable area disappeared!")

if usable.geom_type == "MultiPolygon":
    usable_geom = unary_union(usable)
else:
    usable_geom = usable

# Derived table dimensions
table_width = module_width * modules_per_table_horizontal
table_height = module_height * modules_per_table_vertical


# =========================================
# 3) LAYOUT ALGORITHM (FILL WITH TABLES)
# =========================================

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
        x += table_width + 0.1  # a small horizontal gap
    y += row_pitch


# =========================================
# 4) METRICS
# =========================================

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

# Very simple annual energy estimate
annual_kwh = dc_capacity_kwp * specific_yield_kwh_per_kwp
capacity_factor = annual_kwh / (dc_capacity_kwp * 8760) if dc_capacity_kwp > 0 else 0.0

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


# =========================================
# 5) PLOT
# =========================================

fig, ax = plt.subplots(figsize=(10, 5))

# Site boundary
xs, ys = site.exterior.xy
ax.plot(xs, ys, linewidth=2, label="Site boundary")

# Usable area
ux, uy = usable_geom.exterior.xy
ax.plot(ux, uy, linestyle="--", label="Usable area")

# Panel tables
for t in panel_tables:
    x, y = t.exterior.xy
    ax.fill(x, y, alpha=0.6, linewidth=0.2)

ax.set_aspect("equal", "box")
ax.set_xlabel("Meters (E-W)")
ax.set_ylabel("Meters (N-S)")
ax.set_title("Auto-generated solar layout (irregular site demo)")

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

ax.text(
    0.01, 0.99,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox=dict(boxstyle="round", alpha=0.7)
)

ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
