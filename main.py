from shapely.geometry import Polygon

# Import matplotlib optionally — if it isn't installed, run without visualization.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

# 1) Define a very simple site: rectangle in meters
site = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])  # 200m x 100m

# 2) Panel + row parameters
panel_width = 1.1       # m (east-west)
panel_height = 2.0      # m (north-south)
modules_per_table = 2   # 2 high (portrait)
row_spacing = 5.0       # m between row front edges
edge_setback = 3.0      # m offset from site boundary

# 3) Compute usable area (shrink by setbacks)
usable = site.buffer(-edge_setback)

# Quick safety: if usable area is empty (too-large setback), stop early
if usable.is_empty:
    print("Usable area is empty after applying edge_setback. Adjust edge_setback and try again.")
    # Exit when run as script; but keep definitions for imports/tests
    if __name__ == "__main__":
        raise SystemExit(1)

# 4) Generate rows aligned along X-axis
rows = []
y = usable.bounds[1] + row_spacing
y_max = usable.bounds[3] - row_spacing

while y < y_max:
    # row line inside usable polygon (simple approach)
    row_line = Polygon([(usable.bounds[0], y),
                        (usable.bounds[2], y),
                        (usable.bounds[2], y + panel_height * modules_per_table),
                        (usable.bounds[0], y + panel_height * modules_per_table)])
    row_poly = row_line.intersection(usable)
    if not row_poly.is_empty:
        rows.append(row_poly)
    y += row_spacing

# 5) Place panels in each row
panel_polys = []
for row in rows:
    minx, miny, maxx, maxy = row.bounds
    x = minx
    while x + panel_width <= maxx:
        panel = Polygon([
            (x, miny),
            (x + panel_width, miny),
            (x + panel_width, miny + panel_height * modules_per_table),
            (x, miny + panel_height * modules_per_table)
        ])
        if panel.centroid.within(row):
            panel_polys.append(panel)
        x += panel_width + 0.1  # small gap

num_modules = len(panel_polys) * modules_per_table
dc_capacity_kwp = num_modules * 0.55  # assume 550 W modules

def summary():
    print(f"Rows: {len(rows)}")
    print(f"Tables: {len(panel_polys)}")
    print(f"Modules: {num_modules}")
    print(f"DC capacity: {dc_capacity_kwp:.1f} kWp")


def visualize():
    if not HAS_MPL:
        print("matplotlib not available; skipping visualization. Install with: python -m pip install matplotlib")
        return
    fig, ax = plt.subplots()
    xs, ys = site.exterior.xy
    ax.plot(xs, ys, color="black")

    for p in panel_polys:
        x, y = p.exterior.xy
        ax.fill(x, y, alpha=0.6)

    ax.set_aspect("equal", "box")
    ax.set_title("Auto-generated solar layout (flat site demo)")
    plt.show()


if __name__ == "__main__":
    summary()
    visualize()
