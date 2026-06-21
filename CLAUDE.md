# CLAUDE.md — Development Guidelines
## Where Should I Buy? · GeoAI Course Project

This file contains mandatory coding standards for this project.
Follow these rules in every code change — no exceptions.

---

## 🌍 CRS (Coordinate Reference System) Rules

### The One Rule: Always EPSG:4326

**All GeoDataFrames must be in EPSG:4326 (WGS84) at all times.**

```python
# ✅ CORRECT — convert immediately after loading
gdf = gpd.read_file("data.geojson")
gdf = to_wgs84(gdf)  # Always do this

# ❌ WRONG — never leave a GDF in an unknown or projected CRS
gdf = gpd.read_file("data.geojson")
# ... using gdf without checking CRS ...
```

### Enforce with assertions at function boundaries

```python
# ✅ Add this at the top of every function that accepts a GeoDataFrame
from src.geo.crs_utils import assert_wgs84

def my_function(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    assert_wgs84(gdf, "my_function input")
    # ... rest of function
```

### Metric distance calculations: reproject locally, don't persist

```python
# ✅ CORRECT — reproject locally, return WGS84
from src.geo.crs_utils import reproject_for_metric, to_wgs84

gdf_m = reproject_for_metric(gdf)           # ITM (EPSG:2039)
gdf_m["dist_m"] = gdf_m.geometry.distance(other_geom_itm)
gdf = to_wgs84(gdf_m)                       # Back to WGS84

# ❌ WRONG — persisting a projected GDF
gdf_itm = gdf.to_crs("EPSG:2039")
return gdf_itm  # Never return a projected GDF
```

---

## 💾 Storage Format

### Use GeoParquet for all processed data

```python
# ✅ Save
gdf.to_parquet("data/processed/neighborhoods.parquet")

# ✅ Load
gdf = gpd.read_parquet("data/processed/neighborhoods.parquet")

# ❌ Avoid CSV for geospatial data (loses geometry)
# ❌ Avoid Shapefile (column name truncation, multi-file format)
```

GeoParquet advantages:
- Stores geometry natively (no WKT/WKB conversion)
- Preserves CRS metadata
- 3–10× faster than GeoJSON for large files
- Single file format

---

## 🌳 Spatial Indexes — Always Use R-tree

Never loop over geometries for spatial operations. Use vectorised methods.

```python
# ✅ CORRECT — uses R-tree index internally
joined = gpd.sjoin_nearest(source, target, distance_col="dist_m")

# ✅ CORRECT — buffered spatial join with R-tree
joined = gpd.sjoin(source_buffered, target, predicate="contains")

# ❌ WRONG — O(n²) brute-force loop, never do this
for _, row in source.iterrows():
    for _, other in target.iterrows():
        dist = row.geometry.distance(other.geometry)
```

---

## 📐 Spatial Joins

Always prefer `sjoin` / `sjoin_nearest` over manual loops.

```python
# ✅ Correct pattern for spatial join
from src.geo.crs_utils import safe_spatial_join

result = safe_spatial_join(left_gdf, right_gdf, how="left", predicate="within")

# Always check for unmatched rows after left joins
unmatched = result["index_right"].isna().sum()
if unmatched:
    logger.warning(f"{unmatched} rows did not match any polygon.")
```

---

## 🔧 Function Design

### Keep functions small and testable

**One function = one responsibility.**

```python
# ✅ CORRECT — small, testable, single purpose
def compute_distance_to_nearest_park(
    points: gpd.GeoDataFrame,
    parks:  gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Returns points GDF with 'dist_to_park_m' column added."""
    assert_wgs84(points, "points")
    assert_wgs84(parks, "parks")
    # ... 10-20 lines of logic ...

# ❌ WRONG — 200-line function doing everything
def process_all_data(raw_path, output_path, weights, ...):
    # loads, cleans, joins, scores, saves — untestable monolith
```

### Type hints everywhere

```python
# ✅ Always annotate function signatures
def score_neighborhoods(
    gdf: gpd.GeoDataFrame,
    weights: ScoringWeights,
    budget: float | None = None,
) -> gpd.GeoDataFrame:
```

---

## 📦 Path Configuration

**Never hardcode paths.** Always use `src/config.py`.

```python
# ✅ CORRECT
from src.config import RAW_TRANSACTIONS_PATH, PROCESSED_NEIGHBORHOODS_PATH

gdf = gpd.read_file(RAW_TRANSACTIONS_PATH)
gdf.to_parquet(PROCESSED_NEIGHBORHOODS_PATH)

# ❌ WRONG
gdf = gpd.read_file("../../data/raw/transactions.csv")
```

---

## 📝 Logging

Use `loguru` (not `print`).

```python
from loguru import logger

logger.info("Processing 500 transactions.")
logger.warning("12 transactions outside AOI — skipping.")
logger.error("Missing required column 'neighborhood'.")
logger.debug(f"CRS: {gdf.crs}")
```

---

## 🧪 Testing Expectations

Every `src/` module should be independently importable and testable:

```python
# Quick smoke test pattern for any module
from src.data.loaders import load_transactions
gdf = load_transactions()
assert gdf.crs.to_epsg() == 4326
assert len(gdf) > 0
print("✅ load_transactions OK")
```

---

## 🗺️ Folium Maps

- Always set `location` from `GUSH_DAN_CENTER` in config.
- Use `folium.FeatureGroup` for toggleable layers.
- Always add `folium.LayerControl()`.
- Keep popup HTML minimal and readable.
- Default basemap: `CartoDB positron` (clean, light).

---

## 📊 Streamlit Patterns

- Cache expensive operations with `@st.cache_data`.
- Prefix GeoDataFrame parameters with `_` in cached functions (skips hashing).
- Keep UI logic in `streamlit_app.py`, keep geospatial logic in `src/`.
- Never call `gpd.read_file` directly from `streamlit_app.py` — use loaders.

---

## 🚫 Things to Avoid

| Avoid | Instead |
|---|---|
| `print()` for logging | `logger.info()` |
| Hardcoded file paths | `src/config.py` constants |
| Iterating over GDF rows | Vectorised GeoPandas operations |
| Saving as CSV/Shapefile | GeoParquet |
| Mixed CRS in one pipeline | `assert_wgs84()` at every function entry |
| Persisting projected CRS | Reproject locally, return WGS84 |
| Monolithic functions | Small, single-purpose functions |
| Cloud infra in MVP | Local-first, cloud-ready architecture |

---

## 📁 File Conventions

| File | Convention |
|---|---|
| Raw data | `data/raw/` — never committed |
| Processed data | `data/processed/*.parquet` — never committed |
| Source modules | `src/<domain>/<module>.py` — snake_case |
| Notebooks | `notebooks/NN_description.ipynb` — numbered prefix |
| Config vars | `UPPER_SNAKE_CASE` in `config.py` |
| Function args | `lower_snake_case` |
| Classes | `PascalCase` |
