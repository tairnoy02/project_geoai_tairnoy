"""
loaders.py — Data Loading Layer
================================
Responsible for:
  1. Loading raw data from disk (CSV, GeoJSON, GeoParquet).
  2. Generating mock/sample data when real data is unavailable.
  3. Ensuring all outputs are GeoDataFrames in EPSG:4326.

Design principle:
  Every loader follows the same contract:
    - Input:  a file path (from config.py) or None
    - Output: GeoDataFrame in EPSG:4326
    - Fallback: if path doesn't exist, return mock data with a warning

Future extension:
  Add an OSM loader that fetches real transit/park data using osmnx.
"""

from __future__ import annotations

import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import Point, Polygon

from src.geo.crs_utils import to_wgs84, crs_summary, TARGET_CRS
from src.config import (
    GUSH_DAN_MUNICIPALITIES,
    GUSH_DAN_BBOX,
    RAW_TRANSACTIONS_PATH,
    RAW_NEIGHBORHOODS_PATH,
    RAW_TRANSIT_STOPS_PATH,
    RAW_PARKS_PATH,
    PROCESSED_TRANSACTIONS_PATH,
    PROCESSED_NEIGHBORHOODS_PATH,
    PROCESSED_TRANSIT_PATH,
    PROCESSED_PARKS_PATH,
)

# ── Seed for reproducible mock data ────────────────────────────────────────────
MOCK_SEED = 42
rng = np.random.default_rng(MOCK_SEED)


# ── Generic helpers ─────────────────────────────────────────────────────────────

def _random_points_in_bbox(n: int) -> list[Point]:
    """Generate n random Points within the Gush Dan bounding box."""
    lons = rng.uniform(GUSH_DAN_BBOX["min_lon"], GUSH_DAN_BBOX["max_lon"], n)
    lats = rng.uniform(GUSH_DAN_BBOX["min_lat"], GUSH_DAN_BBOX["max_lat"], n)
    return [Point(lon, lat) for lon, lat in zip(lons, lats)]


def _load_or_fallback(
    path: Path,
    mock_fn,
    label: str,
) -> gpd.GeoDataFrame:
    """
    Try to load a GeoParquet or GeoJSON file; fall back to mock data.

    Args:
        path:    Path to the file to load.
        mock_fn: Callable that returns mock GeoDataFrame.
        label:   Human-readable name for logging.

    Returns:
        GeoDataFrame in EPSG:4326.
    """
    if path.exists():
        logger.info(f"Loading {label} from {path}")
        if path.suffix == ".parquet":
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path)
        gdf = to_wgs84(gdf)
        logger.info(crs_summary(gdf, label))
        return gdf
    else:
        logger.warning(
            f"{label} file not found at {path}. "
            f"Using mock data — run the data pipeline to load real data."
        )
        return mock_fn()


# ── Real estate transactions ────────────────────────────────────────────────────

def load_transactions(path: Path | None = None) -> gpd.GeoDataFrame:
    """
    Load real estate transaction data.

    Expected columns:
        address         (str)   : Street address
        neighborhood    (str)   : Neighborhood/municipality name
        transaction_price (int) : Sale price in NIS
        rooms           (float) : Number of rooms
        area_m2         (float) : Apartment size in m²
        geometry        (Point) : Location in EPSG:4326

    Args:
        path: Override path. Defaults to config value.

    Returns:
        GeoDataFrame with transaction points in EPSG:4326.
    """
    resolved = path or RAW_TRANSACTIONS_PATH
    return _load_or_fallback(resolved, _mock_transactions, "Transactions")


def _mock_transactions() -> gpd.GeoDataFrame:
    """
    Generate realistic-looking mock transaction data for Gush Dan.

    Prices follow rough real-world distributions per municipality.
    """
    n = 500  # total mock transactions

    # Price ranges per neighborhood (₪ per m²) — rough 2024 approximations
    price_per_m2 = {
        "Tel Aviv-Yafo":    (45_000, 80_000),
        "Ramat Gan":        (30_000, 50_000),
        "Givatayim":        (35_000, 55_000),
        "Bnei Brak":        (20_000, 35_000),
        "Holon":            (18_000, 30_000),
        "Bat Yam":          (16_000, 28_000),
        "Petah Tikva":      (18_000, 32_000),
        "Rishon LeZion":    (18_000, 30_000),
        "Herzliya":         (30_000, 55_000),
        "Ramat HaSharon":   (28_000, 48_000),
        "Kiryat Ono":       (22_000, 38_000),
        "Givat Shmuel":     (22_000, 36_000),
    }

    municipalities = rng.choice(GUSH_DAN_MUNICIPALITIES, size=n)
    areas = rng.uniform(45, 140, n)  # m²
    rooms = np.round(rng.uniform(2, 5, n) * 2) / 2  # 2.0, 2.5, 3.0 ... 5.0

    prices = []
    for muni, area in zip(municipalities, areas):
        lo, hi = price_per_m2[muni]
        ppm2 = rng.uniform(lo, hi)
        prices.append(int(ppm2 * area))

    geometries = _random_points_in_bbox(n)

    gdf = gpd.GeoDataFrame(
        {
            "address":           [f"{int(rng.uniform(1, 150))} Mock St, {m}" for m in municipalities],
            "neighborhood":      municipalities,
            "transaction_price": prices,
            "rooms":             rooms,
            "area_m2":           areas.round(1),
            "price_per_m2":      [p / a for p, a in zip(prices, areas)],
        },
        geometry=geometries,
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} mock transactions.")
    return gdf


# ── Neighborhood polygons ───────────────────────────────────────────────────────

def load_neighborhoods(path: Path | None = None) -> gpd.GeoDataFrame:
    """
    Load neighborhood/municipality boundary polygons.

    Expected columns:
        name     (str)     : Municipality name (must match GUSH_DAN_MUNICIPALITIES)
        geometry (Polygon) : Boundary in EPSG:4326

    Args:
        path: Override path. Defaults to config value.

    Returns:
        GeoDataFrame with polygon boundaries in EPSG:4326.
    """
    resolved = path or RAW_NEIGHBORHOODS_PATH
    return _load_or_fallback(resolved, _mock_neighborhoods, "Neighborhoods")


def _mock_neighborhoods() -> gpd.GeoDataFrame:
    """
    Generate mock neighborhood polygons as grid-like rectangles across Gush Dan.

    In production, replace with real boundaries from:
      - Israeli CBS (Central Bureau of Statistics) statistical areas
      - OpenStreetMap administrative boundaries
    """
    # Tile the bounding box into a 3×4 grid, one cell per municipality
    cols, rows = 4, 3
    lon_step = (GUSH_DAN_BBOX["max_lon"] - GUSH_DAN_BBOX["min_lon"]) / cols
    lat_step = (GUSH_DAN_BBOX["max_lat"] - GUSH_DAN_BBOX["min_lat"]) / rows

    polygons = []
    for i, muni in enumerate(GUSH_DAN_MUNICIPALITIES):
        row = i // cols
        col = i % cols
        min_lon = GUSH_DAN_BBOX["min_lon"] + col * lon_step
        min_lat = GUSH_DAN_BBOX["min_lat"] + row * lat_step
        max_lon = min_lon + lon_step
        max_lat = min_lat + lat_step

        # Slightly shrink cells so borders are visible on the map
        pad = 0.005
        polygons.append(
            Polygon([
                (min_lon + pad, min_lat + pad),
                (max_lon - pad, min_lat + pad),
                (max_lon - pad, max_lat - pad),
                (min_lon + pad, max_lat - pad),
            ])
        )

    gdf = gpd.GeoDataFrame(
        {
            "name":       GUSH_DAN_MUNICIPALITIES,
            "population": rng.integers(50_000, 500_000, len(GUSH_DAN_MUNICIPALITIES)),
        },
        geometry=polygons,
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} mock neighborhood polygons.")
    return gdf


# ── Public transit stops ────────────────────────────────────────────────────────

def load_transit_stops(path: Path | None = None) -> gpd.GeoDataFrame:
    """
    Load public transportation stop locations.

    Expected columns:
        stop_name  (str)   : Name of the stop
        stop_type  (str)   : 'bus', 'rail', 'light_rail'
        routes     (str)   : Comma-separated route numbers (optional)
        geometry   (Point) : Location in EPSG:4326

    Args:
        path: Override path. Defaults to config value.

    Returns:
        GeoDataFrame with stop points in EPSG:4326.

    Note:
        Real data can be fetched with:
            osmnx.features_from_place("Gush Dan", tags={"public_transport": True})
        or from the Israeli Ministry of Transport GTFS feed.
    """
    resolved = path or RAW_TRANSIT_STOPS_PATH
    return _load_or_fallback(resolved, _mock_transit_stops, "Transit Stops")


def _mock_transit_stops() -> gpd.GeoDataFrame:
    """Generate mock transit stops distributed across Gush Dan."""
    n = 300
    stop_types = rng.choice(["bus", "bus", "bus", "light_rail", "rail"], size=n)

    gdf = gpd.GeoDataFrame(
        {
            "stop_name": [f"Stop {i:03d}" for i in range(n)],
            "stop_type": stop_types,
            "routes":    [f"{rng.integers(1, 200)}" for _ in range(n)],
        },
        geometry=_random_points_in_bbox(n),
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} mock transit stops.")
    return gdf


# ── Parks and green spaces ──────────────────────────────────────────────────────

def load_parks(path: Path | None = None) -> gpd.GeoDataFrame:
    """
    Load park and green space geometries.

    Expected columns:
        name     (str)             : Park name
        area_m2  (float)           : Park area in m² (optional)
        geometry (Point | Polygon) : Location/boundary in EPSG:4326

    Args:
        path: Override path. Defaults to config value.

    Returns:
        GeoDataFrame with park geometries in EPSG:4326.

    Note:
        Real data can be fetched with:
            osmnx.features_from_place("Gush Dan", tags={"leisure": "park"})
    """
    resolved = path or RAW_PARKS_PATH
    return _load_or_fallback(resolved, _mock_parks, "Parks")


def _mock_parks() -> gpd.GeoDataFrame:
    """Generate mock park centroids distributed across Gush Dan."""
    n = 80
    gdf = gpd.GeoDataFrame(
        {
            "name":    [f"Park {i:02d}" for i in range(n)],
            "area_m2": rng.uniform(2_000, 50_000, n).round(0),
        },
        geometry=_random_points_in_bbox(n),
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} mock parks.")
    return gdf


# ── Processed data loaders ──────────────────────────────────────────────────────

def load_processed_scores(path: Path | None = None) -> pd.DataFrame | None:
    """
    Load pre-computed neighborhood scores from GeoParquet.

    Returns None if the file doesn't exist (scores not yet computed).

    Args:
        path: Override path. Defaults to config value.

    Returns:
        DataFrame with neighborhood scores, or None.
    """
    resolved = path or PROCESSED_NEIGHBORHOODS_PATH
    if not resolved.exists():
        logger.info("No pre-computed scores found. Run the scoring pipeline first.")
        return None

    gdf = gpd.read_parquet(resolved)
    logger.info(f"Loaded pre-computed scores for {len(gdf)} neighborhoods.")
    return gdf
