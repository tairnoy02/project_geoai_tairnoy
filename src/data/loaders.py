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
    RAW_DIR,
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

# ── City lookup tables for external API fetching ────────────────────────────────
_CITY_NAMES_HE: dict[str, str] = {
    "Tel Aviv-Yafo":  "תל אביב-יפו",
    "Ramat Gan":      "רמת גן",
    "Givatayim":      "גבעתיים",
    "Bnei Brak":      "בני ברק",
    "Holon":          "חולון",
    "Bat Yam":        "בת ים",
    "Petah Tikva":    "פתח תקווה",
    "Rishon LeZion":  "ראשון לציון",
    "Herzliya":       "הרצליה",
    "Ramat HaSharon": "רמת השרון",
    "Kiryat Ono":     "קריית אונו",
    "Givat Shmuel":   "גבעת שמואל",
}

# Approximate city centroids (lon, lat) for coordinate placement
_CITY_CENTROIDS: dict[str, tuple[float, float]] = {
    "Tel Aviv-Yafo":  (34.780, 32.065),
    "Ramat Gan":      (34.835, 32.070),
    "Givatayim":      (34.812, 32.048),
    "Bnei Brak":      (34.840, 32.083),
    "Holon":          (34.795, 32.015),
    "Bat Yam":        (34.750, 31.985),
    "Petah Tikva":    (34.885, 32.090),
    "Rishon LeZion":  (34.800, 31.975),
    "Herzliya":       (34.845, 32.165),
    "Ramat HaSharon": (34.843, 32.125),
    "Kiryat Ono":     (34.855, 32.055),
    "Givat Shmuel":   (34.857, 32.075),
}

# Overpass API mirrors to try in order (first available wins)
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/",
    "https://overpass.kumi.systems/api/",
]
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
        address           (str)   : Street address
        neighborhood      (str)   : Neighborhood/municipality name
        transaction_price (int)   : Sale price in NIS
        rooms             (float) : Number of rooms
        area_m2           (float) : Apartment size in m²
        geometry          (Point) : Location in EPSG:4326

    Priority order:
      1. File at `path` (any explicit override)
      2. Cached GeoJSON at data/raw/transactions.geojson
      3. data.gov.il NADLAN API (fetched once, then cached)
      4. Mock data (offline fallback)
    """
    resolved = path or RAW_TRANSACTIONS_PATH

    # 1. Explicit file override (CSV legacy or any format)
    if resolved.exists():
        logger.info(f"Loading Transactions from {resolved}")
        if resolved.suffix == ".parquet":
            gdf = gpd.read_parquet(resolved)
        else:
            gdf = gpd.read_file(str(resolved))
        return to_wgs84(gdf)

    # 2. GeoJSON cache from a previous API fetch
    geojson_cache = RAW_DIR / "transactions.geojson"
    if geojson_cache.exists():
        logger.info(f"Loading cached transactions from {geojson_cache}")
        gdf = gpd.read_file(str(geojson_cache))
        return to_wgs84(gdf)

    # 3. Fetch from data.gov.il NADLAN API
    gdf = _fetch_transactions_from_govil()
    if gdf is not None:
        geojson_cache.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(geojson_cache), driver="GeoJSON")
        logger.info(f"Cached {len(gdf)} real transactions to {geojson_cache}")
        return gdf

    # 4. Mock fallback
    logger.warning("Using mock transaction data.")
    return _mock_transactions()


def _fetch_transactions_from_govil() -> gpd.GeoDataFrame | None:
    """
    Fetch real estate transactions from data.gov.il NADLAN API.

    Source: Israeli Land Authority open data — quarterly residential transactions.
    Coordinates are approximated with municipality centroid + Gaussian jitter
    (~1–2 km spread) since the raw records lack geocoordinates.

    Returns None if the API is unreachable or returns no usable data.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — cannot fetch NADLAN data.")
        return None

    BASE_URL = "https://data.gov.il/api/3/action"
    logger.info("Fetching real estate transactions from data.gov.il NADLAN API...")

    # Discover the most recent resource in the nadlan package
    try:
        resp = requests.get(
            f"{BASE_URL}/package_show",
            params={"id": "nadlan"},
            timeout=20,
        )
        resp.raise_for_status()
        pkg = resp.json()
    except Exception as e:
        logger.warning(f"Could not retrieve NADLAN package metadata: {e}")
        return None

    if not pkg.get("success"):
        logger.warning("NADLAN package_show returned failure.")
        return None

    resources = pkg["result"].get("resources", [])
    if not resources:
        logger.warning("No resources found in NADLAN package.")
        return None

    # Use the last resource (typically the most recent year)
    resource_id = resources[-1]["id"]
    logger.info(f"Using NADLAN resource: {resources[-1].get('name', resource_id)}")

    # Fetch up to 500 transactions per city
    all_records: list[dict] = []
    rng_jitter = np.random.default_rng(1234)

    for en_name, he_name in _CITY_NAMES_HE.items():
        try:
            resp = requests.get(
                f"{BASE_URL}/datastore_search",
                params={
                    "resource_id": resource_id,
                    "filters": f'{{"CITYNAME": "{he_name}"}}',
                    "limit": 500,
                },
                timeout=20,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                records = result["result"].get("records", [])
                for r in records:
                    r["_city_en"] = en_name
                all_records.extend(records)
                logger.info(f"  ✓ {en_name}: {len(records)} transactions")
            else:
                logger.warning(f"  ✗ {en_name}: API returned failure")
        except Exception as e:
            logger.warning(f"  ✗ {en_name}: {e}")

    if not all_records:
        logger.warning("No transactions fetched from NADLAN API.")
        return None

    df = pd.DataFrame(all_records)

    # Parse numeric columns
    transaction_price = pd.to_numeric(df.get("DEALAMOUNT", pd.Series(dtype=float)), errors="coerce")
    area_m2 = pd.to_numeric(df.get("AREA", pd.Series(dtype=float)), errors="coerce")
    rooms = pd.to_numeric(df.get("ASSETROOMNUM", pd.Series(dtype=float)), errors="coerce")

    valid = transaction_price.notna() & (transaction_price > 100_000) & area_m2.notna() & (area_m2 > 10)
    df = df[valid].reset_index(drop=True)
    transaction_price = transaction_price[valid].reset_index(drop=True)
    area_m2 = area_m2[valid].reset_index(drop=True)
    rooms = rooms[valid].reset_index(drop=True)

    if df.empty:
        logger.warning("All NADLAN records had invalid price/area.")
        return None

    # Approximate coordinates: city centroid + Gaussian jitter (±1.5 km)
    lons, lats = [], []
    for city in df["_city_en"]:
        cx, cy = _CITY_CENTROIDS.get(city, (34.81, 32.06))
        lons.append(cx + rng_jitter.normal(0, 0.012))
        lats.append(cy + rng_jitter.normal(0, 0.010))

    address_col = df.get("FULLADRESS", pd.Series("", index=df.index))

    gdf = gpd.GeoDataFrame(
        {
            "address":           address_col.fillna("").values,
            "neighborhood":      df["_city_en"].values,
            "transaction_price": transaction_price.astype(int).values,
            "rooms":             rooms.fillna(0).values,
            "area_m2":           area_m2.round(1).values,
            "price_per_m2":      (transaction_price / area_m2).round(0).values,
        },
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )

    logger.info(
        f"Fetched {len(gdf)} real NADLAN transactions. "
        f"Price range: ₪{int(gdf['transaction_price'].min()):,} – "
        f"₪{int(gdf['transaction_price'].max()):,}"
    )
    return gdf


def _mock_transactions() -> gpd.GeoDataFrame:
    """
    Generate realistic mock transaction data for Gush Dan.

    Prices reflect 2024 Israeli real estate market values per municipality.
    Transaction points are placed within each city's approximate area using
    centroid + bounded Gaussian jitter, so spatial joins work correctly.
    """
    n = 500

    # ₪ per m² ranges — 2024 Israeli real estate market approximations
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
    areas = rng.uniform(45, 140, n)
    rooms = np.round(rng.uniform(2, 5, n) * 2) / 2

    prices = []
    for muni, area in zip(municipalities, areas):
        lo, hi = price_per_m2[muni]
        prices.append(int(rng.uniform(lo, hi) * area))

    # Place each transaction within its city using centroid + bounded jitter
    # (±0.015° ≈ 1.5 km spread, clipped to ±0.025°)
    points = []
    for muni in municipalities:
        cx, cy = _CITY_CENTROIDS.get(muni, (34.81, 32.06))
        lon = float(np.clip(cx + rng.normal(0, 0.012), cx - 0.025, cx + 0.025))
        lat = float(np.clip(cy + rng.normal(0, 0.010), cy - 0.020, cy + 0.020))
        points.append(Point(lon, lat))

    gdf = gpd.GeoDataFrame(
        {
            "address":           [f"{int(rng.uniform(1, 150))} Rehov, {m}" for m in municipalities],
            "neighborhood":      municipalities,
            "transaction_price": prices,
            "rooms":             rooms,
            "area_m2":           areas.round(1),
            "price_per_m2":      [p / a for p, a in zip(prices, areas)],
        },
        geometry=points,
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

    Priority order:
      1. File at `path` (real data)
      2. OpenStreetMap via osmnx (fetched once, cached to RAW_NEIGHBORHOODS_PATH)
      3. Approximate hardcoded polygons (offline fallback)

    Args:
        path: Override path. Defaults to config value.

    Returns:
        GeoDataFrame with polygon boundaries in EPSG:4326.
    """
    resolved = path or RAW_NEIGHBORHOODS_PATH

    if resolved.exists():
        logger.info(f"Loading Neighborhoods from {resolved}")
        gdf = gpd.read_file(str(resolved)) if resolved.suffix != ".parquet" else gpd.read_parquet(resolved)
        return to_wgs84(gdf)

    # Try real boundaries from OpenStreetMap
    gdf = _fetch_neighborhoods_from_osm()
    if gdf is not None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(resolved), driver="GeoJSON")
        logger.info(f"Cached real boundaries to {resolved}")
        return gdf

    logger.warning("Using approximate hardcoded neighborhood boundaries.")
    return _approximate_neighborhoods()


def _fetch_neighborhoods_from_osm() -> gpd.GeoDataFrame | None:
    """
    Fetch real municipality boundaries from OpenStreetMap via osmnx.

    Returns None if osmnx is unavailable or all queries fail.
    """
    try:
        import osmnx as ox
    except ImportError:
        logger.warning("osmnx not installed — cannot fetch real boundaries.")
        return None

    # Alternative name spellings tried if the primary name fails
    name_variants: dict[str, list[str]] = {
        "Tel Aviv-Yafo":   ["Tel Aviv-Yafo, Israel", "Tel Aviv, Israel"],
        "Ramat Gan":       ["Ramat Gan, Israel", "Ramat-Gan, Israel", "Ramat Gan municipality, Israel"],
        "Givatayim":       ["Givatayim, Israel"],
        "Bnei Brak":       ["Bnei Brak, Israel", "Bene Beraq, Israel"],
        "Holon":           ["Holon, Israel"],
        "Bat Yam":         ["Bat Yam, Israel"],
        "Petah Tikva":     ["Petah Tikva, Israel", "Petah Tiqwa, Israel"],
        "Rishon LeZion":   ["Rishon LeZion, Israel", "Rishon le-Zion, Israel"],
        "Herzliya":        ["Herzliya, Israel"],
        "Ramat HaSharon":  ["Ramat HaSharon, Israel", "Ramat Hasharon, Israel"],
        "Kiryat Ono":      ["Kiryat Ono, Israel", "Qiryat Ono, Israel"],
        "Givat Shmuel":    ["Givat Shmuel, Israel", "Giv'at Shemu'el, Israel"],
    }

    gdfs = []
    for muni in GUSH_DAN_MUNICIPALITIES:
        fetched = False
        for query in name_variants.get(muni, [f"{muni}, Israel"]):
            try:
                result = ox.geocode_to_gdf(query)
                result = result[["geometry"]].copy()
                result["name"] = muni
                gdfs.append(result)
                logger.info(f"  ✓ {muni}")
                fetched = True
                break
            except Exception:
                continue
        if not fetched:
            logger.warning(f"  ✗ Could not fetch boundary for {muni}")

    if not gdfs:
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Fill in any missing municipalities from approximate shapes
    fetched_names = set(combined["name"])
    missing = [m for m in GUSH_DAN_MUNICIPALITIES if m not in fetched_names]
    if missing:
        logger.warning(f"Filling {len(missing)} missing municipalities from approximate shapes: {missing}")
        approx = _approximate_neighborhoods()
        approx_missing = approx[approx["name"].isin(missing)]
        combined = gpd.GeoDataFrame(
            pd.concat([combined, approx_missing], ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )

    logger.info(f"Fetched {len(combined)} municipality boundaries from OpenStreetMap.")
    return combined


def _approximate_neighborhoods() -> gpd.GeoDataFrame:
    """
    Approximate municipality boundaries based on real geographic positions.

    Used only when osmnx is unavailable. Shapes are simplified but positioned
    and sized to roughly match actual Gush Dan municipalities.
    """
    # Coordinates derived from real municipal boundaries (simplified, non-overlapping)
    coords: dict[str, list[tuple[float, float]]] = {
        "Tel Aviv-Yafo": [
            (34.748, 31.993), (34.747, 32.015), (34.748, 32.043),
            (34.750, 32.070), (34.755, 32.092), (34.762, 32.108),
            (34.776, 32.113), (34.792, 32.108), (34.804, 32.100),
            (34.813, 32.088), (34.817, 32.072), (34.815, 32.052),
            (34.808, 32.038), (34.798, 32.025), (34.783, 32.013),
            (34.768, 32.003), (34.758, 31.997),
        ],
        "Ramat Gan": [
            (34.808, 32.038), (34.815, 32.052), (34.817, 32.072),
            (34.820, 32.090), (34.830, 32.097), (34.843, 32.099),
            (34.855, 32.092), (34.860, 32.078), (34.856, 32.063),
            (34.847, 32.048), (34.835, 32.040), (34.820, 32.038),
        ],
        "Givatayim": [
            (34.808, 32.038), (34.808, 32.053), (34.813, 32.060),
            (34.820, 32.058), (34.820, 32.048), (34.817, 32.040),
        ],
        "Bnei Brak": [
            (34.820, 32.058), (34.820, 32.072), (34.820, 32.090),
            (34.830, 32.097), (34.840, 32.108), (34.850, 32.108),
            (34.862, 32.103), (34.865, 32.090), (34.860, 32.078),
            (34.855, 32.065), (34.845, 32.060), (34.835, 32.057),
        ],
        "Holon": [
            (34.770, 32.003), (34.783, 32.013), (34.798, 32.025),
            (34.808, 32.038), (34.820, 32.038), (34.823, 32.025),
            (34.820, 32.010), (34.810, 31.998), (34.797, 31.993),
            (34.783, 31.995), (34.770, 31.998),
        ],
        "Bat Yam": [
            (34.748, 31.993), (34.758, 31.997), (34.768, 32.003),
            (34.770, 31.998), (34.783, 31.995), (34.785, 31.982),
            (34.778, 31.970), (34.762, 31.968), (34.750, 31.973),
            (34.745, 31.983),
        ],
        "Petah Tikva": [
            (34.855, 32.065), (34.860, 32.078), (34.862, 32.103),
            (34.868, 32.115), (34.878, 32.122), (34.895, 32.122),
            (34.910, 32.115), (34.915, 32.100), (34.912, 32.082),
            (34.905, 32.068), (34.890, 32.060), (34.873, 32.058),
        ],
        "Rishon LeZion": [
            (34.785, 31.982), (34.783, 31.995), (34.797, 31.993),
            (34.810, 31.998), (34.820, 32.010), (34.830, 32.010),
            (34.845, 32.005), (34.858, 31.995), (34.862, 31.978),
            (34.855, 31.960), (34.840, 31.948), (34.820, 31.943),
            (34.800, 31.945), (34.785, 31.955), (34.778, 31.968),
        ],
        "Herzliya": [
            (34.830, 32.150), (34.832, 32.165), (34.838, 32.180),
            (34.848, 32.188), (34.862, 32.188), (34.875, 32.182),
            (34.880, 32.168), (34.878, 32.153), (34.868, 32.143),
            (34.852, 32.140), (34.838, 32.143),
        ],
        "Ramat HaSharon": [
            (34.825, 32.112), (34.825, 32.130), (34.828, 32.148),
            (34.838, 32.143), (34.852, 32.140), (34.862, 32.130),
            (34.862, 32.115), (34.855, 32.108), (34.843, 32.105),
            (34.832, 32.108),
        ],
        "Kiryat Ono": [
            (34.847, 32.048), (34.856, 32.063), (34.860, 32.075),
            (34.868, 32.073), (34.873, 32.060), (34.868, 32.045),
            (34.858, 32.038), (34.847, 32.040),
        ],
        "Givat Shmuel": [
            (34.845, 32.060), (34.855, 32.065), (34.865, 32.063),
            (34.868, 32.073), (34.870, 32.082), (34.862, 32.085),
            (34.850, 32.080), (34.843, 32.070), (34.843, 32.062),
        ],
    }

    polygons = []
    names = []
    for muni in GUSH_DAN_MUNICIPALITIES:
        if muni in coords:
            polygons.append(Polygon(coords[muni]))
            names.append(muni)

    gdf = gpd.GeoDataFrame(
        {
            "name":       names,
            "population": rng.integers(50_000, 500_000, len(names)),
        },
        geometry=polygons,
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} approximate neighborhood polygons.")
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

    Priority order:
      1. File at `path` (cached real data)
      2. OpenStreetMap via osmnx (fetched once, cached to RAW_PARKS_PATH)
      3. Mock random points (offline fallback)
    """
    resolved = path or RAW_PARKS_PATH

    if resolved.exists():
        logger.info(f"Loading Parks from {resolved}")
        gdf = gpd.read_file(str(resolved)) if resolved.suffix != ".parquet" else gpd.read_parquet(resolved)
        return to_wgs84(gdf)

    gdf = _fetch_parks_from_osm()
    if gdf is not None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(resolved), driver="GeoJSON")
        logger.info(f"Cached {len(gdf)} parks to {resolved}")
        return gdf

    logger.warning("Using mock park data.")
    return _mock_parks()


def _overpass_reachable() -> bool:
    """
    Quick HTTPS probe: returns True only if overpass-api.de responds within 5 s.

    Uses a (connect, read) timeout tuple so a hung TLS handshake is also caught.
    """
    import requests as _req
    try:
        # Minimal Overpass query: count 0 nodes → tiny response, fast if server works
        _req.post(
            "https://overpass-api.de/api/interpreter",
            data="[out:json][timeout:3];out 0;",
            timeout=(4, 5),
        )
        return True
    except Exception:
        return False


def _fetch_parks_from_osm() -> gpd.GeoDataFrame | None:
    """
    Fetch real park polygons from Overpass API for the Gush Dan area.

    Uses direct HTTP requests (bypasses osmnx retry machinery) so connect and
    read timeouts are strictly enforced.  Fetches `way` geometries inline using
    `out geom`, which avoids the separate node-lookup step.

    Returns None if Overpass is unreachable or returns no usable data.
    """
    import requests as _req

    if not _overpass_reachable():
        logger.info("Overpass API unreachable — skipping OSM parks fetch.")
        return None

    query = (
        f"[out:json][timeout:25];"
        f"way[\"leisure\"=\"park\"]"
        f"({GUSH_DAN_BBOX['min_lat']},{GUSH_DAN_BBOX['min_lon']},"
        f"{GUSH_DAN_BBOX['max_lat']},{GUSH_DAN_BBOX['max_lon']});"
        f"out geom;"
    )

    for endpoint in _OVERPASS_ENDPOINTS:
        url = endpoint.rstrip("/") + "/interpreter"
        try:
            logger.info(f"Fetching parks from Overpass via {url} ...")
            resp = _req.post(url, data={"data": query}, timeout=(6, 60))
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
        except Exception as e:
            logger.warning(f"Overpass parks fetch failed ({url}): {str(e)[:100]}")
            continue

        # Parse inline way geometries → Shapely Polygons
        rows: list[dict] = []
        for elem in elements:
            if elem.get("type") != "way":
                continue
            geom_pts = elem.get("geometry", [])
            if len(geom_pts) < 3:
                continue
            try:
                poly = Polygon([(pt["lon"], pt["lat"]) for pt in geom_pts])
                if poly.is_valid and poly.area > 0:
                    name = elem.get("tags", {}).get("name", "Park")
                    rows.append({"name": name, "geometry": poly})
            except Exception:
                continue

        if not rows:
            logger.warning(f"No park polygons in response from {url}.")
            continue

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf["area_m2"] = gdf.to_crs("EPSG:2039").geometry.area.round(0)
        gdf = gdf[["name", "area_m2", "geometry"]].reset_index(drop=True)
        logger.info(f"Fetched {len(gdf)} park polygons from Overpass ({url}).")
        return gdf

    logger.warning("All Overpass endpoints failed — using improved mock parks.")
    return None


def _mock_parks() -> gpd.GeoDataFrame:
    """
    Return known major parks in Gush Dan + smaller neighbourhood parks.

    Major parks are positioned at real coordinates with realistic areas.
    Smaller parks use city-centroid jitter.
    """
    # Known major parks (lon, lat, area_m2, name)
    known_parks = [
        (34.789, 32.105, 3_800_000, "Yarkon Park (Gan HaYarkon)"),
        (34.780, 32.108, 1_200_000, "Ganei Yehoshua"),
        (34.848, 32.083,   680_000, "Ramat Gan Safari & National Park"),
        (34.763, 32.074,   120_000, "Charles Clore Park"),
        (34.775, 32.074,    50_000, "Independence Park (Gan HaAtzmaut)"),
        (34.782, 32.073,    30_000, "Kikar Rabin Square Park"),
        (34.772, 32.069,    20_000, "Meir Park (Tel Aviv)"),
        (34.826, 32.082,    40_000, "Begin Park (Ramat Gan)"),
        (34.835, 32.088,    35_000, "Katzenelson Park (Bnei Brak)"),
        (34.797, 32.014,    25_000, "Holon Park"),
        (34.750, 31.988,    18_000, "Bat Yam Promenade Park"),
        (34.889, 32.088,    60_000, "Gan Sacher (Petah Tikva)"),
        (34.805, 31.970,    45_000, "Rishon LeZion Park"),
        (34.844, 32.163,    90_000, "Herzliya Beach Park"),
        (34.843, 32.128,    40_000, "Ramat HaSharon Park"),
        (34.857, 32.053,    22_000, "Kiryat Ono Park"),
        (34.858, 32.078,    20_000, "Givat Shmuel Park"),
        (34.760, 32.078,    55_000, "Old Jaffa Park"),
        (34.793, 32.096,    80_000, "Yarkon River East Park"),
        (34.810, 32.108,    35_000, "Ramat Gan Park North"),
    ]

    names, areas, points = [], [], []
    for lon, lat, area, name in known_parks:
        names.append(name)
        areas.append(area)
        points.append(Point(lon, lat))

    # Add ~60 smaller neighbourhood parks with city-centroid jitter
    small_n = 60
    for i in range(small_n):
        muni = GUSH_DAN_MUNICIPALITIES[i % len(GUSH_DAN_MUNICIPALITIES)]
        cx, cy = _CITY_CENTROIDS.get(muni, (34.81, 32.06))
        lon = float(np.clip(cx + rng.normal(0, 0.012), cx - 0.025, cx + 0.025))
        lat = float(np.clip(cy + rng.normal(0, 0.010), cy - 0.020, cy + 0.020))
        names.append(f"Neighbourhood Park ({muni})")
        areas.append(float(rng.uniform(2_000, 15_000)))
        points.append(Point(lon, lat))

    gdf = gpd.GeoDataFrame(
        {"name": names, "area_m2": np.array(areas).round(0)},
        geometry=points,
        crs=TARGET_CRS,
    )
    logger.info(f"Generated {len(gdf)} parks ({len(known_parks)} major + {small_n} neighbourhood).")
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
