"""
config.py — Project Configuration
===================================
Single source of truth for:
  - Data paths (never hardcoded elsewhere)
  - Area of Interest (AOI) municipalities
  - Default scoring weights
  - Spatial analysis parameters

To override settings locally, create a .env file in the project root.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env overrides (optional — won't fail if file is missing)
load_dotenv()

# ── Project Root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Directories ────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EXTERNAL_DIR   = DATA_DIR / "external"

# ── Raw Input File Paths ────────────────────────────────────────────────────────
# These should point to real data once available.
# For now they serve as documented expectations.
RAW_TRANSACTIONS_PATH = RAW_DIR / os.getenv(
    "TRANSACTIONS_FILE", "real_estate_transactions.csv"
)
RAW_NEIGHBORHOODS_PATH = RAW_DIR / os.getenv(
    "NEIGHBORHOODS_FILE", "neighborhoods.geojson"
)
RAW_TRANSIT_STOPS_PATH = RAW_DIR / os.getenv(
    "TRANSIT_FILE", "transit_stops.geojson"
)
RAW_PARKS_PATH = RAW_DIR / os.getenv(
    "PARKS_FILE", "parks.geojson"
)

# ── Processed Output File Paths (GeoParquet) ─────────────────────────────────
PROCESSED_TRANSACTIONS_PATH  = PROCESSED_DIR / "transactions.parquet"
PROCESSED_NEIGHBORHOODS_PATH = PROCESSED_DIR / "neighborhoods.parquet"
PROCESSED_TRANSIT_PATH       = PROCESSED_DIR / "transit_stops.parquet"
PROCESSED_PARKS_PATH         = PROCESSED_DIR / "parks.parquet"
NEIGHBORHOOD_SCORES_PATH     = PROCESSED_DIR / "neighborhood_scores.parquet"

# ── CRS Settings ────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:4326"   # WGS84 — universal standard for this project
METRIC_CRS = "EPSG:2039"   # ITM — used locally for metric distance calculations

# ── Area of Interest (AOI) — Gush Dan Metropolitan Area ─────────────────────────
# These are the ONLY municipalities supported in the MVP.
# The architecture is designed to expand to all of Israel in future phases.
GUSH_DAN_MUNICIPALITIES = [
    "Tel Aviv-Yafo",
    "Ramat Gan",
    "Givatayim",
    "Bnei Brak",
    "Holon",
    "Bat Yam",
    "Petah Tikva",
    "Rishon LeZion",
    "Herzliya",
    "Ramat HaSharon",
    "Kiryat Ono",
    "Givat Shmuel",
]

# Approximate bounding box for Gush Dan [min_lon, min_lat, max_lon, max_lat]
# Used for spatial filtering and map initialization
GUSH_DAN_BBOX = {
    "min_lon": 34.70,
    "min_lat": 31.90,
    "max_lon": 34.95,
    "max_lat": 32.20,
}

# Map center for Folium initialization
GUSH_DAN_CENTER = {
    "lat": 32.05,
    "lon": 34.82,
}

# ── Spatial Analysis Parameters ─────────────────────────────────────────────────
TRANSIT_RADIUS_M     = 500    # Buffer radius (meters) for transit stop density
PARK_RADIUS_M        = 1_000  # Buffer radius (meters) for park accessibility
MIN_TRANSACTIONS     = 3      # Minimum transactions to include a neighborhood

# ── Default Scoring Weights ──────────────────────────────────────────────────────
# These are overridable from the Streamlit sidebar.
# Must sum to 1.0 (enforced by the scoring engine).
DEFAULT_WEIGHTS = {
    "affordability": 0.40,   # Lower price = higher score
    "transit":       0.35,   # More/closer transit stops = higher score
    "parks":         0.25,   # More/closer parks = higher score
}

# ── Scoring Normalization ────────────────────────────────────────────────────────
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# ── Streamlit UI Defaults ────────────────────────────────────────────────────────
UI_DEFAULT_BUDGET_ILS       = 3_000_000   # ₪3M
UI_DEFAULT_MAX_COMMUTE_MIN  = 30          # minutes
UI_DEFAULT_TRANSIT_WEIGHT   = 35          # slider 0-100
UI_DEFAULT_PARKS_WEIGHT     = 25          # slider 0-100

# ── OSMnx / External API Settings ───────────────────────────────────────────────
# Used when fetching data from OpenStreetMap
OSM_NETWORK_TYPE = "walk"       # Walking network for transit accessibility
OSM_TAGS_TRANSIT = {"public_transport": "stop_position"}
OSM_TAGS_PARKS   = {"leisure": "park"}
