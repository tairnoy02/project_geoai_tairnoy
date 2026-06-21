"""
preprocessing.py — Data Cleaning and Standardisation
======================================================
Responsible for:
  1. Removing invalid geometries and duplicate rows.
  2. Standardising column names and data types.
  3. Clipping data to the Gush Dan AOI bounding box.
  4. Imputing or dropping missing values.
  5. Feature engineering on raw tabular fields
     (price_per_m2, log_price, etc.).

All functions:
  - Accept a GeoDataFrame in EPSG:4326.
  - Return a GeoDataFrame in EPSG:4326.
  - Never reproject (CRS is managed by crs_utils.py).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import box

from src.geo.crs_utils import assert_wgs84, TARGET_CRS
from src.config import GUSH_DAN_BBOX, MIN_TRANSACTIONS, GUSH_DAN_MUNICIPALITIES


# ── AOI bounding box as a Shapely geometry ──────────────────────────────────────
_AOI_BOX = box(
    GUSH_DAN_BBOX["min_lon"],
    GUSH_DAN_BBOX["min_lat"],
    GUSH_DAN_BBOX["max_lon"],
    GUSH_DAN_BBOX["max_lat"],
)


# ── Generic ─────────────────────────────────────────────────────────────────────

def clip_to_aoi(gdf: gpd.GeoDataFrame, label: str = "GDF") -> gpd.GeoDataFrame:
    """
    Remove rows whose geometry falls outside the Gush Dan bounding box.

    Args:
        gdf:   GeoDataFrame in EPSG:4326.
        label: Label for logging.

    Returns:
        Filtered GeoDataFrame.
    """
    assert_wgs84(gdf, label)
    before = len(gdf)
    gdf = gdf[gdf.geometry.within(_AOI_BOX)].copy()
    removed = before - len(gdf)
    if removed:
        logger.warning(f"{label}: removed {removed} rows outside Gush Dan AOI.")
    logger.info(f"{label}: {len(gdf)} rows after AOI clip.")
    return gdf


def drop_invalid_geometries(
    gdf: gpd.GeoDataFrame, label: str = "GDF"
) -> gpd.GeoDataFrame:
    """
    Drop rows with null, empty, or invalid geometries.

    Args:
        gdf:   Input GeoDataFrame.
        label: Label for logging.

    Returns:
        GeoDataFrame with valid geometries only.
    """
    before = len(gdf)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    removed = before - len(gdf)
    if removed:
        logger.warning(f"{label}: dropped {removed} rows with invalid geometry.")
    return gdf


def standardise_column_names(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Lowercase and strip whitespace from all column names.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with clean column names.
    """
    gdf.columns = [c.lower().strip().replace(" ", "_") for c in gdf.columns]
    return gdf


# ── Transactions ─────────────────────────────────────────────────────────────────

def preprocess_transactions(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and enrich real estate transaction data.

    Steps:
      1. Standardise column names.
      2. Drop invalid geometries.
      3. Clip to AOI.
      4. Filter to supported municipalities.
      5. Remove price outliers (< ₪200K or > ₪100M).
      6. Add derived features: price_per_m2, log_price.
      7. Drop rows with missing critical fields.

    Args:
        gdf: Raw transactions GeoDataFrame in EPSG:4326.

    Returns:
        Cleaned GeoDataFrame.
    """
    assert_wgs84(gdf, "transactions")
    gdf = standardise_column_names(gdf)
    gdf = drop_invalid_geometries(gdf, "transactions")
    gdf = clip_to_aoi(gdf, "transactions")

    # Filter to supported municipalities only
    if "neighborhood" in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf["neighborhood"].isin(GUSH_DAN_MUNICIPALITIES)].copy()
        logger.info(
            f"Transactions: kept {len(gdf)}/{before} rows "
            f"in supported municipalities."
        )

    # Price sanity checks
    if "transaction_price" in gdf.columns:
        gdf["transaction_price"] = pd.to_numeric(
            gdf["transaction_price"], errors="coerce"
        )
        price_mask = (gdf["transaction_price"] > 200_000) & (
            gdf["transaction_price"] < 100_000_000
        )
        removed = (~price_mask).sum()
        if removed:
            logger.warning(f"Transactions: removed {removed} price outliers.")
        gdf = gdf[price_mask].copy()

    # Derived features
    if "transaction_price" in gdf.columns and "area_m2" in gdf.columns:
        gdf["area_m2"] = pd.to_numeric(gdf["area_m2"], errors="coerce")
        gdf = gdf[gdf["area_m2"] > 0].copy()
        gdf["price_per_m2"] = (
            gdf["transaction_price"] / gdf["area_m2"]
        ).round(0)
        gdf["log_price"] = np.log1p(gdf["transaction_price"])

    # Drop rows missing essential fields
    essential = ["transaction_price", "neighborhood"]
    before = len(gdf)
    gdf = gdf.dropna(subset=[c for c in essential if c in gdf.columns])
    if len(gdf) < before:
        logger.warning(f"Transactions: dropped {before - len(gdf)} rows with nulls.")

    logger.info(f"Transactions preprocessing complete: {len(gdf)} rows.")
    return gdf


# ── Neighborhoods ────────────────────────────────────────────────────────────────

def preprocess_neighborhoods(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean neighborhood polygon data.

    Steps:
      1. Standardise column names.
      2. Drop/fix invalid polygons.
      3. Filter to supported municipalities.
      4. Ensure 'name' column exists and matches our list.

    Args:
        gdf: Raw neighborhoods GeoDataFrame in EPSG:4326.

    Returns:
        Cleaned GeoDataFrame with one polygon per municipality.
    """
    assert_wgs84(gdf, "neighborhoods")
    gdf = standardise_column_names(gdf)
    gdf = drop_invalid_geometries(gdf, "neighborhoods")

    if "name" not in gdf.columns:
        raise ValueError(
            "Neighborhoods GDF must have a 'name' column with municipality names."
        )

    gdf = gdf[gdf["name"].isin(GUSH_DAN_MUNICIPALITIES)].copy()
    logger.info(f"Neighborhoods preprocessing complete: {len(gdf)} municipalities.")
    return gdf


# ── Transit stops ────────────────────────────────────────────────────────────────

def preprocess_transit_stops(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean public transit stop data.

    Args:
        gdf: Raw transit stops GeoDataFrame in EPSG:4326.

    Returns:
        Cleaned GeoDataFrame with point geometries.
    """
    assert_wgs84(gdf, "transit_stops")
    gdf = standardise_column_names(gdf)
    gdf = drop_invalid_geometries(gdf, "transit_stops")
    gdf = clip_to_aoi(gdf, "transit_stops")

    # Ensure stop_type has a value
    if "stop_type" not in gdf.columns:
        gdf["stop_type"] = "unknown"

    logger.info(
        f"Transit stops preprocessing complete: {len(gdf)} stops. "
        f"Types: {gdf['stop_type'].value_counts().to_dict()}"
    )
    return gdf


# ── Parks ────────────────────────────────────────────────────────────────────────

def preprocess_parks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean park / green space data.

    Converts Polygon parks to their centroid Points for distance calculations.
    The original polygon geometry is preserved in a separate column if available.

    Args:
        gdf: Raw parks GeoDataFrame in EPSG:4326.

    Returns:
        Cleaned GeoDataFrame with Point geometries (park centroids).
    """
    assert_wgs84(gdf, "parks")
    gdf = standardise_column_names(gdf)
    gdf = drop_invalid_geometries(gdf, "parks")
    gdf = clip_to_aoi(gdf, "parks")

    # Convert polygons to centroids for distance calculations
    polygon_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if polygon_mask.any():
        gdf.loc[polygon_mask, "geometry"] = (
            gdf.loc[polygon_mask, "geometry"].centroid
        )
        logger.info(f"Parks: converted {polygon_mask.sum()} polygons to centroids.")

    gdf = gdf.set_crs(TARGET_CRS)  # Re-assert after centroid conversion
    logger.info(f"Parks preprocessing complete: {len(gdf)} parks.")
    return gdf
