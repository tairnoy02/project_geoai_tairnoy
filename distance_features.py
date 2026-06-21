"""
distance_features.py — Spatial Distance and Density Features
=============================================================
Computes spatial features used by the scoring engine:

  1. distance_to_nearest_park        — Euclidean distance (m) to closest park
  2. distance_to_nearest_transit     — Distance (m) to closest transit stop
  3. transit_stops_within_radius     — Count of stops within N metres
  4. park_access_score               — Normalised 0–1 park accessibility score

All distance calculations use the metric ITM CRS (EPSG:2039) internally
and return results in metres. Input/output GeoDataFrames remain in EPSG:4326.

Performance note:
  All nearest-neighbour operations use GeoPandas' built-in spatial index
  (GEOS R-tree via STRtree), which scales to millions of points efficiently.
  The `sjoin_nearest` method uses this internally.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.ops import unary_union

from src.geo.crs_utils import assert_wgs84, reproject_for_metric, to_wgs84
from src.config import TRANSIT_RADIUS_M, PARK_RADIUS_M, METRIC_CRS


# ── Nearest-neighbour distances ──────────────────────────────────────────────────

def distance_to_nearest(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    result_col: str,
    source_label: str = "source",
    target_label: str = "target",
) -> gpd.GeoDataFrame:
    """
    Compute the distance from each row in source_gdf to the nearest
    geometry in target_gdf.

    Uses sjoin_nearest with a metric CRS for accurate metre-based distances.

    Args:
        source_gdf:   Points to measure FROM (EPSG:4326).
        target_gdf:   Points/polygons to measure TO (EPSG:4326).
        result_col:   Name of the output distance column (in metres).
        source_label: Label for logging.
        target_label: Label for logging.

    Returns:
        source_gdf with an additional column `result_col` in metres.
        Returns EPSG:4326.
    """
    assert_wgs84(source_gdf, source_label)
    assert_wgs84(target_gdf, target_label)

    if len(target_gdf) == 0:
        logger.warning(
            f"distance_to_nearest: {target_label} GDF is empty. "
            f"Setting {result_col} to NaN."
        )
        source_gdf = source_gdf.copy()
        source_gdf[result_col] = np.nan
        return source_gdf

    # Reproject both to metric ITM for accurate metre distances
    source_m = reproject_for_metric(source_gdf)
    target_m = reproject_for_metric(target_gdf)

    # sjoin_nearest adds 'distance' column automatically
    joined = gpd.sjoin_nearest(
        source_m[["geometry"]],
        target_m[["geometry"]],
        how="left",
        distance_col="distance",
    )

    # Handle potential duplicate rows from sjoin (take minimum distance)
    distances = joined.groupby(joined.index)["distance"].min()

    result = source_gdf.copy()
    result[result_col] = distances.values.round(1)

    n_null = result[result_col].isna().sum()
    if n_null:
        logger.warning(f"{result_col}: {n_null} nulls (no nearest found).")

    logger.debug(
        f"{result_col}: min={result[result_col].min():.0f}m, "
        f"mean={result[result_col].mean():.0f}m, "
        f"max={result[result_col].max():.0f}m"
    )
    return result  # Still EPSG:4326


# ── Count within radius ──────────────────────────────────────────────────────────

def count_within_radius(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    radius_m: float,
    result_col: str,
) -> gpd.GeoDataFrame:
    """
    Count how many target points fall within radius_m metres of each source point.

    Uses a buffered spatial join in metric coordinates for efficiency.

    Args:
        source_gdf:  Points to measure from (EPSG:4326).
        target_gdf:  Points to count (EPSG:4326).
        radius_m:    Search radius in metres.
        result_col:  Name of the output count column.

    Returns:
        source_gdf with additional integer column `result_col`.
        Returns EPSG:4326.
    """
    assert_wgs84(source_gdf, "source")
    assert_wgs84(target_gdf, "target")

    if len(target_gdf) == 0:
        logger.warning(
            f"count_within_radius: target GDF is empty. Setting {result_col} = 0."
        )
        result = source_gdf.copy()
        result[result_col] = 0
        return result

    # Project to metric CRS
    source_m = reproject_for_metric(source_gdf).copy()
    target_m = reproject_for_metric(target_gdf).copy()

    # Buffer source points by radius
    source_m["_buffer"] = source_m.geometry.buffer(radius_m)
    source_buffered = source_m.set_geometry("_buffer")

    # Spatial join: count targets within each buffer
    joined = gpd.sjoin(
        source_buffered[["_buffer"]],
        target_m[["geometry"]],
        how="left",
        predicate="contains",
    )

    counts = joined.groupby(joined.index).size()

    result = source_gdf.copy()
    result[result_col] = counts.reindex(result.index, fill_value=0).astype(int)

    logger.debug(
        f"{result_col} (r={radius_m}m): "
        f"min={result[result_col].min()}, "
        f"mean={result[result_col].mean():.1f}, "
        f"max={result[result_col].max()}"
    )
    return result


# ── Neighbourhood-level aggregation ─────────────────────────────────────────────

def compute_neighborhood_spatial_features(
    neighborhoods: gpd.GeoDataFrame,
    transit_stops: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Compute all spatial features for each neighborhood polygon.

    Uses neighborhood centroids as measurement origins.

    Features added:
      - dist_to_nearest_transit_m  : metres to closest transit stop
      - dist_to_nearest_park_m     : metres to closest park
      - transit_stops_500m         : count of stops within TRANSIT_RADIUS_M
      - transit_stops_1km          : count of stops within 1000m
      - parks_1km                  : count of parks within PARK_RADIUS_M

    Args:
        neighborhoods:  Neighborhood polygons (EPSG:4326) with 'name' column.
        transit_stops:  Transit stop points (EPSG:4326).
        parks:          Park points (EPSG:4326).

    Returns:
        neighborhoods GDF with spatial feature columns appended.
    """
    assert_wgs84(neighborhoods, "neighborhoods")
    assert_wgs84(transit_stops, "transit_stops")
    assert_wgs84(parks, "parks")

    # Work with centroids for distance calculations
    centroids = neighborhoods.copy()
    centroids["geometry"] = neighborhoods.geometry.centroid

    logger.info("Computing distance to nearest transit stop...")
    centroids = distance_to_nearest(
        centroids, transit_stops,
        result_col="dist_to_nearest_transit_m",
        source_label="neighborhood_centroids",
        target_label="transit_stops",
    )

    logger.info("Computing distance to nearest park...")
    centroids = distance_to_nearest(
        centroids, parks,
        result_col="dist_to_nearest_park_m",
        source_label="neighborhood_centroids",
        target_label="parks",
    )

    logger.info(f"Counting transit stops within {TRANSIT_RADIUS_M}m...")
    centroids = count_within_radius(
        centroids, transit_stops,
        radius_m=TRANSIT_RADIUS_M,
        result_col="transit_stops_500m",
    )

    logger.info("Counting transit stops within 1000m...")
    centroids = count_within_radius(
        centroids, transit_stops,
        radius_m=1_000,
        result_col="transit_stops_1km",
    )

    logger.info(f"Counting parks within {PARK_RADIUS_M}m...")
    centroids = count_within_radius(
        centroids, parks,
        radius_m=PARK_RADIUS_M,
        result_col="parks_1km",
    )

    # Attach computed features back to the original polygon GDF
    feature_cols = [
        "dist_to_nearest_transit_m",
        "dist_to_nearest_park_m",
        "transit_stops_500m",
        "transit_stops_1km",
        "parks_1km",
    ]
    for col in feature_cols:
        neighborhoods[col] = centroids[col].values

    logger.info("Spatial feature computation complete.")
    return neighborhoods
