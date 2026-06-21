"""
spatial_join.py — Spatial Join Operations
==========================================
Higher-level spatial join wrappers for the project's common join patterns.

All joins enforce EPSG:4326 via crs_utils.
Joins use GeoPandas' built-in R-tree spatial index for performance.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.geo.crs_utils import assert_wgs84, safe_spatial_join
from src.config import MIN_TRANSACTIONS


def join_transactions_to_neighborhoods(
    transactions: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially join transaction points to neighborhood polygons.

    Each transaction point gets the 'name' of the neighborhood polygon it
    falls within. Points outside all polygons are kept with name=NaN.

    Args:
        transactions:  Transaction points (EPSG:4326).
        neighborhoods: Neighborhood polygons (EPSG:4326) with 'name' column.

    Returns:
        transactions GDF with 'neighborhood_polygon' column added.
    """
    assert_wgs84(transactions, "transactions")
    assert_wgs84(neighborhoods, "neighborhoods")

    # Only keep the polygon name from the right GDF to avoid column conflicts
    hood_cols = neighborhoods[["name", "geometry"]].rename(
        columns={"name": "neighborhood_polygon"}
    )

    joined = gpd.sjoin(
        transactions,
        hood_cols,
        how="left",
        predicate="within",
    )

    # Drop the sjoin metadata column
    joined = joined.drop(columns=["index_right"], errors="ignore")

    n_unmatched = joined["neighborhood_polygon"].isna().sum()
    if n_unmatched:
        logger.warning(
            f"join_transactions_to_neighborhoods: "
            f"{n_unmatched}/{len(transactions)} transactions "
            f"did not fall within any neighborhood polygon."
        )

    logger.info(
        f"Joined {len(transactions)} transactions → "
        f"{joined['neighborhood_polygon'].nunique()} neighborhoods."
    )
    return joined


def aggregate_transactions_by_neighborhood(
    transactions: gpd.GeoDataFrame,
    neighborhood_col: str = "neighborhood",
) -> pd.DataFrame:
    """
    Compute per-neighborhood transaction statistics.

    Aggregated features:
      - avg_price           : Mean transaction price (₪)
      - median_price        : Median transaction price (₪)
      - avg_price_per_m2    : Mean price per m² (₪/m²)
      - transaction_count   : Number of transactions
      - avg_rooms           : Mean room count
      - avg_area_m2         : Mean apartment size

    Args:
        transactions:     Preprocessed transactions GeoDataFrame.
        neighborhood_col: Column containing neighborhood names.

    Returns:
        DataFrame indexed by neighborhood name with aggregated features.
        Neighborhoods with fewer than MIN_TRANSACTIONS are excluded.
    """
    if neighborhood_col not in transactions.columns:
        raise ValueError(
            f"Column '{neighborhood_col}' not found in transactions GDF. "
            f"Available columns: {list(transactions.columns)}"
        )

    agg = (
        transactions
        .groupby(neighborhood_col)
        .agg(
            avg_price=("transaction_price", "mean"),
            median_price=("transaction_price", "median"),
            avg_price_per_m2=("price_per_m2", "mean"),
            transaction_count=("transaction_price", "count"),
            avg_rooms=("rooms", "mean"),
            avg_area_m2=("area_m2", "mean"),
        )
        .round(0)
        .reset_index()
        .rename(columns={neighborhood_col: "name"})
    )

    # Filter neighborhoods with too few transactions
    before = len(agg)
    agg = agg[agg["transaction_count"] >= MIN_TRANSACTIONS].copy()
    excluded = before - len(agg)
    if excluded:
        logger.warning(
            f"aggregate_transactions_by_neighborhood: excluded {excluded} "
            f"neighborhoods with < {MIN_TRANSACTIONS} transactions."
        )

    logger.info(
        f"Aggregated transactions for {len(agg)} neighborhoods. "
        f"Price range: ₪{agg['avg_price'].min():,.0f} – "
        f"₪{agg['avg_price'].max():,.0f}"
    )
    return agg


def attach_transaction_stats_to_neighborhoods(
    neighborhoods: gpd.GeoDataFrame,
    transactions: gpd.GeoDataFrame,
    neighborhood_col: str = "neighborhood",
) -> gpd.GeoDataFrame:
    """
    Aggregate transaction data and merge it onto the neighborhoods GDF.

    Convenience wrapper combining:
      1. aggregate_transactions_by_neighborhood()
      2. DataFrame merge on neighborhood name

    Args:
        neighborhoods:    Neighborhood polygons GDF.
        transactions:     Transaction points GDF.
        neighborhood_col: Column in transactions for neighborhood name.

    Returns:
        neighborhoods GDF enriched with transaction statistics.
    """
    stats = aggregate_transactions_by_neighborhood(transactions, neighborhood_col)

    enriched = neighborhoods.merge(stats, on="name", how="left")

    n_missing = enriched["avg_price"].isna().sum()
    if n_missing:
        logger.warning(
            f"{n_missing} neighborhoods have no transaction data. "
            f"Their scores will be NaN and excluded from rankings."
        )

    logger.info(
        f"Attached transaction stats to {len(enriched)} neighborhoods."
    )
    return enriched
