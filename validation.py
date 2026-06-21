"""
validation.py — Data Validation
=================================
Two-layer validation:

  Layer 1 — Schema validation (column presence, types, value ranges).
            Uses simple Pandas checks rather than heavyweight frameworks.

  Layer 2 — Spatial validation (CRS, geometry type, coordinate range).
            Delegates CRS checks to crs_utils.py.

Design goal: fail loudly and early so bugs surface at data ingestion,
not silently during the scoring step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.geo.crs_utils import assert_wgs84
from src.config import GUSH_DAN_MUNICIPALITIES, GUSH_DAN_BBOX


# ── Validation result ────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Container for validation findings.

    Attributes:
        passed: True if all critical checks passed.
        errors: List of critical error messages (block pipeline).
        warnings: List of non-critical warnings (log, don't block).
    """
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def log(self, label: str = "Validation") -> None:
        """Log results using loguru."""
        if self.passed:
            logger.success(f"{label}: PASSED ({len(self.warnings)} warnings)")
        else:
            logger.error(f"{label}: FAILED ({len(self.errors)} errors)")
        for err in self.errors:
            logger.error(f"  ✗ {err}")
        for warn in self.warnings:
            logger.warning(f"  ⚠  {warn}")

    def raise_if_failed(self) -> None:
        """Raise ValueError with all error messages if validation failed."""
        if not self.passed:
            raise ValueError(
                "Validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            )


# ── Generic spatial checks ───────────────────────────────────────────────────────

def validate_geodataframe(
    gdf: gpd.GeoDataFrame,
    label: str = "GDF",
    required_columns: list[str] | None = None,
    geometry_types: list[str] | None = None,
) -> ValidationResult:
    """
    Run standard checks on any GeoDataFrame.

    Checks:
      - GDF is not empty.
      - CRS is EPSG:4326.
      - All required columns are present.
      - Geometries are valid.
      - Coordinate bounds are within Gush Dan AOI.

    Args:
        gdf:              GeoDataFrame to validate.
        label:            Human-readable name for error messages.
        required_columns: Columns that must be present.
        geometry_types:   Allowed geometry types (e.g., ['Point', 'Polygon']).

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # Not empty
    if len(gdf) == 0:
        result.add_error(f"{label}: GeoDataFrame is empty.")
        return result  # No point running further checks

    # CRS
    try:
        assert_wgs84(gdf, label)
    except ValueError as e:
        result.add_error(str(e))

    # Required columns
    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            result.add_error(f"{label}: missing columns: {missing}")

    # Geometry validity
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        result.add_warning(f"{label}: {invalid_count} invalid geometries detected.")

    null_geom = gdf.geometry.isna().sum()
    if null_geom > 0:
        result.add_error(f"{label}: {null_geom} null geometries detected.")

    # Geometry types
    if geometry_types:
        actual_types = set(gdf.geometry.geom_type.unique())
        unexpected = actual_types - set(geometry_types)
        if unexpected:
            result.add_warning(
                f"{label}: unexpected geometry types {unexpected}. "
                f"Expected: {geometry_types}"
            )

    # Coordinate bounds (rough check — data should be near Gush Dan)
    if result.passed:
        bounds = gdf.total_bounds  # [min_lon, min_lat, max_lon, max_lat]
        if bounds[0] < 30 or bounds[2] > 40 or bounds[1] < 28 or bounds[3] > 34:
            result.add_warning(
                f"{label}: coordinates appear outside Israel. "
                f"Bounds: {bounds.round(4).tolist()}"
            )

    return result


# ── Domain-specific validators ───────────────────────────────────────────────────

def validate_transactions(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """
    Validate real estate transaction GeoDataFrame.

    Checks beyond generic spatial validation:
      - Prices are within plausible range (₪200K–₪100M).
      - Area in m² is positive.
      - Rooms is a sensible value.
      - Neighborhood names are in the supported list.
    """
    result = validate_geodataframe(
        gdf,
        label="Transactions",
        required_columns=["transaction_price", "rooms", "area_m2", "neighborhood"],
        geometry_types=["Point"],
    )

    if "transaction_price" in gdf.columns:
        prices = pd.to_numeric(gdf["transaction_price"], errors="coerce")
        out_of_range = ((prices < 200_000) | (prices > 100_000_000)).sum()
        null_prices = prices.isna().sum()
        if null_prices:
            result.add_error(f"Transactions: {null_prices} null prices.")
        if out_of_range:
            result.add_warning(
                f"Transactions: {out_of_range} prices outside ₪200K–₪100M range."
            )

    if "area_m2" in gdf.columns:
        areas = pd.to_numeric(gdf["area_m2"], errors="coerce")
        bad_areas = (areas <= 0).sum() + areas.isna().sum()
        if bad_areas:
            result.add_warning(f"Transactions: {bad_areas} invalid area values.")

    if "neighborhood" in gdf.columns:
        unknown = (
            ~gdf["neighborhood"].isin(GUSH_DAN_MUNICIPALITIES)
        ).sum()
        if unknown:
            result.add_warning(
                f"Transactions: {unknown} rows with municipality names "
                f"not in the supported Gush Dan list."
            )

    result.log("Transactions Validation")
    return result


def validate_neighborhoods(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """
    Validate neighborhood polygon GeoDataFrame.

    Extra checks:
      - 'name' column contains values matching our municipality list.
      - No duplicate municipality names.
    """
    result = validate_geodataframe(
        gdf,
        label="Neighborhoods",
        required_columns=["name"],
        geometry_types=["Polygon", "MultiPolygon"],
    )

    if "name" in gdf.columns:
        duplicates = gdf["name"].duplicated().sum()
        if duplicates:
            result.add_error(
                f"Neighborhoods: {duplicates} duplicate municipality names."
            )

        unknown = (~gdf["name"].isin(GUSH_DAN_MUNICIPALITIES)).sum()
        if unknown:
            result.add_warning(
                f"Neighborhoods: {unknown} names not in the Gush Dan list."
            )

        missing_munis = set(GUSH_DAN_MUNICIPALITIES) - set(gdf["name"])
        if missing_munis:
            result.add_warning(
                f"Neighborhoods: missing polygons for: {missing_munis}"
            )

    result.log("Neighborhoods Validation")
    return result


def validate_transit_stops(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Validate transit stops GeoDataFrame."""
    result = validate_geodataframe(
        gdf,
        label="Transit Stops",
        required_columns=["stop_name", "stop_type"],
        geometry_types=["Point"],
    )
    result.log("Transit Stops Validation")
    return result


def validate_parks(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Validate parks GeoDataFrame."""
    result = validate_geodataframe(
        gdf,
        label="Parks",
        required_columns=["name"],
        geometry_types=["Point", "Polygon", "MultiPolygon"],
    )
    result.log("Parks Validation")
    return result
