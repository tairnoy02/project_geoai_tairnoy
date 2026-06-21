"""
crs_utils.py — CRS Enforcement Utilities
=========================================
All GeoDataFrames in this project MUST use EPSG:4326 (WGS84 lat/lon).

Rules:
  - Never store intermediate data in a projected CRS.
  - Always call `to_wgs84()` immediately after loading any geospatial data.
  - Use `assert_wgs84()` at function entry points to catch mismatches early.
  - If you need metric distances, reproject *locally* for the calculation
    then convert back — do NOT persist projected GDFs.

Why EPSG:4326?
  - Universal compatibility with Folium, web tiles, and most APIs.
  - Avoids silent coordinate mismatches across data sources.
  - GeoParquet files round-trip cleanly in WGS84.
"""

import geopandas as gpd
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:4326"

# ITM — Israel Transverse Mercator, used in Israeli government datasets
ITM_CRS = "EPSG:2039"

# Pseudo-Mercator — common in web map sources
WEB_MERCATOR_CRS = "EPSG:3857"


# ── Core helpers ───────────────────────────────────────────────────────────────

def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reproject any GeoDataFrame to EPSG:4326.

    Safe to call even if the GDF is already in WGS84.
    Always call this immediately after loading raw data.

    Args:
        gdf: Input GeoDataFrame (any CRS).

    Returns:
        GeoDataFrame in EPSG:4326.

    Raises:
        ValueError: If the GDF has no CRS set at all.
    """
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS. Set it explicitly before calling to_wgs84(). "
            "Example: gdf = gdf.set_crs('EPSG:2039')"
        )

    if gdf.crs.to_epsg() == 4326:
        logger.debug("GDF already in EPSG:4326 — no reprojection needed.")
        return gdf

    original_crs = gdf.crs.to_string()
    gdf = gdf.to_crs(TARGET_CRS)
    logger.info(f"Reprojected GDF from {original_crs} → EPSG:4326")
    return gdf


def assert_wgs84(gdf: gpd.GeoDataFrame, label: str = "GeoDataFrame") -> None:
    """
    Assert that a GeoDataFrame is in EPSG:4326.

    Use this as a guard at function entry points.

    Args:
        gdf:   GeoDataFrame to check.
        label: Human-readable name used in the error message.

    Raises:
        ValueError: If the CRS is not EPSG:4326.
    """
    if gdf.crs is None:
        raise ValueError(f"{label} has no CRS. Expected EPSG:4326.")

    if gdf.crs.to_epsg() != 4326:
        raise ValueError(
            f"{label} is in {gdf.crs.to_string()}, expected EPSG:4326. "
            f"Call to_wgs84() before passing this GDF."
        )


def safe_spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    how: str = "left",
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """
    Spatial join with automatic CRS validation.

    Ensures both GDFs are in EPSG:4326 before joining.
    This prevents silent wrong-result bugs from mixed CRS systems.

    Args:
        left:      Left GeoDataFrame.
        right:     Right GeoDataFrame.
        how:       Join type: 'left', 'right', 'inner'.
        predicate: Spatial predicate: 'intersects', 'within', 'contains'.

    Returns:
        Joined GeoDataFrame in EPSG:4326.
    """
    assert_wgs84(left, "left GDF")
    assert_wgs84(right, "right GDF")

    result = gpd.sjoin(left, right, how=how, predicate=predicate)
    logger.debug(
        f"Spatial join complete: {len(left)} × {len(right)} → {len(result)} rows"
    )
    return result


def reproject_for_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Temporarily reproject to ITM (EPSG:2039) for metric distance calculations.

    IMPORTANT: The result is in meters, NOT in degrees.
    Always convert back to WGS84 after your calculation.

    Example:
        gdf_m = reproject_for_metric(gdf)
        gdf_m["dist_m"] = gdf_m.geometry.distance(point_in_itm)
        gdf = to_wgs84(gdf_m)  # back to WGS84

    Args:
        gdf: Input GeoDataFrame in EPSG:4326.

    Returns:
        GeoDataFrame in EPSG:2039 (metric).
    """
    assert_wgs84(gdf, "input GDF")
    return gdf.to_crs(ITM_CRS)


def crs_summary(gdf: gpd.GeoDataFrame, label: str = "") -> str:
    """
    Return a human-readable CRS summary string for logging/debugging.

    Args:
        gdf:   GeoDataFrame to inspect.
        label: Optional label prefix.

    Returns:
        Summary string.
    """
    prefix = f"[{label}] " if label else ""
    if gdf.crs is None:
        return f"{prefix}CRS: None (⚠️  missing!)"
    return (
        f"{prefix}CRS: {gdf.crs.to_string()} | "
        f"EPSG: {gdf.crs.to_epsg()} | "
        f"Rows: {len(gdf)} | "
        f"Bounds: {gdf.total_bounds.round(4).tolist()}"
    )
