"""
neighborhood_score.py — Suitability Scoring Engine
====================================================
Computes a composite suitability score (0–100) for each neighborhood
using weighted sub-scores across three dimensions:

  1. Affordability  — based on average transaction price
  2. Transit access — based on stop proximity and density
  3. Park access    — based on park proximity and count

Architecture decision:
  The `NeighborhoodScorer` class is designed with a `fit/transform` API
  so it can be replaced by a scikit-learn-compatible ML model later.
  The rule-based scorer implements the same interface as future ML models.

Scoring pipeline:
  Raw features → Normalise (0–1) → Weight → Sum → Scale to 0–100

Normalisation:
  All sub-scores use min-max scaling so weights are comparable.
  Affordability is inverted (lower price = higher score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.config import DEFAULT_WEIGHTS, SCORE_MIN, SCORE_MAX


# ── Weight configuration ─────────────────────────────────────────────────────────

@dataclass
class ScoringWeights:
    """
    Scoring dimension weights. Must sum to 1.0.

    Attributes:
        affordability: Weight for price-based score (higher = cheaper areas score better).
        transit:       Weight for transit accessibility score.
        parks:         Weight for park accessibility score.
    """
    affordability: float = DEFAULT_WEIGHTS["affordability"]
    transit:       float = DEFAULT_WEIGHTS["transit"]
    parks:         float = DEFAULT_WEIGHTS["parks"]

    def __post_init__(self):
        total = self.affordability + self.transit + self.parks
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0. Got: {total:.4f}. "
                f"(affordability={self.affordability}, transit={self.transit}, "
                f"parks={self.parks})"
            )

    @classmethod
    def from_ui_sliders(
        cls,
        transit_pct: int,
        parks_pct: int,
    ) -> "ScoringWeights":
        """
        Create weights from Streamlit slider values (0–100 each).

        Affordability gets the remaining weight after transit and parks.

        Args:
            transit_pct: Transit importance (0–100).
            parks_pct:   Parks importance (0–100).

        Returns:
            ScoringWeights with normalised weights summing to 1.0.

        Raises:
            ValueError if transit_pct + parks_pct > 100.
        """
        if transit_pct + parks_pct > 100:
            raise ValueError(
                f"transit_pct ({transit_pct}) + parks_pct ({parks_pct}) > 100. "
                f"Reduce one of them."
            )
        t = transit_pct / 100
        p = parks_pct / 100
        a = 1.0 - t - p
        return cls(affordability=round(a, 4), transit=round(t, 4), parks=round(p, 4))


# ── Scoring engine ───────────────────────────────────────────────────────────────

class NeighborhoodScorer:
    """
    Rule-based neighborhood suitability scorer.

    Implements a fit/transform interface so it can be swapped for an
    ML model (e.g., RandomForestRegressor) without changing the pipeline.

    Usage:
        scorer = NeighborhoodScorer(weights)
        scorer.fit(neighborhoods_gdf)          # learns min/max for scaling
        scored = scorer.transform(neighborhoods_gdf)  # returns GDF with scores

    Or shorthand:
        scored = scorer.fit_transform(neighborhoods_gdf)
    """

    # Required input columns (must be present in the GDF)
    REQUIRED_FEATURES = [
        "avg_price",
        "dist_to_nearest_transit_m",
        "dist_to_nearest_park_m",
        "transit_stops_500m",
        "parks_1km",
    ]

    def __init__(self, weights: ScoringWeights | None = None):
        """
        Args:
            weights: ScoringWeights. Defaults to config DEFAULT_WEIGHTS.
        """
        self.weights = weights or ScoringWeights()
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, gdf: gpd.GeoDataFrame) -> "NeighborhoodScorer":
        """
        Learn min/max scaling parameters from the data.

        Must be called before transform().

        Args:
            gdf: Neighborhoods GDF with feature columns.

        Returns:
            self (for chaining).
        """
        self._validate_features(gdf)
        feature_matrix = self._build_feature_matrix(gdf)
        self._scaler.fit(feature_matrix)
        self._fitted = True
        logger.info(
            f"NeighborhoodScorer fitted on {len(gdf)} neighborhoods. "
            f"Weights: affordability={self.weights.affordability}, "
            f"transit={self.weights.transit}, parks={self.weights.parks}"
        )
        return self

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute suitability scores for each neighborhood.

        Args:
            gdf: Neighborhoods GDF with feature columns.

        Returns:
            GDF with additional columns:
              - score_affordability (0–1)
              - score_transit (0–1)
              - score_parks (0–1)
              - neighborhood_score (0–100)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        self._validate_features(gdf)
        result = gdf.copy()

        feature_matrix = self._build_feature_matrix(result)
        scaled = self._scaler.transform(feature_matrix)

        # Column order matches _build_feature_matrix
        # avg_price is inverted: lower price → higher affordability score
        result["score_affordability"] = (1 - scaled[:, 0]).round(4)
        result["score_transit"]        = scaled[:, 1].round(4)
        result["score_parks"]          = scaled[:, 2].round(4)

        # Weighted composite score
        result["neighborhood_score"] = (
            (
                result["score_affordability"] * self.weights.affordability
                + result["score_transit"]       * self.weights.transit
                + result["score_parks"]         * self.weights.parks
            ) * SCORE_MAX
        ).round(1)

        # Rank (1 = best)
        result["rank"] = (
            result["neighborhood_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        logger.info(
            f"Scoring complete. Score range: "
            f"{result['neighborhood_score'].min():.1f} – "
            f"{result['neighborhood_score'].max():.1f}"
        )
        return result

    def fit_transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convenience method: fit then transform."""
        return self.fit(gdf).transform(gdf)

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _build_feature_matrix(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Build the numeric feature matrix for scaling.

        Transit score uses a composite of stop proximity + density.
        Park score uses a composite of park proximity + count.
        """
        # Affordability: avg_price (will be inverted after scaling)
        f_affordability = gdf["avg_price"].fillna(gdf["avg_price"].median())

        # Transit: combine distance (inverse) and stop count
        transit_dist = gdf["dist_to_nearest_transit_m"].fillna(
            gdf["dist_to_nearest_transit_m"].max()
        )
        transit_count = gdf["transit_stops_500m"].fillna(0)
        # Simple composite: invert distance, add normalised count
        f_transit = (1 / (1 + transit_dist)) + transit_count

        # Parks: combine distance (inverse) and park count
        park_dist = gdf["dist_to_nearest_park_m"].fillna(
            gdf["dist_to_nearest_park_m"].max()
        )
        park_count = gdf["parks_1km"].fillna(0)
        f_parks = (1 / (1 + park_dist)) + park_count

        return np.column_stack([f_affordability, f_transit, f_parks])

    def _validate_features(self, gdf: gpd.GeoDataFrame) -> None:
        """Raise ValueError if required feature columns are missing."""
        missing = [c for c in self.REQUIRED_FEATURES if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"NeighborhoodScorer: missing required columns: {missing}. "
                f"Run the full feature engineering pipeline first."
            )


# ── Budget filter ────────────────────────────────────────────────────────────────

def filter_by_budget(
    scored_gdf: gpd.GeoDataFrame,
    max_budget_ils: float,
    budget_col: str = "avg_price",
) -> gpd.GeoDataFrame:
    """
    Filter neighborhoods to those where the average price is within budget.

    Args:
        scored_gdf:     Scored neighborhoods GDF.
        max_budget_ils: Maximum budget in NIS.
        budget_col:     Column to filter on.

    Returns:
        Filtered GDF. If no neighborhoods pass the filter, returns all
        (with a warning) to avoid an empty map.
    """
    filtered = scored_gdf[scored_gdf[budget_col] <= max_budget_ils].copy()

    if len(filtered) == 0:
        logger.warning(
            f"No neighborhoods within budget ₪{max_budget_ils:,.0f}. "
            f"Returning all neighborhoods."
        )
        return scored_gdf

    logger.info(
        f"Budget filter (≤₪{max_budget_ils:,.0f}): "
        f"{len(filtered)}/{len(scored_gdf)} neighborhoods."
    )
    return filtered


# ── Convenience pipeline ─────────────────────────────────────────────────────────

def run_scoring_pipeline(
    neighborhoods: gpd.GeoDataFrame,
    weights: ScoringWeights | None = None,
) -> gpd.GeoDataFrame:
    """
    End-to-end scoring pipeline.

    Expects `neighborhoods` to already have all feature columns attached
    (from spatial_join.py and distance_features.py).

    Args:
        neighborhoods: Enriched neighborhoods GDF.
        weights:       Scoring weights. Defaults to config values.

    Returns:
        Scored neighborhoods GDF with score columns and rank.
    """
    scorer = NeighborhoodScorer(weights)
    return scorer.fit_transform(neighborhoods)
