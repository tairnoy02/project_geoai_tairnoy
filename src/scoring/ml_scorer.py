"""
ml_scorer.py — Machine Learning Scoring Model
==============================================
Predicts city suitability score (0–100) using supervised regression.

Target variable : neighborhood_score  (rule-based composite, 0–100)
Problem type    : Regression
KPI             : MAE  (mean absolute error in score points, lower = better)
CV strategy     : Leave-One-Out — correct for n=12 cities (mock data);
                  switch to 80/20 temporal split when real data arrives.

Pipeline
  Raw geospatial features
  → Feature engineering (composites + centroid coordinates)
  → LOO-CV model comparison
  → Best model fit on all cities
  → Prescriptive recommendations for low-scoring cities

Stage reached: Predictive → Prescriptive
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.geo.crs_utils import assert_wgs84

# Raw feature columns produced by the full scoring pipeline
FEATURE_COLS = [
    "avg_price",
    "avg_price_per_m2",
    "avg_rooms",
    "avg_area_m2",
    "transaction_count",
    "dist_to_nearest_transit_m",
    "dist_to_nearest_park_m",
    "transit_stops_500m",
    "transit_stops_1km",
    "parks_1km",
]

TARGET_COL = "neighborhood_score"


def build_feature_matrix(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build the ML feature matrix from a scored GeoDataFrame.

    Uses only columns that exist in the GDF (handles partial pipelines).
    Adds derived composites and spatial centroid coordinates.

    Args:
        gdf: Cities GeoDataFrame with feature columns (EPSG:4326).

    Returns:
        DataFrame with numeric features; NaNs filled with column medians.
    """
    assert_wgs84(gdf, "build_feature_matrix")

    available = [c for c in FEATURE_COLS if c in gdf.columns]
    df = gdf[available].copy().astype(float)

    # Derived composite features — same logic as the rule-based scorer
    if "dist_to_nearest_transit_m" in df.columns and "transit_stops_500m" in df.columns:
        df["transit_composite"] = (
            1.0 / (1.0 + df["dist_to_nearest_transit_m"]) + df["transit_stops_500m"]
        )

    if "dist_to_nearest_park_m" in df.columns and "parks_1km" in df.columns:
        df["park_composite"] = (
            1.0 / (1.0 + df["dist_to_nearest_park_m"]) + df["parks_1km"]
        )

    # Spatial position features: centroid coordinates in ITM (metres)
    centroids = gdf.to_crs("EPSG:2039").geometry.centroid
    df["centroid_x"] = centroids.x.values
    df["centroid_y"] = centroids.y.values

    return df.fillna(df.median())


def evaluate_models(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compare regression models using Leave-One-Out cross-validation.

    With n=12 cities, LOO-CV is the correct evaluation strategy:
    each city is held out once as the test set.

    Args:
        gdf: Scored cities GDF with TARGET_COL present.

    Returns:
        DataFrame sorted by MAE with columns:
        [Model, MAE (LOO-CV), MAE Std, R²]
    """
    assert_wgs84(gdf, "evaluate_models")

    if TARGET_COL not in gdf.columns:
        raise ValueError(
            f"evaluate_models: '{TARGET_COL}' column not found. "
            "Run run_scoring_pipeline() before evaluate_models()."
        )

    X = build_feature_matrix(gdf)
    y = gdf[TARGET_COL].values

    loo = LeaveOneOut()

    candidates = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=42
        ),
    }

    rows = []
    for name, model in candidates.items():
        mae_scores = -cross_val_score(
            model, X, y, cv=loo, scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(model, X, y, cv=loo, scoring="r2")
        rows.append({
            "Model": name,
            "MAE (LOO-CV)": round(float(mae_scores.mean()), 2),
            "MAE Std": round(float(mae_scores.std()), 2),
            "R²": round(float(r2_scores.mean()), 3),
        })
        logger.info(
            f"  {name}: MAE={mae_scores.mean():.2f} ± {mae_scores.std():.2f}, "
            f"R²={r2_scores.mean():.3f}"
        )

    return pd.DataFrame(rows).sort_values("MAE (LOO-CV)").reset_index(drop=True)


class MLScorer:
    """
    Supervised regression scorer for city suitability.

    Implements the same fit / transform interface as NeighborhoodScorer
    (src/scoring/neighborhood_score.py) so it can be swapped in the pipeline.

    Adds columns to the GDF:
      - ml_score : predicted suitability (0–100)
      - ml_rank  : rank by ml_score (1 = best)
    """

    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        self._fitted = False
        self._feature_importances: Optional[pd.Series] = None

    def fit(self, gdf: gpd.GeoDataFrame) -> "MLScorer":
        """
        Train on all available cities.

        Args:
            gdf: Scored cities GDF with TARGET_COL and feature columns.
        """
        assert_wgs84(gdf, "MLScorer.fit")
        if TARGET_COL not in gdf.columns:
            raise ValueError(
                f"MLScorer.fit: '{TARGET_COL}' not found. "
                "Run run_scoring_pipeline() first."
            )

        X = build_feature_matrix(gdf)
        y = gdf[TARGET_COL].values
        self.model.fit(X, y)
        self._fitted = True

        # Extract feature importances if the model supports them
        inner = getattr(self.model, "named_steps", {}).get("model", self.model)
        if hasattr(inner, "feature_importances_"):
            self._feature_importances = pd.Series(
                inner.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            logger.debug(
                "Top-3 features: "
                + str(self._feature_importances.head(3).to_dict())
            )

        logger.info(f"MLScorer fitted on {len(gdf)} cities.")
        return self

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Predict suitability scores for each city.

        Args:
            gdf: Cities GDF with feature columns (EPSG:4326).

        Returns:
            GDF with ml_score and ml_rank columns added.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        assert_wgs84(gdf, "MLScorer.transform")

        X = build_feature_matrix(gdf)
        result = gdf.copy()
        result["ml_score"] = np.clip(self.model.predict(X), 0, 100).round(1)
        result["ml_rank"] = (
            result["ml_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        logger.info(
            f"ML predictions: "
            f"{result['ml_score'].min():.1f}–{result['ml_score'].max():.1f}"
        )
        return result

    def fit_transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fit then transform."""
        return self.fit(gdf).transform(gdf)

    @property
    def feature_importances(self) -> Optional[pd.Series]:
        """Feature importance Series (Random Forest / GBR only)."""
        return self._feature_importances

    def prescriptive_recommendations(
        self, gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Generate one actionable recommendation per low-scoring city.

        Cities below the median ml_score receive a suggestion that targets
        their weakest geospatial dimension.

        Args:
            gdf: GDF with ml_score column (output of fit_transform).

        Returns:
            DataFrame with columns [City, ML Score, Recommendation].
        """
        assert_wgs84(gdf, "prescriptive_recommendations")
        if "ml_score" not in gdf.columns:
            raise RuntimeError("Run fit_transform() before calling this method.")

        threshold = gdf["ml_score"].median()
        low = gdf[gdf["ml_score"] < threshold].copy()

        q75_price = gdf["avg_price"].quantile(0.75) if "avg_price" in gdf.columns else np.inf
        q25_transit = gdf["transit_stops_500m"].quantile(0.25) if "transit_stops_500m" in gdf.columns else 0

        rows = []
        for _, row in low.iterrows():
            dist_transit = row.get("dist_to_nearest_transit_m", 0)
            dist_park = row.get("dist_to_nearest_park_m", 0)
            price = row.get("avg_price", 0)
            transit_density = row.get("transit_stops_500m", 0)

            if dist_transit > 1_000:
                action = "Improve transit access — nearest stop >1 km away"
            elif dist_park > 2_000:
                action = "Add green spaces — nearest park >2 km away"
            elif price > q75_price:
                action = "High prices limit affordability score"
            elif transit_density < q25_transit:
                action = "Low transit density within 500 m"
            else:
                action = "Multiple dimensions below average"

            rows.append({
                "City": row.get("name", "—"),
                "ML Score": row["ml_score"],
                "Recommendation": action,
            })

        return pd.DataFrame(rows).sort_values("ML Score").reset_index(drop=True)
