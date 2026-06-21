"""
streamlit_app.py — Where Should I Buy? 🏠
==========================================
Main Streamlit entry point.

Run locally:
    streamlit run streamlit_app.py

The app:
  1. Loads data (real or mock) via src/data/loaders.py
  2. Preprocesses data via src/data/preprocessing.py
  3. Computes spatial features via src/geo/distance_features.py
  4. Joins transactions to neighborhoods via src/geo/spatial_join.py
  5. Scores neighborhoods via src/scoring/neighborhood_score.py
  6. Renders an interactive Folium map + ranking table
"""

import sys
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
from loguru import logger

from src.data.loaders import (
    load_transactions,
    load_neighborhoods,
    load_transit_stops,
    load_parks,
)
from src.data.preprocessing import (
    preprocess_transactions,
    preprocess_neighborhoods,
    preprocess_transit_stops,
    preprocess_parks,
)
from src.geo.distance_features import compute_neighborhood_spatial_features
from src.geo.spatial_join import attach_transaction_stats_to_neighborhoods
from src.scoring.neighborhood_score import (
    ScoringWeights,
    run_scoring_pipeline,
    filter_by_budget,
)
from src.scoring.ml_scorer import MLScorer, evaluate_models
from src.app.map_utils import build_neighborhood_map
from src.config import (
    UI_DEFAULT_BUDGET_ILS,
    UI_DEFAULT_MAX_COMMUTE_MIN,
    UI_DEFAULT_TRANSIT_WEIGHT,
    UI_DEFAULT_PARKS_WEIGHT,
    GUSH_DAN_MUNICIPALITIES,
)

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Where Should I Buy? 🏠",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data pipeline (cached) ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="📦 Loading and processing data...")
def load_and_process_all():
    """
    Run the full data pipeline and return processed GeoDataFrames.

    Cached by Streamlit — only runs once per session.
    Re-runs if source files change (via cache invalidation).

    Returns:
        Tuple of (transactions, neighborhoods, transit_stops, parks)
        all preprocessed and in EPSG:4326.
    """
    transactions  = preprocess_transactions(load_transactions())
    neighborhoods = preprocess_neighborhoods(load_neighborhoods())
    transit_stops = preprocess_transit_stops(load_transit_stops())
    parks         = preprocess_parks(load_parks())
    return transactions, neighborhoods, transit_stops, parks


@st.cache_data(show_spinner="🗺️ Computing spatial features...")
def build_scored_neighborhoods(
    _neighborhoods: gpd.GeoDataFrame,  # underscore prefix skips hashing
    _transactions:  gpd.GeoDataFrame,
    _transit_stops: gpd.GeoDataFrame,
    _parks:         gpd.GeoDataFrame,
    transit_weight: int,
    parks_weight:   int,
) -> gpd.GeoDataFrame:
    """
    Feature engineering + scoring pipeline.

    Cached separately from data loading so weight changes don't trigger
    a full reload — only re-score.

    Args:
        _neighborhoods, _transactions, _transit_stops, _parks:
            Preprocessed GeoDataFrames (underscore to skip st.cache_data hashing).
        transit_weight: Transit importance slider value (0–100).
        parks_weight:   Parks importance slider value (0–100).

    Returns:
        Scored neighborhoods GeoDataFrame.
    """
    # Spatial features
    enriched = compute_neighborhood_spatial_features(
        _neighborhoods.copy(), _transit_stops, _parks
    )

    # Transaction statistics
    enriched = attach_transaction_stats_to_neighborhoods(
        enriched, _transactions, neighborhood_col="neighborhood"
    )

    # Scoring
    weights = ScoringWeights.from_ui_sliders(transit_weight, parks_weight)
    scored = run_scoring_pipeline(enriched, weights)

    return scored


# ── Sidebar ──────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the sidebar controls and return the user's selections as a dict.

    Returns:
        {
            budget:         int   — Max budget in NIS
            transit_weight: int   — Transit importance 0–100
            parks_weight:   int   — Parks importance 0–100
            show_transit:   bool  — Show transit layer on map
            show_parks:     bool  — Show parks layer on map
        }
    """
    with st.sidebar:
        st.title("🏠 Where Should I Buy?")
        st.caption("GeoAI Course Project · Gush Dan, Israel")
        st.divider()

        st.subheader("💰 Budget")
        budget = st.slider(
            "Maximum budget (₪)",
            min_value=500_000,
            max_value=10_000_000,
            value=UI_DEFAULT_BUDGET_ILS,
            step=100_000,
            format="₪%d",
            help="Filter neighborhoods by average apartment price.",
        )

        st.subheader("⚖️ What matters most to you?")
        st.caption(
            "Transit + Parks weights must total ≤ 100. "
            "Affordability gets the remainder."
        )

        transit_weight = st.slider(
            "🚌 Public transit importance",
            min_value=0,
            max_value=100,
            value=UI_DEFAULT_TRANSIT_WEIGHT,
            step=5,
            help="Higher = prefer areas with more transit options.",
        )

        max_parks = max(0, 100 - transit_weight)
        parks_weight = st.slider(
            "🌳 Parks & green space importance",
            min_value=0,
            max_value=max_parks,
            value=min(UI_DEFAULT_PARKS_WEIGHT, max_parks),
            step=5,
            help="Higher = prefer areas with more parks nearby.",
        )

        affordability_weight = 100 - transit_weight - parks_weight
        st.info(
            f"💰 Affordability weight: **{affordability_weight}%**  \n"
            f"🚌 Transit weight: **{transit_weight}%**  \n"
            f"🌳 Parks weight: **{parks_weight}%**"
        )

        st.subheader("🗺️ Map Layers")
        show_transit = st.checkbox("Show transit stops", value=False)
        show_parks   = st.checkbox("Show parks",         value=False)

        st.subheader("🤖 ML Insights")
        show_ml = st.checkbox(
            "Show ML model results",
            value=True,
            help="Train a Random Forest model to predict city scores and show prescriptive recommendations.",
        )

        st.divider()
        st.caption(
            "📌 Data: Mock data (MVP)\n\n"
            "Area of Interest: Gush Dan metropolitan area\n\n"
            "Built with GeoPandas · Folium · Streamlit"
        )

    return {
        "budget":         budget,
        "transit_weight": transit_weight,
        "parks_weight":   parks_weight,
        "show_transit":   show_transit,
        "show_parks":     show_parks,
        "show_ml":        show_ml,
    }


# ── ML pipeline (cached) ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="🤖 Training ML scoring models...")
def run_ml_pipeline(_scored: gpd.GeoDataFrame):
    """
    Train and evaluate ML models on the scored cities GDF.

    Returns plain DataFrames (cache-safe — no GeoDataFrame serialization).
    """
    logger.info("Running ML pipeline...")
    model_results = evaluate_models(_scored)
    scorer = MLScorer()
    scored_ml = scorer.fit_transform(_scored)
    recommendations = scorer.prescriptive_recommendations(scored_ml)
    importances_dict = (
        scorer.feature_importances.to_dict()
        if scorer.feature_importances is not None else {}
    )
    scores_df = scored_ml[["name", "neighborhood_score", "ml_score", "ml_rank"]].copy()
    return model_results, scores_df, recommendations, importances_dict


# ── Main page ────────────────────────────────────────────────────────────────────

def render_main(
    scored: gpd.GeoDataFrame,
    transit_stops: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    ui: dict,
) -> None:
    """Render the main content area: map + ranking table."""

    st.title("📍 City Suitability Map")
    st.caption(
        "Explore the best cities to buy in Gush Dan. "
        "Click any city polygon for details. "
        "Adjust weights in the sidebar to personalise your search."
    )

    # Filter by budget
    filtered = filter_by_budget(scored, ui["budget"])

    # ── Map ──────────────────────────────────────────────────────────────────────
    fmap = build_neighborhood_map(
        neighborhoods=filtered,
        transit_stops=transit_stops if ui["show_transit"] else None,
        parks=parks if ui["show_parks"] else None,
        show_transit=ui["show_transit"],
        show_parks=ui["show_parks"],
    )

    # Merge ML scores into filtered so they appear in the ranking table
    if ui.get("show_ml"):
        _, scores_df, _, _ = run_ml_pipeline(scored)
        filtered = filtered.merge(
            scores_df[["name", "ml_score", "ml_rank"]], on="name", how="left"
        )

    col_map, col_table = st.columns([3, 2])

    with col_map:
        st.subheader("🗺️ Interactive Map")
        st_folium(fmap, width=700, height=520, returned_objects=[])

    with col_table:
        st.subheader("🏆 City Rankings")
        _render_ranking_table(filtered)

    # ── Summary metrics ──────────────────────────────────────────────────────────
    st.divider()
    _render_summary_metrics(filtered, ui)

    # ── Score breakdown chart ─────────────────────────────────────────────────────
    if "score_affordability" in filtered.columns:
        _render_score_breakdown(filtered)

    # ── ML section ────────────────────────────────────────────────────────────────
    if ui.get("show_ml"):
        _render_ml_section(scored)


def _render_ranking_table(scored: gpd.GeoDataFrame) -> None:
    """Display a styled ranking table."""
    display_cols = {
        "rank":                    "Rank",
        "name":                    "City",
        "neighborhood_score":      "Score",
        "ml_score":                "ML Score",
        "avg_price":               "Avg Price (₪)",
        "transit_stops_500m":      "Transit (500m)",
        "parks_1km":               "Parks (1km)",
    }

    available = [c for c in display_cols if c in scored.columns]
    df = scored[available].copy()
    df = df.sort_values("rank", ascending=True)
    df = df.rename(columns={c: display_cols[c] for c in available})

    # Format price
    if "Avg Price (₪)" in df.columns:
        df["Avg Price (₪)"] = df["Avg Price (₪)"].apply(
            lambda x: f"₪{x:,.0f}" if pd.notna(x) else "N/A"
        )

    # Format score
    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(lambda x: f"{x:.1f}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=480,
    )


def _render_summary_metrics(scored: gpd.GeoDataFrame, ui: dict) -> None:
    """Show KPI summary metrics."""
    col1, col2, col3, col4 = st.columns(4)

    best = scored.sort_values("neighborhood_score", ascending=False).iloc[0]

    with col1:
        st.metric("🏘️ Cities Found", len(scored))
    with col2:
        st.metric(
            "🥇 Top City",
            best.get("name", "—"),
            help=f"Score: {best.get('neighborhood_score', 0):.1f}/100",
        )
    with col3:
        if "avg_price" in scored.columns:
            min_price = scored["avg_price"].min()
            st.metric("💰 Min Avg Price", f"₪{min_price:,.0f}")
    with col4:
        if "neighborhood_score" in scored.columns:
            avg_score = scored["neighborhood_score"].mean()
            st.metric("⭐ Avg Score", f"{avg_score:.1f}/100")


def _render_score_breakdown(scored: gpd.GeoDataFrame) -> None:
    """Render a bar chart of score components per neighborhood."""
    import plotly.graph_objects as go

    st.subheader("📊 Score Breakdown by City")

    df = scored.sort_values("rank").head(12)  # Top 12 for readability

    names = df["name"].tolist()

    fig = go.Figure()
    fig.add_bar(
        x=names,
        y=(df["score_affordability"] * 100).round(1),
        name="💰 Affordability",
        marker_color="#4c72b0",
    )
    fig.add_bar(
        x=names,
        y=(df["score_transit"] * 100).round(1),
        name="🚌 Transit",
        marker_color="#dd8452",
    )
    fig.add_bar(
        x=names,
        y=(df["score_parks"] * 100).round(1),
        name="🌳 Parks",
        marker_color="#55a868",
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="City",
        yaxis_title="Sub-score (0–100)",
        height=350,
        margin=dict(l=20, r=20, t=20, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ml_section(scored: gpd.GeoDataFrame) -> None:
    """Render ML model comparison, feature importance, and prescriptive recommendations."""
    import plotly.graph_objects as go

    st.divider()
    st.subheader("🤖 ML Model — Predictive & Prescriptive Layer")
    st.caption(
        "A Random Forest regressor learns to predict city suitability scores "
        "from geospatial features. Evaluation uses Leave-One-Out CV (LOO) — "
        "the correct strategy for n=12 cities."
    )

    model_results, scores_df, recommendations, importances_dict = run_ml_pipeline(scored)
    importances = (
        pd.Series(importances_dict).sort_values(ascending=False)
        if importances_dict else None
    )

    col_table, col_chart = st.columns(2)

    with col_table:
        st.markdown("**Model comparison — LOO-CV MAE (lower is better)**")
        st.dataframe(model_results, hide_index=True, use_container_width=True)
        best_mae = model_results.iloc[0]["MAE (LOO-CV)"]
        baseline_mae = model_results[model_results["Model"] == "Baseline (mean)"]["MAE (LOO-CV)"].values[0]
        improvement = baseline_mae - best_mae
        st.success(
            f"Best model improves over baseline by **{improvement:.2f} score points** MAE"
        )

    with col_chart:
        if importances is not None and len(importances) > 0:
            st.markdown("**Feature importance (Random Forest)**")
            top = importances.head(8)
            fig = go.Figure(go.Bar(
                x=top.values,
                y=top.index,
                orientation="h",
                marker_color="#4c72b0",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Importance",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ML score vs rule-based score comparison
    st.markdown("**Rule-based score vs ML predicted score (per city)**")
    compare_df = scores_df.rename(columns={
        "name": "City",
        "neighborhood_score": "Rule-based Score",
        "ml_score": "ML Score",
        "ml_rank": "ML Rank",
    }).sort_values("ML Score", ascending=False).reset_index(drop=True)
    compare_df["Δ Score"] = (compare_df["ML Score"] - compare_df["Rule-based Score"]).round(1)
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

    # Prescriptive recommendations
    if not recommendations.empty:
        st.divider()
        st.markdown("**📋 Prescriptive recommendations — low-scoring cities**")
        st.caption(
            "Cities below the median ML score and the main improvement action "
            "based on their weakest geospatial dimension."
        )
        st.dataframe(recommendations, hide_index=True, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    ui = render_sidebar()

    # Data pipeline
    with st.spinner("Loading data and computing features..."):
        transactions, neighborhoods, transit_stops, parks = load_and_process_all()

    # Scoring (cached by weight values)
    scored = build_scored_neighborhoods(
        neighborhoods,
        transactions,
        transit_stops,
        parks,
        transit_weight=ui["transit_weight"],
        parks_weight=ui["parks_weight"],
    )

    # Main content
    render_main(scored, transit_stops, parks, ui)


if __name__ == "__main__":
    main()
