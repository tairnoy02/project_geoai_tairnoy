"""
map_utils.py — Folium Map Building Utilities
=============================================
Creates interactive Folium maps for the Streamlit application.

Map layers:
  1. Neighborhood polygons — choropleth colored by score
  2. Transit stops         — clustered markers
  3. Parks                 — green circle markers
  4. Transactions          — optional point layer

Design principles:
  - Every function returns a standalone folium.Map object.
  - Keep map logic here, keep UI logic in streamlit_app.py.
  - Maps should be readable without interaction (good default zoom/center).
"""

from __future__ import annotations

from typing import Optional

import folium
from folium.plugins import MarkerCluster, MiniMap
import geopandas as gpd
import pandas as pd
import numpy as np
from loguru import logger

from src.geo.crs_utils import assert_wgs84
from src.config import GUSH_DAN_CENTER, SCORE_MIN, SCORE_MAX


# ── Color utilities ──────────────────────────────────────────────────────────────

def score_to_color(score: float, max_score: float = 100.0) -> str:
    """
    Map a numeric score (0–100) to a hex color on a red-yellow-green gradient.

    Args:
        score:     Score value.
        max_score: Maximum expected score (default 100).

    Returns:
        Hex color string e.g. '#4daf4a'.
    """
    ratio = min(max(score / max_score, 0), 1)

    if ratio < 0.5:
        # Red → Yellow
        r = 220
        g = int(ratio * 2 * 200)
        b = 0
    else:
        # Yellow → Green
        r = int((1 - (ratio - 0.5) * 2) * 220)
        g = 180
        b = 0

    return f"#{r:02x}{g:02x}{b:02x}"


# ── Main map builders ────────────────────────────────────────────────────────────

def build_neighborhood_map(
    neighborhoods: gpd.GeoDataFrame,
    transit_stops: Optional[gpd.GeoDataFrame] = None,
    parks: Optional[gpd.GeoDataFrame] = None,
    show_transit: bool = True,
    show_parks: bool = True,
    zoom_start: int = 12,
) -> folium.Map:
    """
    Build the main interactive map with neighborhood polygons and overlays.

    Args:
        neighborhoods:  Scored neighborhood polygons GDF. Must have
                        'neighborhood_score', 'name', 'rank' columns.
        transit_stops:  Optional transit stops GDF (Point).
        parks:          Optional parks GDF (Point).
        show_transit:   Whether to add transit layer.
        show_parks:     Whether to add parks layer.
        zoom_start:     Initial zoom level.

    Returns:
        folium.Map object ready for st_folium().
    """
    assert_wgs84(neighborhoods, "neighborhoods")

    m = folium.Map(
        location=[GUSH_DAN_CENTER["lat"], GUSH_DAN_CENTER["lon"]],
        zoom_start=zoom_start,
        tiles="CartoDB positron",  # Clean light basemap
    )

    # Add neighborhood polygons
    _add_neighborhood_layer(m, neighborhoods)

    # Optional overlays
    if show_transit and transit_stops is not None and len(transit_stops) > 0:
        assert_wgs84(transit_stops, "transit_stops")
        _add_transit_layer(m, transit_stops)

    if show_parks and parks is not None and len(parks) > 0:
        assert_wgs84(parks, "parks")
        _add_parks_layer(m, parks)

    # UI elements
    _add_legend(m)
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    logger.info(
        f"Built map with {len(neighborhoods)} neighborhoods, "
        f"transit={'yes' if show_transit else 'no'}, "
        f"parks={'yes' if show_parks else 'no'}"
    )
    return m


def _add_neighborhood_layer(
    m: folium.Map,
    neighborhoods: gpd.GeoDataFrame,
) -> None:
    """Add neighborhood polygon layer with score-based fill colors."""

    score_col = "neighborhood_score"
    has_scores = score_col in neighborhoods.columns

    group = folium.FeatureGroup(name="🏙️ Neighborhoods", show=True)

    for _, row in neighborhoods.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        score = row.get(score_col, 50) if has_scores else 50
        rank  = row.get("rank", "—")
        name  = row.get("name", "Unknown")

        color = score_to_color(score)

        # Build popup content
        avg_price = row.get("avg_price", None)
        transit_count = row.get("transit_stops_500m", None)
        park_count = row.get("parks_1km", None)

        price_str = f"₪{avg_price:,.0f}" if pd.notna(avg_price) else "N/A"
        transit_str = f"{int(transit_count)}" if pd.notna(transit_count) else "N/A"
        parks_str = f"{int(park_count)}" if pd.notna(park_count) else "N/A"

        popup_html = f"""
        <div style='font-family: sans-serif; min-width: 180px;'>
          <h4 style='margin:0; color:#333;'>{name}</h4>
          <hr style='margin:6px 0;'>
          <table style='width:100%; font-size:13px;'>
            <tr><td>🏆 Rank</td><td><b>#{rank}</b></td></tr>
            <tr><td>⭐ Score</td><td><b>{score:.1f}/100</b></td></tr>
            <tr><td>💰 Avg Price</td><td>{price_str}</td></tr>
            <tr><td>🚌 Stops (500m)</td><td>{transit_str}</td></tr>
            <tr><td>🌳 Parks (1km)</td><td>{parks_str}</td></tr>
          </table>
        </div>
        """

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "#555",
                "weight": 1.5,
                "fillOpacity": 0.65,
            },
            highlight_function=lambda _: {
                "fillOpacity": 0.85,
                "weight": 3,
                "color": "#222",
            },
            tooltip=folium.Tooltip(f"<b>{name}</b> — Score: {score:.1f}"),
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(group)

    group.add_to(m)


def _add_transit_layer(
    m: folium.Map,
    transit_stops: gpd.GeoDataFrame,
) -> None:
    """Add clustered transit stop markers."""

    group = folium.FeatureGroup(name="🚌 Transit Stops", show=False)
    cluster = MarkerCluster(name="Transit cluster").add_to(group)

    # Icons by stop type
    type_icons = {
        "bus":         ("blue", "bus"),
        "rail":        ("darkred", "train"),
        "light_rail":  ("purple", "train"),
        "unknown":     ("gray", "info-sign"),
    }

    for _, row in transit_stops.iterrows():
        if row.geometry is None:
            continue

        stop_type = row.get("stop_type", "unknown")
        stop_name = row.get("stop_name", "Stop")
        color, icon = type_icons.get(stop_type, ("gray", "info-sign"))

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=f"{stop_type.title()}: {stop_name}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(cluster)

    group.add_to(m)


def _add_parks_layer(
    m: folium.Map,
    parks: gpd.GeoDataFrame,
) -> None:
    """Add park location markers as green circles."""

    group = folium.FeatureGroup(name="🌳 Parks", show=False)

    for _, row in parks.iterrows():
        if row.geometry is None:
            continue
        name = row.get("name", "Park")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="#2d7a2d",
            fill=True,
            fill_color="#4caf50",
            fill_opacity=0.7,
            tooltip=f"🌳 {name}",
        ).add_to(group)

    group.add_to(m)


def _add_legend(m: folium.Map) -> None:
    """Add a score color legend to the map."""
    legend_html = """
    <div style='
        position: fixed;
        bottom: 30px; left: 30px;
        z-index: 1000;
        background: white;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: sans-serif;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    '>
        <b>Suitability Score</b><br>
        <div style='display:flex; gap:4px; margin-top:6px;'>
            <div style='width:20px; height:16px; background:#dc0000; border-radius:2px;'></div>
            <span>Low</span>
        </div>
        <div style='display:flex; gap:4px; margin-top:2px;'>
            <div style='width:20px; height:16px; background:#dcc800; border-radius:2px;'></div>
            <span>Medium</span>
        </div>
        <div style='display:flex; gap:4px; margin-top:2px;'>
            <div style='width:20px; height:16px; background:#00b400; border-radius:2px;'></div>
            <span>High</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
