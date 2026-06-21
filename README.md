# 🏠 Where Should I Buy?
### GeoAI-Powered City Suitability Analysis — Gush Dan Metropolitan Area

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projectgeoaitairnoy.streamlit.app/)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14-green)](https://geopandas.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem Description

Apartment buyers in the Gush Dan metropolitan area (Tel Aviv and surroundings) face a multi-criteria decision: balancing price, proximity to public transport, and access to green spaces across 12 municipalities — each with different characteristics, prices, and urban density.

**Where Should I Buy?** is a GeoAI decision-support system that:
1. **Aggregates** geospatial layers (real estate transactions, transit stops, parks, city boundaries) into a unified dataset
2. **Scores** each city on a composite suitability index (0–100) — rule-based and ML-predicted scores shown side by side in the ranking table
3. **Predicts** suitability scores using a supervised ML model (Random Forest Regression) with LOO-CV evaluation and feature importance
4. **Recommends** targeted improvements for low-scoring cities (prescriptive stage)

---

## 🌍 Geospatial Aspect

This is a **GeoAI project** — the geospatial dimension is central to every stage:

| Geospatial Operation | How it's used |
|---|---|
| **GIS layers** | City boundary polygons (OSM), transaction points, transit stops, park polygons |
| **CRS management** | All layers in EPSG:4326 (WGS84); metric calculations in EPSG:2039 (ITM Israel) |
| **Spatial join** | Transaction points joined to city polygons via `gpd.sjoin` (point-in-polygon) |
| **Distance calculation** | Nearest-neighbour distances (city centroid → closest transit stop / park) using R-tree index (`sjoin_nearest`) |
| **Density analysis** | Count of transit stops within 500 m and parks within 1 km per city centroid (buffered spatial join) |
| **Choropleth map** | Interactive Folium map coloring city polygons by suitability score |
| **Spatial features in ML** | City centroid coordinates (ITM x/y) as spatial features for the regression model |
| **Prescriptive analysis** | Spatial dimension bottleneck detection (transit distance > 1 km, park distance > 2 km) |

---

## 📊 Data Sources

| Layer | Source | Format | Records | Status |
|---|---|---|---|---|
| Real estate transactions | Israeli Land Authority — data.gov.il NADLAN API | CSV / GeoJSON | 500 transactions (mock) · real: ~10,000/quarter | 🔄 Mock data |
| City boundaries | OpenStreetMap via osmnx geocode | GeoJSON | 12 municipalities | ✅ Real (OSM) / mock fallback |
| Public transit stops | Ministry of Transport GTFS / OSM | GeoJSON | 300 stops (mock) | 🔄 Mock data |
| Parks & green spaces | OpenStreetMap via osmnx | GeoJSON | ~60 parks (mock) | 🔄 Mock data |

> **Mock data** is generated with realistic Israeli real estate price ranges (2024 market values) and real geographic coordinates.
> Replace files in `data/raw/` to run with live data — the pipeline is production-ready.

---

## 🔧 Data Preparation

### Cleaning
- Remove transactions outside the Gush Dan bounding box (spatial clipping)
- Drop transactions with missing price, area, or coordinates
- Clip price outliers at the 1st and 99th percentile per municipality
- Fill missing room counts with the municipality median

### Encoding
- Derive `price_per_m2 = transaction_price / area_m2`
- Aggregate transactions per city: `avg_price`, `median_price`, `avg_price_per_m2`, `transaction_count`, `avg_rooms`, `avg_area_m2`

### Spatial Join
- **Transaction → City**: `gpd.sjoin(transactions, city_polygons, predicate="within")`
  → each transaction point gets the name of the city polygon it falls within
- **City centroid → Transit stops**: `gpd.sjoin_nearest(centroids, transit_stops)` with R-tree index
  → distance in metres to nearest stop per city
- **City centroid → Parks**: same pattern for park proximity
- **Density join**: buffered join — city centroids buffered at 500 m / 1 km, then intersect with stops/parks

### CRS Rules
All GeoDataFrames are kept in **EPSG:4326** at all times.
Metric distance calculations reproject locally to **EPSG:2039** (ITM) and return WGS84.

---

## 🤖 Machine Learning

### Task
**Supervised Regression** — predict city suitability score (0–100) from geospatial and real-estate features.

### Target Variable
`neighborhood_score` — composite score computed by the rule-based engine:
```
score = 0.40 × affordability + 0.35 × transit + 0.25 × parks   (× 100)
```

### Features
| Feature | Description |
|---|---|
| `avg_price` | Average apartment price (₪) |
| `avg_price_per_m2` | Average price per m² (₪/m²) |
| `avg_rooms` | Average number of rooms |
| `avg_area_m2` | Average apartment size (m²) |
| `transaction_count` | Number of transactions in city |
| `dist_to_nearest_transit_m` | Distance to nearest transit stop (m) |
| `dist_to_nearest_park_m` | Distance to nearest park (m) |
| `transit_stops_500m` | Count of transit stops within 500 m |
| `transit_stops_1km` | Count of transit stops within 1 km |
| `parks_1km` | Count of parks within 1 km |
| `transit_composite` | Derived: 1/(1+dist_transit) + stops_500m |
| `park_composite` | Derived: 1/(1+dist_park) + parks_1km |
| `centroid_x / centroid_y` | City centroid in ITM coordinates (spatial position) |

### KPI
**MAE (Mean Absolute Error)** — interpretable as "average error in score points".
Example: MAE = 3.2 means the model is off by ±3.2 points on average (out of 100).

### Evaluation Strategy
**Leave-One-Out Cross-Validation (LOO-CV)** — required due to small dataset (n=12 cities in mock data).
Each city is held out once as the test set; the model is trained on the remaining 11.

### Models Compared
| Model | Notes |
|---|---|
| Baseline (mean) | DummyRegressor — predicts the mean score for every city |
| Ridge Regression | Linear model with L2 regularisation + StandardScaler |
| **Random Forest** ✅ | 100 trees, max_depth=3 — best balance of accuracy and interpretability |
| Gradient Boosting | 50 estimators, max_depth=2 — sequential ensemble |

### Results (Mock Data — 12 Cities)
The Random Forest model captures the scoring function with low MAE and provides feature importances showing which geospatial dimensions drive city scores most.

---

## 📈 Project Stage

| Stage | Description | Status |
|---|---|---|
| **Descriptive** | Interactive choropleth map + ranking table showing current city scores | ✅ Complete |
| **Predictive** | Random Forest regressor predicts suitability scores from geospatial features; LOO-CV model comparison | ✅ Complete |
| **Prescriptive** | Per-city actionable recommendations targeting the weakest geospatial dimension | ✅ Complete |

---

## 🏗️ Repository Structure

```
project_geoai_tairnoy/
│
├── data/
│   ├── raw/          ← Raw input files (not committed — gitignored)
│   └── processed/    ← GeoParquet outputs from pipeline (gitignored)
│
├── notebooks/
│   ├── 01_data_loading.ipynb        ← Data inspection and loading
│   ├── 02_spatial_eda.ipynb         ← Exploratory spatial analysis
│   └── 03_feature_engineering.ipynb ← Feature computation walkthrough
│
├── src/
│   ├── config.py                    ← Paths, AOI, scoring weights (single source of truth)
│   │
│   ├── data/
│   │   ├── loaders.py               ← Load raw / OSM / mock data as GeoDataFrames
│   │   ├── preprocessing.py         ← Clean, clip, standardise
│   │   └── validation.py            ← Schema and spatial validation
│   │
│   ├── geo/
│   │   ├── crs_utils.py             ← CRS enforcement (EPSG:4326 everywhere)
│   │   ├── spatial_join.py          ← Spatial joins + transaction aggregation
│   │   └── distance_features.py     ← Distance/density features with R-tree index
│   │
│   ├── scoring/
│   │   ├── neighborhood_score.py    ← Rule-based weighted scoring engine
│   │   └── ml_scorer.py             ← ML regression model (Random Forest)
│   │
│   └── app/
│       └── map_utils.py             ← Folium map builders
│
├── streamlit_app.py    ← Main app entry point (descriptive + predictive + prescriptive)
├── requirements.txt
├── README.md
├── CLAUDE.md           ← Development guidelines
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tairnoy02/project_geoai_tairnoy.git
cd project_geoai_tairnoy
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

The ML layer is enabled by default — the ranking table includes both the rule-based score and the ML-predicted score side by side.
To hide the ML section, uncheck **"Show ML model results"** in the sidebar.

---

## 🔬 Scoring Methodology

### Rule-Based (Descriptive)
```
score = w_affordability × (1 − norm(avg_price))
      + w_transit       × transit_composite
      + w_parks         × parks_composite
```
All sub-scores normalised with Min-Max scaling. Final score scaled to 0–100.
Default weights: Affordability 40%, Transit 35%, Parks 25%. Adjustable via sidebar.

### ML-Based (Predictive)
Random Forest regressor trained on the same features. Learns the scoring function
from data and generalises to new cities. Evaluated with LOO-CV (MAE).

---

## 🌍 Area of Interest — Gush Dan Metropolitan Area

12 municipalities supported in the MVP:

| Municipality | Hebrew |
|---|---|
| Tel Aviv-Yafo | תל אביב-יפו |
| Ramat Gan | רמת גן |
| Givatayim | גבעתיים |
| Bnei Brak | בני ברק |
| Holon | חולון |
| Bat Yam | בת ים |
| Petah Tikva | פתח תקווה |
| Rishon LeZion | ראשון לציון |
| Herzliya | הרצליה |
| Ramat HaSharon | רמת השרון |
| Kiryat Ono | קרית אונו |
| Givat Shmuel | גבעת שמואל |

---

## 🙏 Acknowledgements

Built as part of the GeoAI academic course.
Geospatial stack: [GeoPandas](https://geopandas.org/) · [Folium](https://python-visualization.github.io/folium/) · [osmnx](https://osmnx.readthedocs.io/) · [Shapely](https://shapely.readthedocs.io/) · [scikit-learn](https://scikit-learn.org/).
