# 🏠 Where Should I Buy?
### GeoAI-Powered Apartment Suitability Analysis — Gush Dan Metropolitan Area

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14-green)](https://geopandas.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

**Where Should I Buy?** is a GeoAI-powered decision support system that helps apartment buyers identify the best neighborhoods in the Gush Dan (Tel Aviv metropolitan) area by combining:

- 🏘️ **Real estate transaction data** — historical sale prices, apartment size, room count
- 🚌 **Public transportation accessibility** — proximity and density of transit stops
- 🌳 **Parks and green spaces** — distance and count of parks within walking distance
- 🗺️ **Neighborhood boundaries** — administrative polygons for spatial aggregation

The system produces a **composite suitability score (0–100)** for each neighborhood and visualizes the results on an interactive map.

---

## 🎯 GeoAI Motivation

Traditional apartment search relies on price listings and manual exploration. This project demonstrates how **geospatial AI** transforms that process:

| Challenge | GeoAI Solution |
|---|---|
| Comparing 12+ municipalities at once | Spatial aggregation + choropleth map |
| "How far is transit?" | R-tree nearest-neighbor search in metric space |
| Multi-criteria tradeoffs | Weighted composite scoring engine |
| User-specific priorities | Dynamic weight adjustment via Streamlit sidebar |
| Scalable to all of Israel | Modular pipeline with configuration-driven AOI |

The scoring engine is intentionally designed as a **drop-in replacement target for ML** — the same data pipeline feeds a future scikit-learn model.

---

## 🗺️ Area of Interest (AOI)

The MVP supports these Gush Dan municipalities:

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

## 📊 Data Sources

| Layer | Source | Format | Status |
|---|---|---|---|
| Real estate transactions | Israeli Land Authority (רשות מקרקעי ישראל) | CSV | 🔄 Mock data |
| Neighborhood boundaries | Israeli CBS Statistical Areas | GeoJSON | 🔄 Mock data |
| Public transit stops | Ministry of Transport GTFS / OSM | GeoJSON | 🔄 Mock data |
| Parks & green spaces | OpenStreetMap via osmnx | GeoJSON | 🔄 Mock data |

> **Mock data is used in the MVP.** All spatial analysis and scoring logic is production-ready.
> Replace mock files with real data in `data/raw/` to run with live data.

---

## 🏗️ Repository Structure

```
where_should_i_buy/
│
├── data/
│   ├── raw/          ← Raw input files (not committed)
│   ├── processed/    ← GeoParquet outputs from pipeline
│   └── external/     ← Downloaded external datasets
│
├── notebooks/
│   ├── 01_data_loading.ipynb       ← Data inspection and loading
│   ├── 02_spatial_eda.ipynb        ← Exploratory spatial analysis
│   └── 03_feature_engineering.ipynb← Feature computation walkthrough
│
├── src/
│   ├── config.py                   ← Paths, AOI, default weights (SINGLE SOURCE OF TRUTH)
│   │
│   ├── data/
│   │   ├── loaders.py              ← Load raw/mock data as GeoDataFrames
│   │   ├── preprocessing.py        ← Clean, clip, standardise
│   │   └── validation.py           ← Schema and spatial validation
│   │
│   ├── geo/
│   │   ├── crs_utils.py            ← CRS enforcement (EPSG:4326 everywhere)
│   │   ├── spatial_join.py         ← Spatial joins + transaction aggregation
│   │   └── distance_features.py    ← Distance/density features with R-tree index
│   │
│   ├── scoring/
│   │   └── neighborhood_score.py   ← Weighted scoring engine (ML-ready interface)
│   │
│   └── app/
│       └── map_utils.py            ← Folium map builders
│
├── streamlit_app.py    ← Main app entry point
├── requirements.txt
├── README.md
├── CLAUDE.md           ← Development guidelines for Claude
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/where-should-i-buy.git
cd where-should-i-buy
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

> **Note:** GeoPandas requires GEOS and GDAL. On macOS, run `brew install gdal` first.
> On Ubuntu: `sudo apt install libgdal-dev`.

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501).

### 5. (Optional) Open notebooks

```bash
jupyter lab
```

---

## 🔬 Scoring Methodology

The suitability score is a weighted linear combination of three normalized sub-scores:

```
score = w_affordability × (1 - norm(avg_price))
      + w_transit       × transit_composite
      + w_parks         × parks_composite
```

Where:
- **Affordability**: Inverted normalized average price (lower price = higher score)
- **Transit composite**: Combines inverse distance to nearest stop + stop density within 500m
- **Parks composite**: Combines inverse distance to nearest park + park count within 1km
- **Normalization**: Min-max scaling across all neighborhoods in the dataset
- **Final score**: Scaled to 0–100

Default weights: Affordability 40%, Transit 35%, Parks 25%. Adjustable via UI.

---

## 🛣️ Future Roadmap

### Phase 2 — Real Data Integration
- [ ] Fetch Israeli Land Authority transaction data (API or download)
- [ ] Fetch OSM transit stops via `osmnx`
- [ ] Load Israeli CBS statistical area boundaries
- [ ] Build automated data update pipeline

### Phase 3 — ML Scoring Model
- [ ] Collect user preference labels (good/bad neighborhood ratings)
- [ ] Train `RandomForestRegressor` / `GradientBoostingRegressor` as scoring replacement
- [ ] SHAP explainability for feature importance
- [ ] Cross-validation + leaderboard

### Phase 4 — Advanced Spatial Features
- [ ] Walking isochrones from transit stops (osmnx network analysis)
- [ ] School quality integration (Israeli Ministry of Education data)
- [ ] Noise pollution / air quality layers
- [ ] Property price trend (time-series analysis)

### Phase 5 — Deployment
- [ ] Git repository + GitHub Actions CI
- [ ] Streamlit Community Cloud deployment
- [ ] Automated data refresh schedule

---

## 🧑‍💻 Development

See [CLAUDE.md](CLAUDE.md) for coding standards and development guidelines.

```bash
# Run notebooks
jupyter lab notebooks/

# Validate data pipeline (quick smoke test)
python -c "
from src.data.loaders import load_transactions, load_neighborhoods
t = load_transactions()
n = load_neighborhoods()
print(f'Transactions: {len(t)}, Neighborhoods: {len(n)}')
print(f'CRS: {t.crs}')
"
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built as part of the GeoAI academic course.
Geospatial stack: [GeoPandas](https://geopandas.org/), [Folium](https://python-visualization.github.io/folium/), [osmnx](https://osmnx.readthedocs.io/), [Shapely](https://shapely.readthedocs.io/).
