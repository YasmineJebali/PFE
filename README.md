# Agil EV Charging – MVP

This is a minimal, *functional* starter you can run today.

## Quickstart

```bash
# 1) Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Fetch OpenStreetMap data (chargers + Agil sites)
python fetch_osm_tunisia_ev_agil.py

# 4) Compute nearest-charger distance & ranking
python process_sites.py

# 5) Launch the Streamlit app
streamlit run app.py
```

## What you get

- A **map** of Agil stations and public charging points (from OSM).
- A **ranking** of Agil sites by gap to nearest charger (bigger gap = higher priority).
- A **forecast** tab with sliders for logistic adoption & chargers needed.
- A **risk** tab with Monte Carlo on adoption assumptions to get P50/P90 years.

## Data folder

All data is written to `data/`:
- `tunisia_charging_stations.csv`
- `tunisia_agil_stations.csv`
- `processed_sites.csv` (after processing)

You can re-run steps 3–4 anytime to refresh the data.
