# ⚡ Agil EV Charging – MVP & Analogs + ML Forecasting

This project is a **data-driven decision tool** for SNDP Agil in Tunisia to plan electric vehicle (EV) charging infrastructure.  
It includes a **mapping dashboard**, **site ranking**, **EV adoption forecasting**, and an **analogs-based machine learning page** to estimate when the Tunisian market will be ready for EV chargers.

---

## 🚀 Features

### **MVP Core**
- 🗺 **Map View** – Displays current Agil stations and public EV charging points (from OpenStreetMap).
- 📊 **Ranking** – Ranks Agil stations by gap to the nearest charger (bigger gap = higher rollout priority).
- 📈 **Forecast** – Logistic model to project EV growth & chargers needed in Tunisia.
- 🎲 **Risk Simulation** – Monte Carlo simulation to estimate market viability year.

### **Analogs & ML Page**
- Fit **logistic curves** for EV adoption in **analog countries** (e.g., Morocco, Egypt, Türkiye).
- Fit logistic curves for **Tunisia’s historical tech adoption** (Internet, Mobile, etc.).
- Combine parameters to build **Tunisia EV adoption scenarios**.
- Export forecast CSV for integration into business planning.

---

## 🛠 Tech Stack
- **Python 3.10+**
- **Streamlit** – Interactive web app
- **Pandas / NumPy** – Data wrangling
- **SciPy** – Curve fitting
- **OpenStreetMap Overpass API** – Real geospatial data

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/agil-ev-mvp.git
cd agil-ev-mvp

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fetch OSM data (chargers + Agil sites)
python fetch_osm_tunisia_ev_agil.py

# 5. Compute nearest-charger distance & ranking
python process_sites.py

# 6. Launch the dashboard
streamlit run app.py
