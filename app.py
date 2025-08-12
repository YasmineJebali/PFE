import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import folium
from streamlit_folium import st_folium
import math

st.set_page_config(page_title="Agil EV MVP", layout="wide")

DATA_DIR = Path("data")
AGIL_CSV = DATA_DIR / "tunisia_agil_stations.csv"
CHARGERS_CSV = DATA_DIR / "tunisia_charging_stations.csv"
PROCESSED_CSV = DATA_DIR / "processed_sites.csv"

st.title("⚡ Agil EV Charging – MVP")

with st.sidebar:
    st.header("Data files")
    st.write("Expecting CSVs in `data/`")
    st.code("tunisia_agil_stations.csv\n"
            "tunisia_charging_stations.csv\n"
            "processed_sites.csv (after processing)")

def load_csv(path: Path):
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            st.error(f"Failed to read {path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

agil = load_csv(AGIL_CSV)
chargers = load_csv(CHARGERS_CSV)
processed = load_csv(PROCESSED_CSV)

tab_map, tab_rank, tab_forecast, tab_risk = st.tabs(["🗺️ Map", "🏆 Ranking", "📈 Forecast", "🎲 Risk"])

with tab_map:
    st.subheader("Map of Agil stations and public charging points")
    if agil.empty and chargers.empty:
        st.info("Run the fetch script first to populate data/.")
    else:
        # Center the map
        lat0 = np.nanmean(pd.concat([agil["lat"], chargers["lat"]], axis=0))
        lon0 = np.nanmean(pd.concat([agil["lon"], chargers["lon"]], axis=0))
        if np.isnan(lat0) or np.isnan(lon0):
            lat0, lon0 = 34.0, 9.0  # Tunisia approx
        m = folium.Map(location=[lat0, lon0], zoom_start=6)

        # Add chargers
        for _, row in chargers.iterrows():
            if pd.notnull(row.get("lat")) and pd.notnull(row.get("lon")):
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=4, color="#2a9d8f", fill=True, fill_opacity=0.8,
                    popup=f"⚡ {row.get('name','(Charging)')}<br>{row.get('brand','')}"
                ).add_to(m)

        # Add Agil
        for _, row in agil.iterrows():
            if pd.notnull(row.get("lat")) and pd.notnull(row.get("lon")):
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=4, color="#e76f51", fill=True, fill_opacity=0.8,
                    popup=f"⛽ {row.get('name','Agil station')}"
                ).add_to(m)

        st_folium(m, width=None, height=600)

with tab_rank:
    st.subheader("Site ranking (gap to nearest charger)")
    if processed.empty:
        st.info("Run `python process_sites.py` to create `data/processed_sites.csv`.")
    else:
        k = st.slider("Show top N sites", 5, 100, 20, 5)
        st.write("Sorted by `site_score` (higher = larger gap, likely higher priority).")
        st.dataframe(processed[["name","addr_city","lat","lon","nearest_charger_km","site_score"]].head(k))

with tab_forecast:
    st.subheader("Adoption & charger demand (scenario)")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input("Start year", min_value=2024, max_value=2040, value=2025, step=1)
        end_year = st.number_input("End year", min_value=start_year, max_value=2050, value=2035, step=1)
    with col2:
        K = st.number_input("Max EV stock (K)", min_value=5, max_value=1000, value=150, step=5)
        K = float(K) * 1000  # in units
        r = st.number_input("Growth rate r (logistic)", min_value=0.01, max_value=1.0, value=0.35, step=0.01)
    with col3:
        t0 = st.number_input("Midpoint year t0", min_value=2024, max_value=2050, value=2031, step=1)
        ratio = st.number_input("EVs per public charger", min_value=5, max_value=50, value=18, step=1)

    years = np.arange(start_year, end_year+1)
    # Logistic EV stock: K / (1 + exp(-r*(t-t0)))
    ev_stock = K / (1.0 + np.exp(-r * (years - t0)))
    chargers_needed = ev_stock / ratio

    df = pd.DataFrame({"year": years, "ev_stock": ev_stock.astype(int), "chargers_needed": chargers_needed.astype(int)})
    st.line_chart(df.set_index("year")[["ev_stock","chargers_needed"]])

    st.caption("Tune **K, r, t0** based on analog markets and policy assumptions.")

with tab_risk:
    st.subheader("Monte Carlo on adoption assumptions")
    st.write("We simulate uncertainty on r and t0, then compute the year when chargers_needed exceeds a threshold.")

    col1, col2, col3 = st.columns(3)
    with col1:
        ratio_mc = st.number_input("EVs per charger (risk)", min_value=5, max_value=50, value=18, step=1, key="ratio_mc")
        chargers_threshold = st.number_input("Threshold chargers needed (market viability)", min_value=100, max_value=20000, value=1500, step=50)
    with col2:
        r_mean = st.number_input("r mean", min_value=0.05, max_value=1.0, value=0.35, step=0.01)
        r_sd = st.number_input("r std", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
    with col3:
        t0_mean = st.number_input("t0 mean", min_value=2024, max_value=2050, value=2031, step=1, key="t0_mean")
        t0_sd = st.number_input("t0 std", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

    sims = st.number_input("Simulations", min_value=100, max_value=10000, value=2000, step=100)
    horizon = st.slider("Forecast horizon (years)", 2025, 2045, (2025, 2038))

    years = np.arange(horizon[0], horizon[1] + 1)
    K = st.number_input("Max EV stock K (risk)", min_value=10000, max_value=5000000, value=150000, step=10000)

    rng = np.random.default_rng(42)
    r_draws = rng.normal(r_mean, r_sd, size=int(sims))
    t0_draws = rng.normal(t0_mean, t0_sd, size=int(sims))

    hit_years = []
    for r_i, t0_i in zip(r_draws, t0_draws):
        ev = K / (1.0 + np.exp(-r_i * (years - t0_i)))
        chargers = ev / ratio_mc
        above = years[chargers >= chargers_threshold]
        if len(above) > 0:
            hit_years.append(int(above[0]))
        else:
            hit_years.append(np.nan)

    hit_series = pd.Series(hit_years)
    st.bar_chart(hit_series.value_counts().sort_index())

    p50 = hit_series.dropna().quantile(0.5) if hit_series.notna().any() else np.nan
    p90 = hit_series.dropna().quantile(0.9) if hit_series.notna().any() else np.nan
    st.write(f"**P50 year**: {p50:.0f}" if not math.isnan(p50) else "P50 year: n/a")
    st.write(f"**P90 year**: {p90:.0f}" if not math.isnan(p90) else "P90 year: n/a")
