# app.py — Map & Gaps (polished UI + clustered markers + robust guards)
import streamlit as st
st.set_page_config(page_title="Agil EV MVP", layout="wide")

import pandas as pd
import numpy as np
from pathlib import Path
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import altair as alt
import math

from utils.ui import set_page, sidebar_nav, breadcrumbs, status_pills
set_page("Agil EV MVP • Map & Gaps", icon="🗺️")
sidebar_nav(__file__)
breadcrumbs([
    ("Home", "pages/0_Home.py"),
    ("Map & Gaps", "app.py"),
])
status_pills()



# breadcrumb & nav
st.caption("Path: Home → Data Intake → **Map & Gaps** → Analogs & ML → ROI → Deployment ROI")
st.page_link("pages/0_Home.py", label="🏠 Home", icon="🏠")
st.page_link("pages/2_Analogs_and_ML.py", label="Next → Analogs & ML", icon="📊")

DATA_DIR = Path("data")
AGIL_CSV = DATA_DIR / "tunisia_agil_stations.csv"
CHARGERS_CSV = DATA_DIR / "tunisia_charging_stations.csv"
PROCESSED_CSV = DATA_DIR / "processed_sites.csv"

st.title("⚡ Agil EV Charging – MVP")

with st.sidebar:
    st.header("Data files")
    st.write("Expecting CSVs in `data/`")
    st.code(
        "tunisia_agil_stations.csv\n"
        "tunisia_charging_stations.csv\n"
        "processed_sites.csv (after processing)"
    )
    st.markdown("---")
    st.caption("👩‍💻 *Yasmine Jebali — PFE 2025*\n\n**Prédiction du déploiement des bornes VE — SNDP Agil**")

# ----------------------- IO helpers -----------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV safely → always return a pandas DataFrame (or empty DF)."""
    try:
        if path.exists():
            df = pd.read_csv(path)
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df)
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

agil = load_csv(AGIL_CSV)
chargers = load_csv(CHARGERS_CSV)
processed = load_csv(PROCESSED_CSV)

# extra guard in case upstream changes
if isinstance(agil, np.ndarray): agil = pd.DataFrame(agil)
if isinstance(chargers, np.ndarray): chargers = pd.DataFrame(chargers)
if isinstance(processed, np.ndarray): processed = pd.DataFrame(processed)

# NEW: expose stations to other pages (ROI, Deployment)
try:
    if isinstance(agil, pd.DataFrame) and not agil.empty:
        # prefer OSM id if present; fallback to sequential ids
        df_st = agil.copy()
        if "osm_id" in df_st.columns:
            df_st = df_st.rename(columns={"osm_id": "id"})
        if "id" not in df_st.columns:
            df_st["id"] = np.arange(1, len(df_st) + 1)
        # keep the essentials; make sure name exists
        if "name" not in df_st.columns:
            df_st["name"] = "Agil station"
        df_st = df_st[["id", "name", "lat", "lon"]].dropna(subset=["lat", "lon"]).reset_index(drop=True)
        st.session_state["stations_df"] = df_st
except Exception:
    pass


# ---------------- Governorate reference (24) ----------------
GOV_COORDS = {
    "Ariana": (36.8625, 10.1956),
    "Beja": (36.7330, 9.1830),
    "Ben Arous": (36.7472, 10.3333),
    "Bizerte": (37.2670, 9.8670),
    "Gabes": (33.8830, 10.1170),
    "Gafsa": (34.4170, 8.7830),
    "Jendouba": (36.5000, 8.7830),
    "Kairouan": (35.6670, 10.1000),
    "Kasserine": (35.1670, 8.8330),
    "Kebili": (33.7019, 8.9736),
    "Kef": (36.1822, 8.7147),
    "Mahdia": (35.5000, 11.0670),
    "Manouba": (36.8078, 10.1011),
    "Medenine": (33.3547, 10.5053),
    "Monastir": (35.7830, 10.8330),
    "Nabeul": (36.7500, 10.7500),
    "Sfax": (34.7330, 10.7670),
    "Sidi Bouzid": (35.0330, 9.5000),
    "Siliana": (36.1670, 9.3670),
    "Sousse": (35.8330, 10.6330),
    "Tataouine": (32.9256, 10.4442),
    "Tozeur": (33.9170, 8.1330),
    "Tunis": (36.8000, 10.1700),
    "Zaghouan": (36.4000, 10.1500)
}

# 2024-ish population (approx.)
GOV_POP = {
    "Ariana": 668_552, "Beja": 311_417, "Ben Arous": 722_828, "Bizerte": 607_388,
    "Gabes": 410_847, "Gafsa": 388_776, "Jendouba": 404_352, "Kairouan": 600_803,
    "Kasserine": 492_741, "Kebili": 183_201, "Kef": 237_686, "Mahdia": 449_985,
    "Manouba": 418_354, "Medenine": 537_255, "Monastir": 599_769, "Nabeul": 863_172,
    "Sfax": 1_047_468, "Sidi Bouzid": 489_991, "Siliana": 216_242, "Sousse": 762_281,
    "Tataouine": 162_654, "Tozeur": 120_036, "Tunis": 1_075_306, "Zaghouan": 201_065
}
TOTAL_POP = sum(GOV_POP.values())
GOV_ORDER = list(GOV_COORDS.keys())

def nearest_governorate(lat: float, lon: float) -> str | None:
    """Haversine distance to governorate centroids; return closest name."""
    min_dist = float("inf")
    nearest = None
    for gov, (glat, glon) in GOV_COORDS.items():
        dlat = math.radians(lat - glat)
        dlon = math.radians(lon - glon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(glat))*math.cos(math.radians(lat))*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist = 6371 * c
        if dist < min_dist:
            min_dist = dist
            nearest = gov
    return nearest

# ---------------------------- TABS ----------------------------
tab_map, tab_rank, tab_forecast, tab_risk = st.tabs(
    ["🗺️ Map", "🏆 Ranking", "📈 Forecast", "🎲 Risk"]
)

# ============================= MAP =============================
with tab_map:
    st.subheader("Map of Agil stations and public charging points")
    st.info(
        "This section shows the **current infrastructure footprint**. "
        "Use it to spot geographic gaps and clusters: ⚡ public chargers vs ⛽ Agil stations."
    )

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agil stations loaded", f"{0 if agil.empty else len(agil):,}")
    c2.metric("Public chargers loaded", f"{0 if chargers.empty else len(chargers):,}")
    if not agil.empty and not chargers.empty:
        cov = len(chargers) / max(len(agil), 1)
        c3.metric("Chargers per Agil site", f"{cov:.2f}")
    else:
        c3.metric("Chargers per Agil site", "—")
    c4.metric("Governorates in data", f"{len(GOV_COORDS)}")

    # Compute a reasonable center
    def _center_latlon(agil_df: pd.DataFrame, ch_df: pd.DataFrame) -> tuple[float, float]:
        lat_series = pd.Series(dtype=float)
        lon_series = pd.Series(dtype=float)
        if isinstance(agil_df, pd.DataFrame) and not agil_df.empty and {"lat","lon"}.issubset(agil_df.columns):
            lat_series = pd.concat([lat_series, agil_df["lat"]], ignore_index=True)
            lon_series = pd.concat([lon_series, agil_df["lon"]], ignore_index=True)
        if isinstance(ch_df, pd.DataFrame) and not ch_df.empty and {"lat","lon"}.issubset(ch_df.columns):
            lat_series = pd.concat([lat_series, ch_df["lat"]], ignore_index=True)
            lon_series = pd.concat([lon_series, ch_df["lon"]], ignore_index=True)
        lat0 = float(np.nanmean(lat_series)) if not lat_series.empty else np.nan
        lon0 = float(np.nanmean(lon_series)) if not lon_series.empty else np.nan
        if np.isnan(lat0) or np.isnan(lon0):
            return 34.0, 9.0  # Tunisia approx center
        return lat0, lon0

    lat0, lon0 = _center_latlon(agil, chargers)
    m = folium.Map(location=[lat0, lon0], zoom_start=6)

    # Chargers (clustered)
    if isinstance(chargers, pd.DataFrame) and not chargers.empty and {"lat", "lon"}.issubset(chargers.columns):
        ch_cluster = MarkerCluster(name="Public chargers").add_to(m)
        for _, row in chargers.iterrows():
            if pd.notnull(row.get("lat")) and pd.notnull(row.get("lon")):
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=4, color="#2a9d8f", fill=True, fill_opacity=0.8,
                    popup=f"⚡ {row.get('name','(Charging)')}<br>{row.get('brand','')}"
                ).add_to(ch_cluster)
    else:
        st.warning("⚠️ No valid charger data found (run fetch script or check CSV columns).")

    # Agil stations (clustered)
    if isinstance(agil, pd.DataFrame) and not agil.empty and {"lat", "lon"}.issubset(agil.columns):
        ag_cluster = MarkerCluster(name="Agil stations").add_to(m)
        for _, row in agil.iterrows():
            if pd.notnull(row.get("lat")) and pd.notnull(row.get("lon")):
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=4, color="#e76f51", fill=True, fill_opacity=0.8,
                    popup=f"⛽ {row.get('name','Agil station')}"
                ).add_to(ag_cluster)
    else:
        st.warning("⚠️ No valid Agil station data found (run fetch script or check CSV columns).")

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=600)

    with st.expander("🔎 Data snapshots"):
        cL, cR = st.columns(2)
        with cL:
            st.caption("Agil stations (sample)")
            st.dataframe(agil.head(10), use_container_width=True)
        with cR:
            st.caption("Public chargers (sample)")
            st.dataframe(chargers.head(10), use_container_width=True)

# ========== RANKING (CITY VIEW + CURVES) ==========
with tab_rank:
    st.subheader("City (Governorate) EV Infrastructure & Demand")
    st.info(
        "We map each Agil station to the nearest governorate and compare "
        "**population share** vs **station share** to spot *underserved* regions. "
        "Then we project city-level EV demand by scaling a national logistic curve by population share."
    )

    if not (isinstance(agil, pd.DataFrame) and not agil.empty and {"lat","lon"}.issubset(agil.columns)):
        st.info("Run the fetch script first to populate `data/` or check your station columns.")
    else:
        # Cache mapping to governorate (recompute if count changed)
        if (
            "agil_governorate" not in st.session_state
            or not isinstance(st.session_state.agil_governorate, pd.DataFrame)
            or len(st.session_state.agil_governorate) != len(agil)
        ):
            agil_copy = agil.copy()
            agil_copy["governorate"] = agil_copy.apply(
                lambda r: nearest_governorate(r["lat"], r["lon"]) if pd.notnull(r["lat"]) and pd.notnull(r["lon"]) else None,
                axis=1
            )
            st.session_state.agil_governorate = agil_copy[["lat","lon","governorate"]]

        agil_gov = st.session_state.agil_governorate.copy()
        counts = agil_gov["governorate"].value_counts().reindex(GOV_ORDER, fill_value=0)

        gov_df = pd.DataFrame({
            "governorate": counts.index,
            "station_count": counts.values
        })
        gov_df["population"] = gov_df["governorate"].map(GOV_POP)
        gov_df["pop_share"] = gov_df["population"] / TOTAL_POP

        total_stations = gov_df["station_count"].sum()
        gov_df["station_share"] = gov_df["station_count"] / max(total_stations, 1)
        gov_df["gap_index"] = (gov_df["pop_share"] - gov_df["station_share"]).round(4)  # + => underserved

        # KPI strip
        c1, c2, c3 = st.columns(3)
        c1.metric("Stations mapped", f"{int(total_stations):,}")
        c2.metric("Median gap index", f"{gov_df['gap_index'].median():+.3f}")
        top_underserved = gov_df.sort_values("gap_index", ascending=False).head(1)
        c3.metric("Top underserved", top_underserved.iloc[0]["governorate"] if not top_underserved.empty else "—")

        # Gap chart (population share vs station share)
        st.markdown("### Balance: population vs stations")
        df_long = gov_df.melt(
            id_vars=["governorate"], value_vars=["pop_share", "station_share"],
            var_name="metric", value_name="share"
        )
        chart = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("governorate:N", sort=GOV_ORDER, title="Governorate"),
                y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share"),
                color=alt.Color("metric:N", title="Metric", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("governorate:N"),
                    alt.Tooltip("metric:N"),
                    alt.Tooltip("share:Q", format=".1%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        # Controls for city curves
        st.divider()
        st.markdown("### City EV curves (national logistic × population share)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start_year = st.number_input("Start year", min_value=2024, max_value=2040, value=2025, step=1, key="city_start")
        with c2:
            end_year = st.number_input("End year", min_value=start_year, max_value=2050, value=2035, step=1, key="city_end")
        with c3:
            K = float(st.number_input("National Max EV stock K (thousands)", min_value=5, max_value=1000, value=150, step=5, key="cityK")) * 1000
            r = st.number_input("Growth rate r", min_value=0.01, max_value=1.0, value=0.35, step=0.01, key="cityr")
        with c4:
            t0 = st.number_input("Midpoint year t0", min_value=2024, max_value=2050, value=2031, step=1, key="cityt0")

        years = np.arange(start_year, end_year + 1)
        national_ev = K / (1.0 + np.exp(-r * (years - t0)))

        # City EV level by scaling national curve with population share
        city_curves = []
        for _, row in gov_df.iterrows():
            curve = (row["pop_share"] * national_ev)
            city_curves.append(pd.DataFrame({
                "year": years,
                "governorate": row["governorate"],
                "ev_stock_city": curve
            }))
        city_curves_df = pd.concat(city_curves, ignore_index=True)

        # 2030 point estimate (or end_year if earlier)
        target_year = min(2030, end_year)
        nat_2030 = float(K / (1.0 + np.exp(-r * (target_year - t0))))
        gov_df[f"est_ev_{target_year}"] = (gov_df["pop_share"] * nat_2030).astype(int)

        # Summary & top underserved table
        st.markdown("### Summary tables")
        left, right = st.columns([1.5, 1])
        with left:
            show_cols = ["governorate", "population", "station_count", "pop_share", "station_share", "gap_index", f"est_ev_{target_year}"]
            st.dataframe(
                gov_df.sort_values(["gap_index"], ascending=False).reset_index(drop=True)[show_cols],
                use_container_width=True, height=360
            )
        with right:
            st.write("**Top 8 underserved (by gap index)**")
            st.table(
                gov_df.sort_values("gap_index", ascending=False).head(8)[["governorate","gap_index"]]
                .rename(columns={"governorate":"Gov","gap_index":"Gap"})
            )

        # Interactive selection for curves
        st.markdown("### City EV curves")
        default_top = gov_df.sort_values("gap_index", ascending=False)["governorate"].head(5).tolist()
        selected = st.multiselect(
            "Select governorates to plot",
            options=gov_df["governorate"].tolist(),
            default=default_top,
            help="Tip: start with the most underserved to visualize demand build-up."
        )
        if selected:
            wide = (city_curves_df[city_curves_df["governorate"].isin(selected)]
                    .pivot(index="year", columns="governorate", values="ev_stock_city")
                    .round(0))
            st.line_chart(wide)
            st.caption("Curves show estimated **EV stock per governorate** over time (national logistic × population share).")
        else:
            st.info("Pick at least one governorate to draw curves.")

# ======================== NATIONAL FORECAST ========================
with tab_forecast:
    st.subheader("Adoption & charger demand (scenario)")
    st.info("Model national EV adoption with a logistic curve and derive charger needs via EV/charger ratio.")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input("Start year", min_value=2024, max_value=2040, value=2025, step=1)
        end_year = st.number_input("End year", min_value=start_year, max_value=2050, value=2035, step=1)
    with col2:
        K = st.number_input("Max EV stock (K, thousands)", min_value=5, max_value=1000, value=150, step=5)
        K = float(K) * 1000  # in units
        r = st.number_input("Growth rate r (logistic)", min_value=0.01, max_value=1.0, value=0.35, step=0.01)
    with col3:
        t0 = st.number_input("Midpoint year t0", min_value=2024, max_value=2050, value=2031, step=1)
        ratio = st.number_input("EVs per public charger", min_value=5, max_value=50, value=18, step=1)

    years = np.arange(start_year, end_year+1)
    ev_stock = K / (1.0 + np.exp(-r * (years - t0)))
    chargers_needed = ev_stock / ratio

    df = pd.DataFrame({"year": years, "ev_stock": ev_stock.astype(int), "chargers_needed": chargers_needed.astype(int)})
    st.line_chart(df.set_index("year")[["ev_stock","chargers_needed"]])
    st.caption("Tune **K, r, t0** based on analog markets and policy assumptions.")
    # NEW — save forecast to session
try:
    st.session_state["forecast_df"] = df.copy()
    st.toast("Forecast saved for ROI pages.", icon="💾")
except Exception:
    pass



# ============================== RISK ==============================
with tab_risk:
    st.subheader("Monte Carlo on adoption assumptions")
    st.info("Simulate uncertainty on r and t0 to estimate the **probable year** when charger demand crosses a threshold.")

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
        hit_years.append(int(above[0]) if len(above) > 0 else np.nan)

    hit_series = pd.Series(hit_years)
    st.bar_chart(hit_series.value_counts().sort_index())

    import math as _math
    p50 = hit_series.dropna().quantile(0.5) if hit_series.notna().any() else np.nan
    p90 = hit_series.dropna().quantile(0.9) if hit_series.notna().any() else np.nan
    st.write(f"**P50 year**: {p50:.0f}" if not _math.isnan(p50) else "P50 year: n/a")
    st.write(f"**P90 year**: {p90:.0f}" if not _math.isnan(p90) else "P90 year: n/a")
