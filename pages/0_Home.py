# pages/0_Home.py — Narrative Home & Project Map (guided steps + status)
from pathlib import Path
import sys
import pandas as pd
import streamlit as st


st.markdown("""
### 🏢 About the Host Company — SNDP Agil
**Société Nationale de Distribution des Pétroles (Agil)** is Tunisia’s leading fuel distribution company, 
operating over **200 service stations** nationwide.  
As part of its energy transition strategy, Agil aims to become a key player in **electric mobility**, 
by gradually deploying **public EV charging infrastructure** across its network.  

This project supports Agil’s strategic vision by providing a **data-driven decision system** 
to identify **where**, **when**, and **how many** chargers to deploy — balancing 
**market adoption**, **financial feasibility**, and **network coverage**.
""")



# --- Path bootstrap ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Helpers ---
def exists(rel: str) -> bool:
    p = ROOT / rel if rel != "app.py" else ROOT / "app.py"
    return p.exists()

def status_badge(ok: bool) -> str:
    return "✅" if ok else "❌"

def get_session_df(name: str):
    obj = st.session_state.get(name)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        return obj
    return None

# --- Sidebar: Guided steps ---
st.sidebar.title("🧭 Guided steps")
st.sidebar.markdown("""
1. **Data Intake** → load stations & forecast  
2. **Map & Gaps** → visualize network, coverage  
3. **Analogs & Modeling** → benchmark + ML  
4. **ROI & Sensitivity** → NPV, payback, tornado  
5. **Deployment ROI** → per-station ranking  
6. **Assistant** → Q&A over project context  
""")

# Data status tiles for quick sanity
stations_df = get_session_df("stations_df")
forecast_df = get_session_df("forecast_df")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data status**")
st.sidebar.markdown(f"- Stations: **{status_badge(stations_df is not None)}**")
st.sidebar.markdown(f"- Forecast: **{status_badge(forecast_df is not None)}**")

# --- Header ---
st.title("⚡ Project  Overview")
st.caption("A transparent, model-first decision tool for **when/where/how** to deploy EV chargers in Tunisia (SNDP Agil).")

# --- Scenario strip (the story) ---
with st.container():
    st.markdown("### The storyline")
    st.markdown("""
**Goal:** Recommend a realistic, risk-aware rollout for public EV charging at SNDP Agil.  
**How:** combine **analog markets**, **forecasting models**, and **finance** → then **rank stations** to deploy first.
""")
    st.markdown(
        """
**End-to-end flow**

1) Data Intake → 2) Map & Gaps → 3) Analogs & Modeling → 4) ROI & Sensitivity → 5) Deployment ROI
"""
    )

# --- Data status bar ---
ok_st = stations_df is not None
ok_fc = forecast_df is not None
st.info(f"**Data snapshot** • Stations loaded: {status_badge(ok_st)} • Forecast loaded: {status_badge(ok_fc)}")

# --- Quick-start cards ---
st.write("---")
st.subheader("Quick start")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 1) Data Intake")
    st.caption("Load Agil stations and the Tunisia adoption/chargers forecast.")
    if exists("pages/0_Smoke_Test.py"):
        st.page_link("pages/0_Smoke_Test.pyy", label="Open Data Intake", icon="🧪")
    else:
        st.warning("Page missing: pages/0_Smoke_Test.py")

with c2:
    st.markdown("#### 2) Map & Gaps")
    st.caption("See network coverage, density, underserved areas, and nearest-charger gaps.")
    if exists("app.py"):
        st.page_link("app.py", label="Open Map & Gaps", icon="🗺️")
    else:
        st.warning("Missing app.py (Map page).")

with c3:
    st.markdown("#### 3) Analogs & Modeling")
    st.caption("Benchmark Tunisia vs analogs + test ML models (rolling-origin CV).")
    if exists("pages/2_Analogs_and_ML.py"):
        st.page_link("pages/2_Analogs_and_ML.py", label="Open Analogs & ML", icon="📊")
    else:
        st.warning("Page missing: pages/2_Analogs_and_ML.py")

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown("#### 4) ROI & Sensitivity")
    st.caption("NPV, payback, discounted payback, IRR, tornado sensitivity, Excel export.")
    if exists("pages/4_ROI_Sensitivity.py"):
        st.page_link("pages/4_ROI_Sensitivity.py", label="Open ROI & Sensitivity", icon="💸")
    else:
        st.warning("Page missing: pages/4_ROI_Sensitivity.py")

with c5:
    st.markdown("#### 5) Deployment ROI")
    st.caption("Allocate chargers by year and rank stations by NPV/payback (round-robin or weighted).")
    if exists("pages/5_Deployment_ROI.py"):
        st.page_link("pages/5_Deployment_ROI.py", label="Open Deployment ROI", icon="🧭")
    else:
        st.warning("Page missing: pages/5_Deployment_ROI.py")

with c6:
    st.markdown("#### Assistant")
    st.caption("Ask questions about your data, forecasts, and methods.")
    if exists("pages/6_Chatbot.py"):
        st.page_link("pages/6_Chatbot.py", label="Open Assistant", icon="🤖")
    else:
        st.warning("Page missing: pages/6_Chatbot.py")

# --- What’s inside each page (for the narrative) ---
st.write("---")
st.subheader("What each page gives you")
with st.expander("🧪 Data Intake"):
    st.markdown("""
- Upload **stations.csv** (id, name, lat, lon[, gov, features]) and **forecast.csv** (year, chargers_needed[, ev_stock]).
- We validate types & required columns and save them into session for the rest of the app.
""")

with st.expander("🗺️ Map & Gaps"):
    st.markdown("""
- Visual map of the Agil network (heat, markers), coverage gaps (e.g., nearest public charger distance), and filters by governorate.
- Ranking tab to highlight **underserved** or **high-potential** clusters.
""")

with st.expander("📊 Analogs & Modeling"):
    st.markdown("""
- Compare **Tunisia vs analog markets** (e.g., Morocco/Egypt/EEA) and fit **Logistic** + **ML models** with **rolling-origin CV**.
- Choose the best model(s) for **EV stock** or **public chargers** series; export forecasts.
""")

with st.expander("💸 ROI & Sensitivity"):
    st.markdown("""
- Compute **NPV**, **payback**, **discounted payback**, **IRR** per charger or as a forecast-linked portfolio.
- Run **tornado** sensitivity (±20%) and export an Excel pack with assumptions and cashflows.
""")

with st.expander("🧭 Deployment ROI"):
    st.markdown("""
- Convert cumulative forecast into yearly **adds**, allocate new chargers to stations (round-robin or **weighted** by need).
- Compute station-level **cashflows**, **NPV**, and **payback**, and **rank** candidates; export details per-station.
""")

with st.expander("🤖 Assistant"):
    st.markdown("""
- Ask “How many chargers in 2029?”, “Stations in Sfax?”, “What’s the NPV at 11%?”  
- If the doc index is built, it answers doc questions too; otherwise it uses the live data.
""")
# --- Page config ---
st.set_page_config(page_title="Home • Agil EV", layout="wide")
st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")



# --- Data previews (if available) ---
st.write("---")
st.subheader("Tiny previews (if you already loaded data)")
show_prev = st.checkbox("Show small previews")
if show_prev:
    cA, cB = st.columns([1.2, 1])
    with cA:
        if stations_df is not None:
            st.markdown("**Stations — first rows**")
            st.dataframe(stations_df.head(12), use_container_width=True, height=260)
        else:
            st.info("No stations in session yet.")
    with cB:
        if forecast_df is not None:
            st.markdown("**Forecast — first rows**")
            st.dataframe(forecast_df.head(12), use_container_width=True, height=260)
        else:
            st.info("No forecast in session yet.")

# --- Footer: sanity + next actions ---
st.write("---")
if not ok_st or not ok_fc:
    st.warning("To unlock the full workflow, please go to **Data Intake** and load both datasets.")
else:
    st.success("All set! Proceed to **Map & Gaps** or **Analogs & Modeling** to continue the story.")
    if exists("app.py"):
        st.page_link("app.py", label="Next → Map & Gaps", icon="🗺️")


