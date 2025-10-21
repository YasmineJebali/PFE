# pages/0_Smoke_Test.py — Explore & Validate Data (fixed)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explore Data — Agil EV MVP", layout="wide")
st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")

st.title("📊 Explore & Validate Data")
st.caption("Upload one CSV (stations or forecast), preview it, and push it to session for the next pages.")

uploaded_file = st.file_uploader(
    "Upload your data file (e.g., tunisia_agil_stations.csv OR a forecast CSV)",
    type=["csv"]
)

def _is_stations_df(df: pd.DataFrame) -> bool:
    req = {"lat", "lon"}
    # treat as stations if it has geo columns and looks like a sites table
    return req.issubset(set(c.strip().lower() for c in df.columns))

def _is_forecast_df(df: pd.DataFrame) -> bool:
    cols = {c.strip().lower() for c in df.columns}
    return ("year" in cols) and ("chargers_needed" in cols or "ev_stock" in cols)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File **{uploaded_file.name}** loaded successfully!")

    # ---------- Preview ----------
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------- Summary ----------
    st.divider()
    st.subheader("📈 Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing values", int(df.isna().sum().sum()))

    with st.expander("Show column details"):
        # fully fixed line (your error came from a truncated '.transpose(')
        try:
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        except Exception:
            st.info("Describe summary not available for this mix of column types.")

    # ---------- Push to session (auto-detect type) ----------
    st.divider()
    stored_as = None
    lower_cols = {c.strip().lower() for c in df.columns}

    if _is_stations_df(df):
        st.session_state["stations_df"] = df.copy()
        stored_as = "stations_df"
    if _is_forecast_df(df):
        st.session_state["forecast_df"] = df.copy()
        # if it looks like both (rare), prefer explicit buttons below
        stored_as = "forecast_df" if stored_as is None else stored_as

    st.success(
        "Saved to session: "
        + ("**stations_df** " if "stations_df" in st.session_state else "")
        + ("**forecast_df**" if "forecast_df" in st.session_state else "")
        if ("stations_df" in st.session_state or "forecast_df" in st.session_state)
        else "Loaded in memory (not saved to session yet)."
    )

    # Manual override buttons (if auto-detect guessed wrong)
    st.caption("If the auto-detection guessed wrong, choose one:")
    b1, b2 = st.columns(2)
    if b1.button("Save as Stations"):
        st.session_state["stations_df"] = df.copy()
        st.success("Saved as **stations_df**.")
    if b2.button("Save as Forecast"):
        st.session_state["forecast_df"] = df.copy()
        st.success("Saved as **forecast_df**.")

    # ---------- Next steps ----------
    st.divider()
    st.info("✅ Data is ready to be used in the next pages:")
    st.page_link("app.py", label="Next → Map & Gaps", icon="🗺️")
    st.page_link("pages/3_Modeling_Lab.py", label="Next → Modeling Lab", icon="🧪")
else:
    st.warning("📂 Please upload one CSV file to begin exploring your data.")
    st.caption("Tip: start with `tunisia_agil_stations.csv` from your `data/` folder.")