import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from models.logistic_bass import logistic, fit_logistic

st.set_page_config(page_title="Analogs & ML", layout="wide")
st.title("🔁 Analogs & ML Forecasting")
st.write("Use analog countries and Tunisia tech adoption to fit S-curves and generate EV forecasts.")

with st.expander("📥 Inputs"):
    st.write("Upload analog EV dataset (CSV). Columns: country, iso3, year, ev_stock, public_chargers")
    analog_file = st.file_uploader("Analog EV CSV", type=["csv"], key="analog")
    st.write("---")
    st.write("Upload Tunisia tech adoption (CSV). Columns: tech, year, adoption_pct")
    tn_tech_file = st.file_uploader("Tunisia tech CSV", type=["csv"], key="tntech")

colA, colB = st.columns(2)

with colA:
    st.subheader("Fit logistic to analog EV stocks")
    if analog_file is not None:
        dfA = pd.read_csv(analog_file)
        countries = dfA["country"].dropna().unique().tolist()
        if countries:
            pick = st.selectbox("Pick a country", countries)
            sub = dfA[dfA["country"]==pick].dropna(subset=["year","ev_stock"])
            if len(sub) >= 4:
                fit = fit_logistic(sub["year"].values, sub["ev_stock"].values)
                st.write(f"**{pick} fit:** K={fit.K:.0f}, r={fit.r:.3f}, t0={fit.t0:.1f}")
                years = np.arange(sub["year"].min()-2, sub["year"].max()+10)
                pred = logistic(years, fit.K, fit.r, fit.t0)
                df_plot = pd.DataFrame({"year": years, "ev_stock_fit": pred.astype(int)}).merge(
                    sub[["year","ev_stock"]], on="year", how="left")
                st.line_chart(df_plot.set_index("year"))
            else:
                st.info("Need at least 4 data points to fit.")

with colB:
    st.subheader("Fit logistic to Tunisia tech adoption")
    if tn_tech_file is not None:
        dfT = pd.read_csv(tn_tech_file)
        techs = dfT["tech"].dropna().unique().tolist()
        if techs:
            pickt = st.selectbox("Pick a tech", techs)
            subT = dfT[dfT["tech"]==pickt].dropna(subset=["year","adoption_pct"])
            if len(subT) >= 4:
                y = subT["adoption_pct"].values
                fitT = fit_logistic(subT["year"].values, y, K0=100.0, r0=0.3)
                st.write(f"**Tunisia {pickt} fit:** r={fitT.r:.3f}, t0={fitT.t0:.1f}")
                yearsT = np.arange(subT["year"].min()-2, subT["year"].max()+10)
                predT = logistic(yearsT, 100.0, fitT.r, fitT.t0)
                df_plotT = pd.DataFrame({"year": yearsT, "adoption_pct_fit": predT}).merge(
                    subT[["year","adoption_pct"]], on="year", how="left")
                st.line_chart(df_plotT.set_index("year"))
            else:
                st.info("Need at least 4 data points to fit.")

st.write("---")
st.subheader("🇹🇳 Build Tunisia EV scenarios from analogs + transfer")
col1, col2, col3 = st.columns(3)
with col1:
    start = st.number_input("Start year", 2024, 2050, 2025)
    end = st.number_input("End year", start, 2055, 2038)
with col2:
    K_tn = st.number_input("Max EV stock in Tunisia (K)", 5, 2000, 150, step=5) * 1000.0
with col3:
    r_tn = st.number_input("r (growth rate) for TN", 0.05, 1.0, 0.35, step=0.01)
    t0_tn = st.number_input("t0 (midpoint year) for TN", 2024, 2050, 2031, step=1)

yearsF = np.arange(int(start), int(end)+1)
ev_tn = logistic(yearsF, K_tn, r_tn, t0_tn)
ratio = st.slider("EVs per public charger (planning ratio)", 5, 50, 18, 1)
chargers_needed = ev_tn / ratio

dfF = pd.DataFrame({"year": yearsF, "ev_stock_tn": ev_tn.astype(int), "chargers_needed": chargers_needed.astype(int)})
st.line_chart(dfF.set_index("year"))
st.download_button("Download Tunisia forecast (CSV)", data=dfF.to_csv(index=False).encode(), file_name="tn_ev_forecast.csv")
