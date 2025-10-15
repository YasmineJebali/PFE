# pages/0_Home.py — Clean Home hub
from pathlib import Path
import streamlit as st
import pandas as pd

# Shared UI
from utils.ui import set_page, sidebar_nav, breadcrumbs, APP_STEPS

set_page("Home • Agil EV", icon="⚡")
sidebar_nav(__file__)
breadcrumbs([("Home", "pages/0_Home.py")])

st.title("⚡ Agil EV — Overview")

st.markdown("""
Welcome! This app helps **SNDP Agil** plan the **deployment of EV charging** with a clear, reproducible scenario:

**Flow**  
1) Data Intake → 2) Map & Gaps → 3) Analogs & ML → 4) ROI → 5) Deployment ROI
""")

# Show status overview using session
ok_st = isinstance(st.session_state.get("stations_df"), pd.DataFrame) and not st.session_state["stations_df"].empty
ok_fc = isinstance(st.session_state.get("forecast_df"), pd.DataFrame) and not st.session_state["forecast_df"].empty
st.write(f"• Stations loaded: {'✅' if ok_st else '❌'}  • Forecast loaded: {'✅' if ok_fc else '❌'}")

st.write("---")
st.subheader("Open a step")
for label, rel in APP_STEPS:
    st.page_link(rel, label=label)

st.write("---")
st.caption("Tip: Start with **🧪 Data Intake** to load/verify files, then continue in order.")
