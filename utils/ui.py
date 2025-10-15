# utils/ui.py — shared UI elements for navigation & layout
from pathlib import Path
import streamlit as st

APP_STEPS = [
    ("🧪 1) Data Intake (Smoke Test)",   "pages/0_Smoke_Test.py"),
    ("🗺️ 2) Map & Gaps",                 "app.py"),
    ("📊 3) Analogs & ML",               "pages/2_Analogs_and_ML.py"),
    ("🧪 3b) Modeling Lab (optional)",   "pages/3_Modeling_Lab.py"),   # only if you added it
    ("💸 4) ROI & Sensitivity",          "pages/4_ROI_Sensitivity.py"),
    ("🧭 5) Deployment ROI",             "pages/5_Deployment_ROI.py"),
    ("💬 Assistant",                     "pages/6_Chatbot.py"),
]

def set_page(title: str, icon: str | None = None, wide: bool = True):
    st.set_page_config(page_title=title, layout="wide", page_icon=icon)

def breadcrumbs(trail: list[tuple[str, str]]):
    """trail: list of (label, path). The last item is the current page."""
    parts = []
    for i, (label, path) in enumerate(trail):
        if i < len(trail) - 1:
            parts.append(st.page_link(path, label=label))
        else:
            st.caption(" → ".join([t[0] for t in trail]))
    # Put Assistant link at top right for convenience
    st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")

def sidebar_nav(current_file: str):
    """Highlight current step and keep links discoverable."""
    st.sidebar.header("Navigation")
    here = Path(current_file).name
    for label, path in APP_STEPS:
        target = Path(path).name
        if target == here:
            st.sidebar.markdown(f"**{label}**")
        else:
            st.sidebar.page_link(path, label=label)

def status_pills():
    """Small status pills for session data presence."""
    ok_st = isinstance(st.session_state.get("stations_df"), type(getattr(st.session_state, 'stations_df', None))) \
            and st.session_state.get("stations_df") is not None \
            and not getattr(st.session_state["stations_df"], "empty", True)
    ok_fc = isinstance(st.session_state.get("forecast_df"), type(getattr(st.session_state, 'forecast_df', None))) \
            and st.session_state.get("forecast_df") is not None \
            and not getattr(st.session_state["forecast_df"], "empty", True)

    st.sidebar.write("---")
    st.sidebar.caption("Session status")
    st.sidebar.write(f"Stations: {'✅' if ok_st else '❌'}")
    st.sidebar.write(f"Forecast: {'✅' if ok_fc else '❌'}")
