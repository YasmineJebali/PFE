from pathlib import Path
import streamlit as st

# Keep a single source of truth for ordering & labels.
NAV = [
    ("pages/app.py",               "🏠 Home"),
    ("pages/0_Smoke_Test.py",         "🧪 Smoke Test"),
    ("pages/1_Map_&_Gaps.py",         "🗺️ Map & Gaps"),        # <-- put your real filename here
    ("pages/2_Analogs_and_ML.py",     "📊 Analogs & ML"),
    ("pages/3_Modeling_Lab.py",       "🧪 Modeling Lab"),
    ("pages/4_ROI_Sensitivity.py",    "💸 ROI & Sensitivity"),
    ("pages/5_Deployment_ROI.py",     "🧭 Deployment ROI"),
    ("pages/6_Chatbot.py",            "💬 Chatbot"),
]

def set_page(title, icon=None):
    st.set_page_config(page_title=title, layout="wide")
    st.title(f"{icon or ''} {title}".strip())

def sidebar_nav(_current_file):
    st.sidebar.markdown("### Navigation")
    for file, label in NAV:
        if Path(file).exists():      # only link if file is really there
            st.sidebar.page_link(file, label=label)

def status_pills():
    pass  # your existing implementation or leave as-is
