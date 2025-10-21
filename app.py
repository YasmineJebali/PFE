# app.py  —  Home / Launcher
import streamlit as st
from utils.ui import set_page, sidebar_nav, status_pills

set_page("Home", icon="🏠")
# This makes your custom sidebar/nav render; it’s safe to keep
sidebar_nav(__file__)
status_pills()

st.markdown("""
**Welcome to the Agil EV MVP.**  
Follow the flow in the sidebar:

1. **Smoke Test** – load data and sanity-check schemas  
2. **Map & Gaps** – where chargers and Agil stations are; underserved areas  
3. **Analogs & ML** – fit analog logistic + build Tunisia scenario  
4. **Modeling Lab** – compare models with rolling-origin CV; pick the **best**  
5. **ROI & Sensitivity** – NPV, payback, IRR (per-charger & portfolio)  
6. **Deployment ROI** – translate forecast into rollout plan  
7. **Chatbot** – RAG assistant with your project notes  
""")

# Quick links (optional, super handy from Home)
st.page_link("pages/0_Smoke_Test.py",       label="Start → Smoke Test", icon="🧪")
st.page_link("pages/1_Map_and_Gaps.py",     label="Next → Map & Gaps",  icon="🗺️")
st.page_link("pages/2_Analogs_and_ML.py",   label="Next → Analogs & ML", icon="📊")
st.page_link("pages/3_Modeling_Lab.py",     label="Next → Modeling Lab", icon="🔬")
st.page_link("pages/4_ROI_Sensitivity.py",  label="Next → ROI & Sensitivity", icon="💸")
st.page_link("pages/5_Deployment_ROI.py",   label="Next → Deployment ROI", icon="🧭")
st.page_link("pages/6_Chatbot.py",          label="Ask the Assistant", icon="💬")
