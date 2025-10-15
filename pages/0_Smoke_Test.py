import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smoke Test: Data Intake", layout="wide")
st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")
st.title("🧪 Smoke Test — Data Intake & Validation")

stations = st.file_uploader("Upload stations CSV", type=["csv"])
forecast = st.file_uploader("Upload Tunisia forecast CSV", type=["csv"])

if stations:
    df = pd.read_csv(stations)
    st.write("Stations preview", df.head())
    st.session_state["stations_df"] = df.copy()

if forecast:
    df = pd.read_csv(forecast)
    st.write("Forecast preview", df.head())
    st.session_state["forecast_df"] = df.copy()

if stations and forecast:
    st.success("✅ Both files loaded successfully! You can go to Analogs & ML page now.")
    st.page_link("app.py", label="Next → Map & Gaps", icon="🗺️")
