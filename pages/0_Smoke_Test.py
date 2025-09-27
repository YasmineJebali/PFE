import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smoke Test: Data Intake", layout="wide")
st.title("🧪 Smoke Test — Data Intake & Validation")

stations = st.file_uploader("Upload stations CSV", type=["csv"])
forecast = st.file_uploader("Upload Tunisia forecast CSV", type=["csv"])

if stations:
    df = pd.read_csv(stations)
    st.write("Stations preview", df.head())

if forecast:
    df = pd.read_csv(forecast)
    st.write("Forecast preview", df.head())

if stations and forecast:
    st.success("✅ Both files loaded successfully! You can go to Analogs & ML page now.")
