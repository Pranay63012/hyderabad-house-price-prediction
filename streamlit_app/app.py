# streamlit_app/app.py
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# PROJECT ROOT & PATH SETUP
# ------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hyderabad.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

sys.path.append(SRC_PATH)

from preprocess import load_data
from utils import prepare_input_df, inverse_log_transform, format_inr

# ------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Hyderabad House Price Estimator",
    layout="wide"
)

st.title("🏠 Hyderabad House Price Estimator")
st.write("Enter property details to get market-based house price estimation.")

# ------------------------------
# LOAD DATA & MODEL
# ------------------------------
@st.cache_data
def load_df():
    return load_data(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_df()
model = load_model()

# Dropdown options
locations = sorted(df["Location"].unique().tolist())
house_types = sorted(df["house_type"].unique().tolist())
yn = ["Yes", "No"]

# ------------------------------
# SIDEBAR INPUTS
# ------------------------------
st.sidebar.header("Enter Property Details")

area = st.sidebar.number_input(
    "Area (sqft)",
    min_value=200.0,
    max_value=20000.0,
    value=1200.0,
    step=50.0
)

bedrooms = st.sidebar.number_input(
    "No. of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

location = st.sidebar.selectbox("Location", locations)
house_type = st.sidebar.selectbox("House Type", house_types)
parking = st.sidebar.selectbox("Car Parking", yn)
lift = st.sidebar.selectbox("Lift Available", yn)
resale = st.sidebar.selectbox("Resale", yn)
gated = st.sidebar.selectbox("Gated Community", yn)

predict_btn = st.sidebar.button("Estimate Price")

# ------------------------------
# PREDICTION
# ------------------------------
if predict_btn:
    try:
        input_df = prepare_input_df(
            area, bedrooms, parking, lift, resale,
            location, house_type, gated
        )
    except Exception as e:
        st.error(f"Input formatting error: {e}")
        st.stop()

    try:
        pred_log = model.predict(input_df)[0]
        pred_price = inverse_log_transform(pred_log)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Prediction Successful!")

    st.markdown(
        f"""
        ### 🏷️ Estimated House Price  
        **{format_inr(pred_price)}**

        #### Price Range  
        {format_inr(pred_price * 0.85)} — {format_inr(pred_price * 1.15)}
        """
    )

    st.write("### Input Summary")
    st.table(input_df.T)

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Developed by **Pranay Rachakonda** – Hyderabad House Price Estimator")
