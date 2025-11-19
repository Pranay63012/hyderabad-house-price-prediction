# streamlit_app/app.py

import os
import sys
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------
# PATH SETUP
# -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hyderabad.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

sys.path.append(SRC_PATH)

from preprocess import load_data
from utils import prepare_input_df, inverse_log_transform, format_inr

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Hyderabad House Price Estimator",
    layout="wide",
    page_icon="🏠",
)

# Custom UI Styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 45px;
            font-weight: 700;
            color: #ffffff;
        }
        .subtitle {
            font-size: 22px;
            color: #d0d0d0;
            margin-bottom: 20px;
        }
        .prediction-box {
            background-color: #133825;
            padding: 18px;
            border-radius: 10px;
            color: white;
            font-size: 20px;
            font-weight: 600;
            text-align: center;
        }
        .price-box {
            background-color: #111;
            padding: 15px;
            border-radius: 10px;
            color: #00e676;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            border: 1px solid #00e676;
        }
        .range-box {
            background-color: #191919;
            padding: 12px;
            border-radius: 8px;
            color: #00bfa5;
            font-size: 20px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------
# LOAD DATA & MODEL
# -----------------------------------------------------
@st.cache_data
def load_df():
    return load_data(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_df()
model = load_model()

locations = sorted(df["Location"].unique().tolist())
house_types = sorted(df["house_type"].unique().tolist())
yn = ["Yes", "No"]

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
st.markdown('<div class="main-title">🏠 Hyderabad House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter property details below to get the estimated market value.</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR INPUT FORM
# -----------------------------------------------------
with st.sidebar:
    st.header("Property Details")

    area = st.number_input("Area (sqft)", 200.0, 20000.0, 1200.0, step=50.0)
    bedrooms = st.number_input("No. of Bedrooms", 1, 10, 2)
    location = st.selectbox("Location", locations)
    house_type = st.selectbox("House Type", house_types)
    parking = st.selectbox("Car Parking", yn)
    lift = st.selectbox("Lift Available", yn)
    resale = st.selectbox("Resale", yn)
    gated = st.selectbox("Gated Community", yn)

    predict_btn = st.button("Estimate Price", use_container_width=True)

# -----------------------------------------------------
# PREDICTION SECTION
# -----------------------------------------------------
if predict_btn:

    # Format inputs
    try:
        input_df = prepare_input_df(
            area, bedrooms, parking, lift, resale,
            location, house_type, gated
        )
    except Exception as e:
        st.error(f"Input formatting error: {e}")
        st.stop()

    # Run prediction
    try:
        pred_log = model.predict(input_df)[0]
        pred_price = inverse_log_transform(pred_log)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # SUCCESS MESSAGE
    st.markdown('<div class="prediction-box">Prediction Successful!</div>', unsafe_allow_html=True)
    st.write("")

    # Estimated price
    st.markdown(f'<div class="price-box">{format_inr(pred_price)}</div>', unsafe_allow_html=True)
    st.write("")

    # Range
    min_price = pred_price * 0.85
    max_price = pred_price * 1.15

    st.markdown(
        f'<div class="range-box">Price Range: {format_inr(min_price)} — {format_inr(max_price)}</div>',
        unsafe_allow_html=True,
    )

    st.write("## 📘 Input Summary")
    st.dataframe(input_df.T, use_container_width=True)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.write("---")
st.caption("Developed by **Pranay Rachakonda** · Hyderabad Real Estate Estimator © 2025")
