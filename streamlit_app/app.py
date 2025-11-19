# streamlit_app/app.py
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# PATH SETUP
# ------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hyderabad.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

sys.path.append(SRC_PATH)

from preprocess import load_data
from utils import prepare_input_df, inverse_log_transform, format_inr

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="HYDERABAD HOUSE PRICE ESTIMATOR",
    layout="wide",
)

# ------------------------------
# CUSTOM THEMING (LIGHT RED + BLACK)
# ------------------------------
st.markdown(
    """
    <style>
        body {
            background-color: #0e0e0e;
            color: #ffffff;
        }

        .title-text {
            font-size: 48px;
            font-weight: 900;
            text-align: center;
            color: #ff6b6b;
            letter-spacing: 2px;
            margin-bottom: -10px;
        }

        .sub-text {
            text-align: center;
            font-size: 20px;
            color: #cccccc;
        }

        .price-box {
            background: #1d1d1d;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #ff6b6b;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .price-text {
            font-size: 36px;
            font-weight: 800;
            color: #ff6b6b;
        }

        .range-text {
            font-size: 18px;
            color: #ffffff;
        }

        .section-header {
            font-size: 28px;
            color: #ff6b6b;
            font-weight: 900;
            margin-top: 20px;
            letter-spacing: 1px;
        }
        
        .stButton button {
            background-color: #ff6b6b !important;
            color: black !important;
            font-weight: bold !important;
            border-radius: 6px !important;
            height: 45px !important;
            border: none !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

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

locations = sorted(df["Location"].unique().tolist())
house_types = sorted(df["house_type"].unique().tolist())
yn = ["Yes", "No"]

# ------------------------------
# TITLE
# ------------------------------
st.markdown('<div class="title-text">HYDERABAD HOUSE PRICE ESTIMATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Market-based house price estimation for Hyderabad real estate</div>', unsafe_allow_html=True)

st.write("")  # spacing

# ------------------------------
# SIDEBAR INPUTS
# ------------------------------
st.sidebar.header("ENTER PROPERTY DETAILS")

area = st.sidebar.number_input("Area (sqft)", min_value=200.0, max_value=20000.0, value=1200.0, step=50.0)
bedrooms = st.sidebar.number_input("No. of Bedrooms", min_value=1, max_value=10, value=2)
location = st.sidebar.selectbox("Location", locations)
house_type = st.sidebar.selectbox("House Type", house_types)
parking = st.sidebar.selectbox("Car Parking", yn)
lift = st.sidebar.selectbox("Lift Available", yn)
resale = st.sidebar.selectbox("Resale", yn)
gated = st.sidebar.selectbox("Gated Community", yn)

predict_btn = st.sidebar.button("Estimate Price")

# ------------------------------
# PRICE CONVERSION — CRORE FORMAT
# ------------------------------
def convert_to_crore(amount):
    if amount >= 1e7:
        return f"{amount/1e7:.2f} Crore"
    elif amount >= 1e5:
        return f"{amount/1e5:.2f} Lakh"
    else:
        return f"{amount:,.0f}"

# ------------------------------
# PREDICTION
# ------------------------------
if predict_btn:
    input_df = prepare_input_df(area, bedrooms, parking, lift, resale, location, house_type, gated)
    pred_log = model.predict(input_df)[0]
    pred_price = inverse_log_transform(pred_log)

    lower = pred_price * 0.85
    upper = pred_price * 1.15

    st.markdown('<div class="price-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="price-text">₹ {pred_price:,.0f}  ({convert_to_crore(pred_price)})</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="range-text">Estimated Range: ₹{lower:,.0f} ({convert_to_crore(lower)}) — ₹{upper:,.0f} ({convert_to_crore(upper)})</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # INPUT SUMMARY
    st.markdown('<div class="section-header">PROPERTY SUMMARY</div>', unsafe_allow_html=True)
    st.table(input_df.T)

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Developed by **Pranay Rachakonda**")
