# streamlit_app/app.py

import os
import sys
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------
# PATHS
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
    page_icon="🏠",
    layout="wide",
)

# -----------------------------------------------------
# PREMIUM STYLING
# -----------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;900&family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

.main-header {
    font-size: 60px;
    font-weight: 900;
    background: linear-gradient(90deg, #00f7ff, #00ffa0, #00ffea, #68ffd2);
    -webkit-background-clip: text;
    color: transparent;
    animation: shine 4s infinite linear;
    text-align: center;
    margin-bottom: -10px;
}

@keyframes shine {
  from { filter: brightness(1); }
  to { filter: brightness(1.35); }
}

.sub-header {
    font-size: 23px;
    font-weight: 400;
    color: #dddddd;
    text-align: center;
    margin-bottom: 40px;
}

.card {
    background: rgba(20,20,20,0.6);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    box-shadow: 0px 4px 18px rgba(0,0,0,0.25);
}

.price-box {
    background: linear-gradient(135deg, #004d40, #007f5f);
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    color: #00ffcc;
    border: 2px solid #00ffcc;
    margin-bottom: 10px;
}

.range-box {
    background: #111;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    color: #8affd4;
}

.stButton>button {
    background: linear-gradient(90deg, #00bbff, #00ffaa);
    color: #000;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 10px;
    transition: 0.2s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.04);
    box-shadow: 0px 0px 12px #00ffe1;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #00ffd0;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="main-header">Hyderabad House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate AI-powered price estimation for Hyderabad real estate</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Enter Property Details</div>', unsafe_allow_html=True)

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
# PREDICTION
# -----------------------------------------------------
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

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="price-box">{}</div>'.format(format_inr(pred_price)), unsafe_allow_html=True)

    min_p = pred_price * 0.85
    max_p = pred_price * 1.15

    st.markdown(
        f'<div class="range-box">Estimated Range: {format_inr(min_p)} — {format_inr(max_p)}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("### Property Summary")
    st.dataframe(input_df.T, use_container_width=True)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.write("---")
st.caption("Developed by **Pranay Rachakonda** · © 2025 Hyderabad Real Estate AI")

