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

# -----------------------------------------------------
# PREMIUM UI STYLING
# -----------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif !important;
        }

        .main-title {
            font-size: 52px;
            font-weight: 900;
            background: linear-gradient(90deg, #00e5ff, #00ff95, #4dff4d);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            animation: glow 2.5s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px #00ffc6; }
            to   { text-shadow: 0 0 25px #00ffd5; }
        }

        .subtitle {
            font-size: 22px;
            text-align: center;
            color: #cccccc;
            margin-top: -10px;
            margin-bottom: 25px;
        }

        .prediction-box {
            background-color: #0f3d29;
            padding: 18px;
            border-radius: 12px;
            color: white;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
            margin-top: 20px;
        }

        .price-box {
            background: linear-gradient(135deg, #003300, #00994d);
            padding: 20px;
            border-radius: 14px;
            color: #00ffcc;
            font-size: 36px;
            font-weight: 800;
            text-align: center;
            border: 2px solid #00ffcc;
        }

        .range-box {
            background-color: #111;
            padding: 15px;
            border-radius: 10px;
            color: #00ffaa;
            font-size: 22px;
            text-align: center;
            margin-top: 10px;
        }

        .stButton>button {
            background: linear-gradient(90deg, #0099ff, #00ff99);
            color: black;
            font-size: 18px;
            font-weight: 700;
            padding: 8px 20px;
            border-radius: 8px;
            border: none;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #00ffbb, #00e1ff);
            cursor: pointer;
            box-shadow: 0px 0px 10px #00ffee;
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
st.markdown('<div class="main-title">Hyderabad House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered estimation based on Hyderabad real estate trends</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR INPUTS
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

    # Success notice
    st.markdown('<div class="prediction-box">Prediction Successful!</div>', unsafe_allow_html=True)

    # Price
    st.markdown(f'<div class="price-box">{format_inr(pred_price)}</div>', unsafe_allow_html=True)

    # Range
    min_p = pred_price * 0.85
    max_p = pred_price * 1.15
    st.markdown(
        f'<div class="range-box">Estimated Range: {format_inr(min_p)} – {format_inr(max_p)}</div>',
        unsafe_allow_html=True,
    )

    st.write("### 📘 Input Summary")
    st.dataframe(input_df.T, use_container_width=True)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.write("---")
st.caption("Developed by **Pranay Rachakonda** · © 2025 Hyderabad Real Estate AI")
