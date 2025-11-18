# streamlit_app/app.py
"""
Upgraded UI — Hyderabad House Price Estimator
Compatible with Streamlit Cloud deployment.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# =============================
# 1) FIX PATHS FOR CLOUD
# =============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hyderabad.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

sys.path.append(SRC_PATH)

# Import utils and preprocess (CLOUD-SAFE)
from utils import prepare_input_df, inverse_log_transform, format_inr
from preprocess import load_data

# =============================
# 2) STREAMLIT PAGE SETUP
# =============================
st.set_page_config(page_title="Hyderabad House Price Estimator",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .big-price {
        font-size:32px;
        font-weight:700;
    }
    .subtle {
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# 3) HEADER
# =============================
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🏠 Hyderabad House Price Estimator")
    st.write("Get an instant market-based house price estimate.")

with col2:
    st.image("https://img.icons8.com/fluency/48/000000/house.png")

# =============================
# 4) LOAD DATA + MODEL
# =============================
@st.cache_data
def load_cleaned_df():
    return load_data(DATA_PATH)

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

df = load_cleaned_df()
model = load_model()

locations = sorted(df["Location"].unique().tolist())
house_types = sorted(df["house_type"].unique().tolist())
binary = ["Yes", "No"]

# =============================
# 5) SIDEBAR INPUTS
# =============================
st.sidebar.header("Property Details")

area = st.sidebar.number_input("Area (sqft)", 200.0, 10000.0, 1200.0, 50.0)
bedrooms = st.sidebar.number_input("No. of Bedrooms", 1, 10, 2)
location = st.sidebar.selectbox("Location", locations)
house_type = st.sidebar.selectbox("House Type", house_types)
parking = st.sidebar.selectbox("Car Parking", binary)
lift = st.sidebar.selectbox("Lift Available", binary)
resale = st.sidebar.selectbox("Resale", binary)
gated = st.sidebar.selectbox("Gated Community", binary)

predict_btn = st.sidebar.button("Estimate Price")

# =============================
# 6) INFO PANELS
# =============================
left, right = st.columns([2, 1])

with left:
    st.subheader("Model Overview")
    st.write("""
    This app uses a Stacked Ensemble (RandomForest + XGBoost) trained on 
    Hyderabad real-estate listings. Prices are log-transformed internally.
    """)

    st.markdown("### Data Sample")
    st.dataframe(df.sample(6), use_container_width=True)

with right:
    st.markdown("### Quick Stats")
    avg = inverse_log_transform(df["Price"].mean())
    med = inverse_log_transform(df["Price"].median())
    st.metric("Average Price", format_inr(avg))
    st.metric("Median Price", format_inr(med))

# =============================
# 7) PREDICTION SECTION
# =============================
if predict_btn:
    if model is None:
        st.error("Model not available!")
    else:
        # Build input DF
        input_df = prepare_input_df(area, bedrooms, parking, lift, resale, location, house_type, gated)

        # ========== STREAMLIT CLOUD FIX: Force CPU predictor ==========
        try:
            if hasattr(model, "named_steps"):
                if "ensemble" in model.named_steps:
                    ens = model.named_steps["ensemble"]
                    for name, est in ens.named_estimators_.items():
                        try:
                            est.set_params(predictor="cpu_predictor")
                        except:
                            pass
        except:
            pass

        try:
            pred_log = model.predict(input_df)[0]
            pred_price = inverse_log_transform(pred_log)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

        st.balloons()

        # RESULT CARD
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class='subtle'>Estimated Price</div>
                    <div class='big-price'>{format_inr(pred_price)}</div>
                </div>
                <div style="text-align:right;">
                    <div class='subtle'>Estimated Range</div>
                    <div style="font-weight:600">{format_inr(pred_price*0.85)} — {format_inr(pred_price*1.15)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Input Summary
        st.write("#### Input Summary")
        st.table(input_df.T)

        # Feature Importance (if available)
        st.write("#### Feature Importance (approx.)")
        try:
            pre = model.named_steps["preprocessor"]
            cat = pre.named_transformers_["cat"]
            cat_names = cat.get_feature_names_out(
                ["CarParking", "LiftAvailable", "Resale", "Location", "house_type", "gated_community"]
            )
            num = np.array(["Area", "No. of Bedrooms"])
            feat = np.concatenate([cat_names, num])

            ensemble = model.named_steps["ensemble"]
            importances = []

            for nm, est in ensemble.named_estimators_.items():
                if hasattr(est, "feature_importances_"):
                    importances.append(est.feature_importances_)

            if len(importances):
                avg_imp = np.mean(importances, axis=0)
                st.bar_chart(pd.Series(avg_imp, index=feat).sort_values(ascending=False).head(12))
            else:
                st.info("Feature importances unavailable.")
        except Exception:
            st.info("Feature importance could not be computed.")

# =============================
# 8) FOOTER
# =============================
st.write("---")
c1, c2 = st.columns([3,1])
with c1:
    st.write("**Developed by Pranay Rachakonda** — AIML | Hyderabad")
    st.write("Hyderabad House Price Estimator — Machine Learning Project")

