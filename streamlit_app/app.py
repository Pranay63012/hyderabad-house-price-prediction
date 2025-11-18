# streamlit_app/app.py
"""
Upgraded UI — Hyderabad House Price Estimator
Replace your existing file with this.
Run:
    streamlit run streamlit_app/app.py
"""

import os
import sys
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# -------------------------
# 1) PROJECT ABSOLUTE PATH
# -------------------------
PROJECT_ROOT = r"C:/Users/prana/hyderabad_house_price_prediction"
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hyderabad.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

sys.path.append(SRC_PATH)

# -------------------------
# 2) IMPORT UTIL & PREPROCESS (manual)
# -------------------------
# utils
utils_file = os.path.join(SRC_PATH, "utils.py")
spec_utils = importlib.util.spec_from_file_location("utils", utils_file)
utils = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(utils)
prepare_input_df = utils.prepare_input_df
inverse_log_transform = utils.inverse_log_transform
format_inr = utils.format_inr

# preprocess
pre_file = os.path.join(SRC_PATH, "preprocess.py")
spec_pre = importlib.util.spec_from_file_location("preprocess", pre_file)
preprocess = importlib.util.module_from_spec(spec_pre)
spec_pre.loader.exec_module(preprocess)
load_data = preprocess.load_data

# -------------------------
# 3) STREAMLIT PAGE SETUP & THEME-LIKE STYLING
# -------------------------
st.set_page_config(page_title="Hyderabad House Price Estimator",
                   layout="wide",
                   initial_sidebar_state="expanded")

# small CSS for cards
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

# Header
col1, col2 = st.columns([4,1])
with col1:
    st.title("🏠 Hyderabad House Price Estimator")
    st.write("Get a market-based estimate instantly. Trained on historical Hyderabad listings.")
with col2:
    st.image("https://img.icons8.com/fluency/48/000000/house.png")  # small icon (internet required)

# -------------------------
# 4) LOAD DATA AND MODEL
# -------------------------
@st.cache_data
def load_cleaned_df(path=DATA_PATH):
    return load_data(path)

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model at {path}\nError: {e}")
        return None

df = load_cleaned_df()
model = load_model()

# Prepare dropdown options
locations = sorted(df["Location"].unique().tolist())
house_types = sorted(df["house_type"].unique().tolist())
binary = ["Yes", "No"]

# -------------------------
# 5) SIDEBAR (Inputs)
# -------------------------
st.sidebar.header("Input Property Details")
area = st.sidebar.number_input("Area (sqft)", min_value=200.0, max_value=10000.0, value=1200.0, step=50.0)
bedrooms = st.sidebar.number_input("No. of Bedrooms", min_value=1, max_value=10, value=2, step=1)
location = st.sidebar.selectbox("Location", locations, index=0)
house_type = st.sidebar.selectbox("House Type", house_types, index=0)
parking = st.sidebar.selectbox("Car Parking", binary, index=0)
lift = st.sidebar.selectbox("Lift Available", binary, index=0)
resale = st.sidebar.selectbox("Resale", binary, index=1)
gated = st.sidebar.selectbox("Gated Community", binary, index=0)

predict_btn = st.sidebar.button("Estimate Price")

# -------------------------
# 6) MAIN AREA: Show EDA/Info + Predict Card
# -------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("Model Overview")
    st.write(
        """
        This app uses a Stacked Ensemble (RandomForest + XGBoost) trained on Hyderabad real-estate data.
        Price target is log-transformed during training — predictions are shown in INR (₹).
        """
    )

    st.markdown("### Data snapshot")
    st.dataframe(df.sample(6), use_container_width=True)

with right:
    st.markdown("### Quick Stats")
    avg_price = inverse_log_transform(df["Price"].mean())
    median_price = inverse_log_transform(df["Price"].median())
    st.metric("Average price (dataset)", format_inr(avg_price))
    st.metric("Median price (dataset)", format_inr(median_price))

# -------------------------
# 7) PREDICTION & CPU PATCH FOR XGBOOST
# -------------------------
if predict_btn:
    if model is None:
        st.error("Model not loaded — check models/best_model.pkl")
    else:
        # build input
        input_df = prepare_input_df(area, bedrooms, parking, lift, resale, location, house_type, gated)

        # patch xgboost inside pipeline/stacking to avoid gpu_id error
        try:
            ensemble = model.named_steps.get("ensemble", None)
            if ensemble is None:
                # maybe pipeline step name differs; search for XGB instances
                for name, step in getattr(model, "named_steps", {}).items():
                    if "xgb" in name.lower():
                        try:
                            step.set_params(predictor="cpu_predictor")
                        except:
                            pass
            else:
                # patch each named estimator inside stacking
                for nm, est in getattr(ensemble, "named_estimators_", {}).items():
                    try:
                        if hasattr(est, "get_xgb_params"):
                            p = est.get_xgb_params()
                            p["predictor"] = "cpu_predictor"
                    except:
                        pass
        except Exception:
            pass

        # predict
        try:
            pred_log = model.predict(input_df)[0]
            pred_price = inverse_log_transform(pred_log)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

        # Show animated success
        st.balloons()

        # Result card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style='display:flex; justify-content:space-between; align-items:center'>", unsafe_allow_html=True)
        st.markdown(f"<div><div class='subtle'>Estimated Price</div><div class='big-price'>{format_inr(pred_price)}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:right'><div class='subtle'>Estimated range</div><div style='font-weight:600'>{format_inr(pred_price*0.85)} — {format_inr(pred_price*1.15)}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input summary
        st.write("#### Input Summary")
        st.table(input_df.T)

        # -------------------------
        # 8) Feature importance (approx): average of estimators that expose importances
        # -------------------------
        st.write("#### Feature importance (approx)")
        try:
            pre = model.named_steps["preprocessor"]
            # get feature names
            cat = pre.named_transformers_["cat"]
            cat_names = cat.get_feature_names_out(["CarParking","LiftAvailable","Resale","Location","house_type","gated_community"])
            numeric = np.array(["Area","No. of Bedrooms"])
            feature_names = np.concatenate([cat_names, numeric])

            # gather importances from estimators in stacking
            ensemble = model.named_steps["ensemble"]
            importances_list = []
            for nm, est in getattr(ensemble, "named_estimators_", {}).items():
                if hasattr(est, "feature_importances_"):
                    importances_list.append(est.feature_importances_)
            if len(importances_list) == 0:
                raise ValueError("No estimator exposes feature_importances_")
            avg_imp = np.mean(importances_list, axis=0)
            imp_ser = pd.Series(avg_imp, index=feature_names).sort_values(ascending=False).head(12)
            st.bar_chart(imp_ser)
        except Exception as e:
            st.info("Feature importance not available (pipeline shape may differ).")
            # print exception on debug mode
            # st.write(e)

# -------------------------
# 9) FOOTER
# -------------------------
st.write("---")
footer_col1, footer_col2 = st.columns([3,1])
with footer_col1:
    st.write("**Developed by Pranay Rachakonda** — AIML | Hyderabad")
    st.write("Project: Hyderabad House Price Estimator — data-driven predictions.")
with footer_col2:
    st.write("")

# End of file

