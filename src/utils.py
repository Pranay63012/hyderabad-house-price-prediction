# src/utils.py
"""
Utility functions used by Streamlit app and scripts.
"""

import numpy as np

def inverse_log_transform(y_log):
    """Inverse of log1p -> returns original scale (float)."""
    return np.expm1(y_log)

def format_inr(amount):
    """
    Format number as Indian-style currency string.
    Example: 1250000 -> '₹ 12,50,000'
    """
    try:
        val = int(round(amount))
    except:
        val = amount
    s = f"{val:,}"
    return f"₹ {s}"

def prepare_input_df(area, bedrooms, parking, lift, resale, location, house_type, gated):
    """
    Build a single-row DataFrame from user inputs in the same order
    as the training X.
    """
    import pandas as pd
    row = {
        "CarParking": [parking],
        "LiftAvailable": [lift],
        "Resale": [resale],
        "Location": [location],
        "house_type": [house_type],
        "gated_community": [gated],
        "Area": [float(area)],
        "No. of Bedrooms": [int(bedrooms)]
    }
    return pd.DataFrame(row)

