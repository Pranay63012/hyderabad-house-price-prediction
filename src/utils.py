# src/utils.py
import pandas as pd
import numpy as np

def yesno_to_int(value):
    """Convert Yes/No to 1/0."""
    if isinstance(value, str):
        return 1 if value.lower() == "yes" else 0
    return value

def prepare_input_df(area, bedrooms, parking, lift, resale, location, house_type, gated):
    """Prepare a single-row DF identical to model training structure."""
    
    row = {
        "Area": float(area),
        "No. of Bedrooms": int(bedrooms),
        "CarParking": yesno_to_int(parking),
        "LiftAvailable": yesno_to_int(lift),
        "Resale": yesno_to_int(resale),
        "Location": location,
        "house_type": house_type,
        "gated_community": yesno_to_int(gated)
    }
    
    return pd.DataFrame([row])

def inverse_log_transform(log_val):
    """Inverse of log(price) = exp(value)."""
    return float(np.exp(log_val))

def format_inr(amount):
    """Format INR nicely."""
    amount = float(amount)
    return "₹{:,.0f}".format(amount)
