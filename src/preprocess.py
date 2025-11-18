# src/preprocess.py
"""
Preprocessing utilities for Hyderabad House Price Estimator.

Functions:
- load_data(path) -> pd.DataFrame
- prepare_data(df, log_transform=True) -> (X, y, df_final)
- get_location_options(df) -> list of locations (for UI)
"""

import pandas as pd
import numpy as np

PREMIUM_LOCATIONS = ["Banjara Hills", "Jubilee Hills", "Gachibowli", "Hitech City", "Kokapet"]

def load_data(path="../data/hyderabad.csv"):
    """Load cleaned dataset (default path points to data/hyderabad.csv)."""
    df = pd.read_csv(path)
    return df

def normalize_binary(series):
    """Return series with 'Yes'/'No' normalized."""
    return series.astype(str).str.strip().str.title().replace({"1":"Yes","0":"No","True":"Yes","False":"No"})

def prepare_data(df, log_transform=True, top_k_locations=10):
    """
    Prepare dataset for modeling.
    - ensures required columns exist
    - normalizes binary cols
    - creates house_type and gated_community if missing
    - groups locations keeping top_k and mapping others to 'Other'
    - optionally applies log1p to Price and returns X,y
    """

    df = df.copy()

    # Ensure required columns are present
    required = ["Area","No. of Bedrooms","CarParking","LiftAvailable","Resale","Location","Price"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # Normalize binary fields
    for col in ["CarParking","LiftAvailable","Resale"]:
        df[col] = normalize_binary(df[col])

    # Create house_type if missing
    if "house_type" not in df.columns:
        df["house_type"] = df.apply(
            lambda r: "Individual" if (r["Area"]>2000 or r["Location"] in PREMIUM_LOCATIONS) else "Apartment",
            axis=1
        )

    # Create gated_community if missing
    if "gated_community" not in df.columns:
        df["gated_community"] = df.apply(
            lambda r: "Yes" if (r["LiftAvailable"]=="Yes" and r["Area"]>1200) else "No",
            axis=1
        )

    # Keep top_k locations
    top_locations = df["Location"].value_counts().nlargest(top_k_locations).index
    df["Location"] = df["Location"].apply(lambda x: x if x in top_locations else "Other")

    # Final features
    categorical_cols = ["CarParking","LiftAvailable","Resale","Location","house_type","gated_community"]
    numeric_cols = ["Area","No. of Bedrooms"]

    # Optionally transform target (log)
    if log_transform:
        df["Price"] = np.log1p(df["Price"])

    X = df[categorical_cols + numeric_cols]
    y = df["Price"]

    return X, y, df

def get_location_options(df, top_k=10):
    """Return list of top locations + 'Other' for UI dropdown."""
    top = df["Location"].value_counts().nlargest(top_k).index.tolist()
    if "Other" not in top:
        top = top + ["Other"]
    return top

