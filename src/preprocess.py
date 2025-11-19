# src/preprocess.py
import pandas as pd

def load_data(path: str):
    """Loads the cleaned dataset used for dropdowns and reference."""
    df = pd.read_csv(path)
    return df

def yesno_to_int(val):
    """Convert 'Yes'/'No' to 1/0."""
    if isinstance(val, str):
        return 1 if val.lower() == "yes" else 0
    return int(val)

def prepare_data(df: pd.DataFrame):
    """Ensures required types before prediction/training."""
    binary_cols = ["CarParking", "LiftAvailable", "Resale", "gated_community"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    df["Area"] = df["Area"].astype(float)
    df["No. of Bedrooms"] = df["No. of Bedrooms"].astype(int)
    df["Price"] = df["Price"].astype(float)
    return df
