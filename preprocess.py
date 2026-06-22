"""
==============================================================
  STEP 1 - PREPROCESSING SCRIPT
  File: preprocess.py
  Dataset: CICIDS2017
  
  What this does:
  - Loads all CICIDS2017 CSV files
  - Cleans column names (removes spaces)
  - Handles infinity and NaN values
  - Encodes attack labels
  - Normalizes features
  - Saves cleaned data and encoders
==============================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_FOLDER  = "data/"          # Place your CICIDS2017 CSVs here
OUTPUT_FOLDER = "models/"       # Saves encoders, scaler here

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# ATTACK LABEL MAPPING
# Consolidates similar attacks to cleaner names
# ─────────────────────────────────────────────
LABEL_MAP = {
    "BENIGN"                       : "BENIGN",
    "DDoS"                         : "DDoS",
    "DoS Hulk"                     : "DoS",
    "DoS GoldenEye"                : "DoS",
    "DoS slowloris"                : "DoS",
    "DoS Slowhttptest"             : "DoS",
    "Heartbleed"                   : "Heartbleed",
    "Web Attack  Brute Force"      : "Web Attack",
    "Web Attack  XSS"              : "Web Attack",
    "Web Attack  Sql Injection"    : "SQL Injection",
    "Infiltration"                 : "Infiltration",
    "Bot"                          : "Botnet",
    "PortScan"                     : "Port Scan",
    "FTP-Patator"                  : "FTP Brute Force",
    "SSH-Patator"                  : "SSH Brute Force",
}

# ─────────────────────────────────────────────
# TOP 20 FEATURES (most important from CICIDS2017)
# Reduces noise and speeds up training
# ─────────────────────────────────────────────
TOP_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Total',
    'Bwd IAT Total',
    'Fwd PSH Flags',
    'SYN Flag Count',
    'RST Flag Count',
    'Avg Fwd Segment Size',
    'Init_Win_bytes_forward',
]


def load_cicids2017(data_folder):
    """Load all CSV files from the CICIDS2017 dataset folder."""
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{data_folder}'")
        print("  Please download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  And place the CSV files inside the 'data/' folder.")
        return None

    print(f"[INFO] Found {len(csv_files)} CSV files:")
    dfs = []
    for f in csv_files:
        print(f"  Loading: {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total records loaded: {len(combined):,}")
    return combined


def clean_dataframe(df):
    """Clean and prepare the dataframe."""
    print("\n[STEP 1] Cleaning column names...")
    # CICIDS2017 has spaces before/after column names
    df.columns = df.columns.str.strip()

    print("[STEP 2] Mapping attack labels...")
    label_col = " Label" if " Label" in df.columns else "Label"
    df[label_col] = df[label_col].str.strip()
    df[label_col] = df[label_col].map(LABEL_MAP).fillna(df[label_col])
    df.rename(columns={label_col: "Label"}, inplace=True)

    print("[STEP 3] Replacing infinity values with NaN...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("[STEP 4] Dropping rows with NaN values...")
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Removed {before - len(df):,} rows with NaN. Remaining: {len(df):,}")

    return df


def select_features(df):
    """Select only top features that exist in this dataframe."""
    available = [f for f in TOP_FEATURES if f in df.columns]
    missing   = [f for f in TOP_FEATURES if f not in df.columns]

    if missing:
        print(f"[WARNING] These features not found in dataset: {missing}")
        print("  Using only available features.")

    print(f"[INFO] Using {len(available)} features for training.")
    return available


def preprocess_and_save(data_folder=DATA_FOLDER):
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("   CICIDS2017 PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_cicids2017(data_folder)
    if df is None:
        return

    # 2. Clean
    df = clean_dataframe(df)

    # 3. Show class distribution
    print("\n[INFO] Attack type distribution:")
    dist = df["Label"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label:<25} : {count:>10,}  ({pct:.2f}%)")

    # 4. Feature selection
    features = select_features(df)
    X = df[features].values.astype(np.float32)
    y = df["Label"].values

    # 5. Encode labels
    print("\n[STEP 5] Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")
    joblib.dump(le, os.path.join(OUTPUT_FOLDER, "label_encoder.pkl"))
    print(f"  Saved: models/label_encoder.pkl")

    # 6. Scale features
    print("\n[STEP 6] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "scaler.pkl"))
    joblib.dump(features, os.path.join(OUTPUT_FOLDER, "feature_names.pkl"))
    print(f"  Saved: models/scaler.pkl")
    print(f"  Saved: models/feature_names.pkl")

    # 7. Train/Test split
    print("\n[STEP 7] Splitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 8. Reshape for LSTM [samples, timesteps, features]
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    # 9. Save processed arrays
    print("\n[STEP 8] Saving processed arrays...")
    np.save(os.path.join(OUTPUT_FOLDER, "X_train.npy"), X_train_lstm)
    np.save(os.path.join(OUTPUT_FOLDER, "X_test.npy"),  X_test_lstm)
    np.save(os.path.join(OUTPUT_FOLDER, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_FOLDER, "y_test.npy"),  y_test)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test samples  : {len(X_test):,}")
    print(f"  Features used : {len(features)}")
    print(f"  Classes       : {len(le.classes_)}")
    print("=" * 60)
    print("\n  Next step: Run  python train_model.py")


if __name__ == "__main__":
    preprocess_and_save()
