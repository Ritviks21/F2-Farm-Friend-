# ============================================================
# Farm Friend — train.py
# Trains Random Forest on cleaned_crop_data.csv
# Saves model.pkl and feature_columns.pkl inside weights/
# Run this once to reproduce the final model
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Dynamically Handle Directory Paths ──────────────────
# This ensures paths work perfectly whether run locally or in production
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_crop_data.csv")
MODEL_OUT = os.path.join(BASE_DIR, "weights", "model.pkl")
COLS_OUT = os.path.join(BASE_DIR, "weights", "feature_columns.pkl")

# ── 2. Load Data ──────────────────────────────────────────
print("Loading data...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing dataset asset at {DATA_PATH}. Please verify your data directory placement.")

df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)} | Columns: {len(df.columns)} [cite: 86]")

# ── 3. Encode + Split ─────────────────────────────────────
print("Encoding categorical columns...")
cat_cols = ['Crop', 'Season', 'State']
X = df.drop(columns=['Crop_Year', 'Production', 'Yield'])
y = df['Yield']
X_encoded = pd.get_dummies(X, columns=cat_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows [cite: 102]")
print(f"  Features: {X_encoded.shape[1]}")

# ── 4. Train & Validate Model ──────────────────────────────
print("\nTraining Random Forest validation engine...")
start = time.time()

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
t = round(time.time() - start, 2)

# ── 5. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test)
r2   = round(r2_score(y_test, y_pred), 4)
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

print(f"  Done in {t}s")
print(f"  Validation R²   : {r2} (Reported: 0.9558) [cite: 104]")
print(f"  Validation RMSE : {rmse} t/ha (Reported: 2.1952) [cite: 104]")

# ── 6. Retrain on Full Data for Production ─────────────────
print("\nRetraining on full dataset for maximum real-world generalization...")
model.fit(X_encoded, y)

# ── 7. Serialize Artifacts to Weights Folder ───────────────
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(X_encoded.columns.tolist(), COLS_OUT)

print(f"\n✅ model.pkl structure saved to {MODEL_OUT} [cite: 117]")
print(f"✅ feature_columns.pkl structure saved to {COLS_OUT} [cite: 118]")
print("\nTraining complete. Repository codebase is fully unified and ready for deployment!")
