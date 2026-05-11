"""
One-time model training script for structural integrity classifier.

Produces:
    - rf_model.joblib: Trained Random Forest classifier
    - scaler.joblib: Fitted StandardScaler
    - feature_names.json: Ordered list of 74 feature names
    - meta_info.json: Training metadata
    - X_features.npy: Feature matrix
    - y_labels.npy: Label vector
    - healthy_baseline.npz: Mean/std of Run A features (for diagnostics)

Usage:
    python model/train_model.py
"""

import os
import sys
import csv
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extraction import extract_features, FEATURE_NAMES

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "Structural.integrity.Seonsor.Board.data",
    "Sensor Board Update Initial Test"
)
BASE = os.path.normpath(BASE)

# Use all available boards (excluding known bad ones)
EXCLUDED_BOARDS = {"2523610160"}
LABEL_MAP = {"A": 0, "B": 1, "C": 2}

print(f"Data path: {BASE}")
print(f"Path exists: {os.path.exists(BASE)}")

# Load metadata
with open(os.path.join(BASE, "Test.csv"), encoding="utf-8") as f:
    meta_rows = list(csv.DictReader(f))

test_meta = {}
for r in meta_rows:
    r["Bolt Status"] = r["Bolt Status"].replace("Mix- 45 Deg", "Mix-45 Deg")
    sf1, ef1 = r.get("Start Freq 1", ""), r.get("End Freq 1", "")
    sf2, ef2 = r.get("Start Freq 2", ""), r.get("End Freq 2", "")
    pat = f"{sf1}-{ef1}Hz" if sf1 else ""
    if sf2:
        pat += f"+{sf2}-{ef2}Hz"
    test_meta[(r["Run"], r["Test"])] = pat

# Extract features
X_all, y_all, meta_all = [], [], []
healthy_features = []  # Track Run A features for baseline

for run in ["A", "B", "C"]:
    rp = os.path.join(BASE, run)
    if not os.path.isdir(rp):
        print(f"WARNING: Run directory {rp} not found")
        continue

    test_dirs = sorted(os.listdir(rp))
    processed = 0

    for td in test_dirs:
        tp = os.path.join(rp, td)
        if not os.path.isdir(tp) or (run, td) not in test_meta:
            continue
        pat = test_meta[(run, td)]

        for b in sorted(os.listdir(tp)):
            if b in EXCLUDED_BOARDS:
                continue
            bp = os.path.join(tp, b)
            if not os.path.isdir(bp):
                continue

            files = [f for f in os.listdir(bp)
                     if f.endswith(".csv") and not f.endswith("_i.csv")]
            if not files:
                continue

            try:
                d = pd.read_csv(
                    os.path.join(bp, sorted(files)[0]),
                    usecols=["X-axis", "Y-Axis", "Z-Axis"]
                ).values.astype(np.float64)

                if d.shape[0] != 8192:
                    continue

                fv = extract_features(d)
                X_all.append(fv)
                y_all.append(LABEL_MAP[run])
                meta_all.append({"run": run, "test": td, "board": b, "pattern": pat})
                processed += 1

                if run == "A":
                    healthy_features.append(fv)

            except Exception as e:
                pass

    print(f"Run {run}: processed {processed} samples")

X = np.array(X_all)
y = np.array(y_all)

print(f"\nTotal samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Class distribution: 30NM={sum(y==0)}, Loose={sum(y==1)}, Mix-45={sum(y==2)}")

# Compute healthy baseline statistics
healthy_X = np.array(healthy_features)
healthy_mean = healthy_X.mean(axis=0)
healthy_std = healthy_X.std(axis=0)

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_scaled, y)

# Training accuracy
train_acc = clf.score(X_scaled, y)
print(f"Training accuracy: {train_acc:.4f}")

# Save artifacts
model_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(clf, os.path.join(model_dir, "rf_model.joblib"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
    json.dump(FEATURE_NAMES, f)

with open(os.path.join(model_dir, "meta_info.json"), "w") as f:
    json.dump(meta_all, f)

np.save(os.path.join(model_dir, "X_features.npy"), X)
np.save(os.path.join(model_dir, "y_labels.npy"), y)
np.savez(os.path.join(model_dir, "healthy_baseline.npz"),
         mean=healthy_mean, std=healthy_std)

print(f"\nArtifacts saved to {model_dir}")
print("Training complete!")
