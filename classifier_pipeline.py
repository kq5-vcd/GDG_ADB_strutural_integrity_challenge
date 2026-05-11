"""
=============================================================================
Phase 2: Feature Engineering + Structural Fault Classifier
ADB Safegate Hackathon — Vibration-Based Fault Classification
=============================================================================
"""

import os, csv, warnings, json, time
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE = "../Structural.integrity.Seonsor.Board.data/Sensor Board Update Initial Test"
OUT = "classifier_results"
os.makedirs(OUT, exist_ok=True)

FS = 27000
N_SAMPLES = 8192

LABEL_MAP = {"A": 0, "B": 1, "C": 2}  # 30NM, Loose, Mix-45
LABEL_NAMES = ["30NM (Healthy)", "Loose", "Mix-45°"]

# =============================================================================
# 1. METADATA LOADING
# =============================================================================
print("=" * 60)
print("STEP 1: Loading metadata")
print("=" * 60)

def load_metadata():
    with open(f"{BASE}/Test.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Bolt Status"] = r["Bolt Status"].replace("Mix- 45 Deg", "Mix-45 Deg")
    return rows

meta = load_metadata()

# Build excitation pattern label for each test
def get_excitation_pattern(row):
    sf1, ef1 = row["Start Freq 1"], row["End Freq 1"]
    sf2, ef2 = row.get("Start Freq 2", ""), row.get("End Freq 2", "")
    pat = f"{sf1}-{ef1}Hz"
    if sf2:
        pat += f"+{sf2}-{ef2}Hz"
    return pat

# Build lookup: (run, test_id) -> metadata
test_meta = {}
for r in meta:
    test_meta[(r["Run"], r["Test"])] = r

# Build excitation group lookup
excitation_groups = defaultdict(list)
for r in meta:
    pat = get_excitation_pattern(r)
    excitation_groups[pat].append((r["Run"], r["Test"]))

print(f"  Total tests in metadata: {len(meta)}")
print(f"  Unique excitation patterns: {len(excitation_groups)}")

# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Physics-based feature extraction")
print("=" * 60)

def load_signal(run, test_id, board):
    path = f"{BASE}/{run}/{test_id}/{board}"
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith(".csv") and not f.endswith("_i.csv")]
    if not files:
        return None
    fpath = os.path.join(path, sorted(files)[0])
    try:
        data = np.genfromtxt(fpath, delimiter=",", skip_header=1, usecols=(1, 2, 3))
        if data.shape[0] != N_SAMPLES or np.any(np.isnan(data)):
            return None
        return data
    except:
        return None

def compute_features(signal_3axis):
    """
    Physics-based feature extraction from 3-axis vibration signal.
    
    Rationale:
    - Bolt loosening reduces mounting stiffness → lower resonant frequencies
    - Increased damping → lower kurtosis, broader spectral peaks
    - Changed modal coupling between axes → altered cross-axis coherence
    - Energy redistribution across frequency bands reflects structural change
    """
    feats = {}
    freqs_fft = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
    
    fft_mags = {}
    for ax_idx, ax in enumerate(["X", "Y", "Z"]):
        s = signal_3axis[:, ax_idx]
        s_c = s - s.mean()
        fft_mag = np.abs(np.fft.rfft(s_c))
        fft_mags[ax] = fft_mag
        
        # --- TIME DOMAIN ---
        rms = np.sqrt(np.mean(s_c ** 2))
        feats[f"{ax}_rms"] = rms
        feats[f"{ax}_std"] = np.std(s)
        feats[f"{ax}_mean"] = s.mean()
        feats[f"{ax}_kurtosis"] = np.mean(s_c**4) / (np.mean(s_c**2)**2 + 1e-10)
        feats[f"{ax}_skewness"] = np.mean(s_c**3) / (np.std(s_c)**3 + 1e-10)
        feats[f"{ax}_crest"] = np.max(np.abs(s_c)) / (rms + 1e-10)
        feats[f"{ax}_peak_to_peak"] = np.max(s) - np.min(s)
        
        # Zero crossing rate (vibration frequency proxy)
        zcr = np.sum(np.abs(np.diff(np.sign(s_c)))) / (2 * len(s_c))
        feats[f"{ax}_zcr"] = zcr
        
        # --- FREQUENCY DOMAIN ---
        mask_full = (freqs_fft >= 50) & (freqs_fft <= 4000)
        fft_band = fft_mag[mask_full]
        freqs_band = freqs_fft[mask_full]
        
        # Spectral centroid (center of mass of spectrum)
        feats[f"{ax}_spectral_centroid"] = np.sum(freqs_band * fft_band) / (np.sum(fft_band) + 1e-10)
        
        # Spectral spread (bandwidth around centroid)
        centroid = feats[f"{ax}_spectral_centroid"]
        feats[f"{ax}_spectral_spread"] = np.sqrt(np.sum((freqs_band - centroid)**2 * fft_band) / (np.sum(fft_band) + 1e-10))
        
        # Peak frequency (dominant resonance)
        if len(fft_band) > 0:
            feats[f"{ax}_peak_freq"] = freqs_band[np.argmax(fft_band)]
        else:
            feats[f"{ax}_peak_freq"] = 0
        
        # Spectral rolloff (95% energy frequency)
        cumsum = np.cumsum(fft_band**2)
        total = cumsum[-1] if len(cumsum) > 0 else 1e-10
        rolloff_idx = np.searchsorted(cumsum, 0.95 * total)
        feats[f"{ax}_spectral_rolloff"] = freqs_band[min(rolloff_idx, len(freqs_band)-1)] if len(freqs_band) > 0 else 0
        
        # Spectral flatness (tonality measure - lower = more tonal/resonant)
        log_mean = np.mean(np.log(fft_band + 1e-10))
        feats[f"{ax}_spectral_flatness"] = np.exp(log_mean) / (np.mean(fft_band) + 1e-10)
        
        # --- BAND ENERGY RATIOS ---
        # Physics: bolt loosening shifts energy from high to low bands
        e_low = np.sum(fft_mag[(freqs_fft >= 50) & (freqs_fft < 200)]**2)
        e_mid = np.sum(fft_mag[(freqs_fft >= 200) & (freqs_fft < 1000)]**2)
        e_high = np.sum(fft_mag[(freqs_fft >= 1000) & (freqs_fft < 4000)]**2)
        e_total = e_low + e_mid + e_high + 1e-10
        
        feats[f"{ax}_energy_low"] = e_low / e_total
        feats[f"{ax}_energy_mid"] = e_mid / e_total
        feats[f"{ax}_energy_high"] = e_high / e_total
        feats[f"{ax}_total_energy"] = e_total
        feats[f"{ax}_energy_ratio_lh"] = (e_low + 1e-10) / (e_high + 1e-10)
        feats[f"{ax}_energy_ratio_mh"] = (e_mid + 1e-10) / (e_high + 1e-10)
        
        # --- MODAL / DAMPING FEATURES ---
        # Find top 3 peaks and their relative heights (proxy for modal structure)
        from scipy.signal import find_peaks
        peaks, props = find_peaks(fft_band, height=np.max(fft_band)*0.1, distance=50)
        if len(peaks) >= 2:
            sorted_peaks = peaks[np.argsort(fft_band[peaks])[::-1]]
            feats[f"{ax}_mode1_freq"] = freqs_band[sorted_peaks[0]]
            feats[f"{ax}_mode2_freq"] = freqs_band[sorted_peaks[1]]
            feats[f"{ax}_mode_ratio"] = fft_band[sorted_peaks[1]] / (fft_band[sorted_peaks[0]] + 1e-10)
            feats[f"{ax}_mode_spacing"] = abs(freqs_band[sorted_peaks[1]] - freqs_band[sorted_peaks[0]])
        else:
            feats[f"{ax}_mode1_freq"] = feats[f"{ax}_peak_freq"]
            feats[f"{ax}_mode2_freq"] = 0
            feats[f"{ax}_mode_ratio"] = 0
            feats[f"{ax}_mode_spacing"] = 0
        
        # Half-power bandwidth of dominant peak (damping proxy)
        peak_val = np.max(fft_band) if len(fft_band) > 0 else 0
        half_power = peak_val / np.sqrt(2)
        above_half = fft_band >= half_power
        if np.any(above_half):
            idx_above = np.where(above_half)[0]
            bw = freqs_band[idx_above[-1]] - freqs_band[idx_above[0]]
            feats[f"{ax}_half_power_bw"] = bw
            feats[f"{ax}_q_factor"] = feats[f"{ax}_peak_freq"] / (bw + 1e-10)
        else:
            feats[f"{ax}_half_power_bw"] = 0
            feats[f"{ax}_q_factor"] = 0

    # --- CROSS-AXIS FEATURES ---
    # Physics: structural changes alter modal coupling between axes
    for a1, a2 in [("X","Y"), ("X","Z"), ("Y","Z")]:
        s1 = signal_3axis[:, ["X","Y","Z"].index(a1)] - signal_3axis[:, ["X","Y","Z"].index(a1)].mean()
        s2 = signal_3axis[:, ["X","Y","Z"].index(a2)] - signal_3axis[:, ["X","Y","Z"].index(a2)].mean()
        
        # Correlation
        corr = np.corrcoef(s1, s2)[0, 1]
        feats[f"corr_{a1}{a2}"] = corr
        
        # RMS ratio
        rms1 = np.sqrt(np.mean(s1**2))
        rms2 = np.sqrt(np.mean(s2**2))
        feats[f"rms_ratio_{a1}{a2}"] = rms1 / (rms2 + 1e-10)
        
        # Spectral coherence (correlation in frequency domain)
        fft1 = fft_mags[a1]
        fft2 = fft_mags[a2]
        mask = (freqs_fft >= 50) & (freqs_fft <= 4000)
        feats[f"spectral_corr_{a1}{a2}"] = np.corrcoef(fft1[mask], fft2[mask])[0, 1]
    
    # Magnitude of 3D acceleration vector
    mag = np.sqrt(np.sum(signal_3axis**2, axis=1))
    mag_c = mag - mag.mean()
    feats["mag_rms"] = np.sqrt(np.mean(mag_c**2))
    feats["mag_kurtosis"] = np.mean(mag_c**4) / (np.mean(mag_c**2)**2 + 1e-10)
    
    return feats

# Extract features from all valid recordings
print("  Extracting features from all recordings...")
t0 = time.time()

all_features = []
all_labels = []
all_meta_info = []  # (run, test_id, board, excitation_pattern)

# Only process tests that exist on disk
for run in ["A", "B", "C"]:
    run_path = f"{BASE}/{run}"
    test_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    
    for test_id in sorted(test_dirs):
        # Get metadata for this test
        meta_key = (run, test_id)
        if meta_key not in test_meta:
            continue
        
        row = test_meta[meta_key]
        pattern = get_excitation_pattern(row)
        
        test_path = os.path.join(run_path, test_id)
        boards = [b for b in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, b))]
        
        for board in boards:
            sig = load_signal(run, test_id, board)
            if sig is None:
                continue
            
            feats = compute_features(sig)
            feats["excitation_pattern"] = pattern
            feats["board"] = board
            feats["test_id"] = test_id
            feats["run"] = run
            
            all_features.append(feats)
            all_labels.append(LABEL_MAP[run])
            all_meta_info.append((run, test_id, board, pattern))

elapsed = time.time() - t0
print(f"  Extracted {len(all_features)} feature vectors in {elapsed:.1f}s")
print(f"  Class distribution: 30NM={all_labels.count(0)}, Loose={all_labels.count(1)}, Mix-45={all_labels.count(2)}")

# Build feature matrix
feature_names = [k for k in sorted(all_features[0].keys()) if k not in ["excitation_pattern", "board", "test_id", "run"]]
X = np.array([[f[k] for k in feature_names] for f in all_features])
y = np.array(all_labels)

# Handle any NaN/inf in features
nan_mask = ~np.isfinite(X)
if nan_mask.any():
    print(f"  Warning: {nan_mask.sum()} NaN/inf values found, replacing with 0")
    X[nan_mask] = 0

print(f"  Feature matrix shape: {X.shape} ({len(feature_names)} features)")

# Save feature names
with open(f"{OUT}/feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

# =============================================================================
# 3. CLASSIFICATION WITH RIGOROUS EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Classification with rigorous evaluation")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
import json

def run_cv(X, y, groups, cv_splitter, model_name, clf):
    scores = []
    y_true_all, y_pred_all = [], []
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        scores.append({"fold": fold, "accuracy": acc, "macro_f1": f1})
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    mean_acc = np.mean([s["accuracy"] for s in scores])
    mean_f1 = np.mean([s["macro_f1"] for s in scores])
    print(f"  {model_name}: Acc={mean_acc:.3f}, F1={mean_f1:.3f}")
    return scores, y_true_all, y_pred_all

# --- 3A. Naive Random Split (Baseline) ---
print("\n--- 3A: Naive Random Split (Leakage Baseline) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_naive, yt_naive, yp_naive = run_cv(
    X, y, None, skf, "RF (Naive)", 
    RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
)

# --- 3B. Leave-Physical-Test-Out CV ---
print("\n--- 3B: Leave-Physical-Test-Out (Replicate Split) ---")
test_ids = np.array([m[1] for m in all_meta_info])
gkf_test = GroupKFold(n_splits=5)
scores_lpto, yt_lpto, yp_lpto = run_cv(
    X, y, test_ids, gkf_test, "RF (LPTO)",
    RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
)

# --- 3C. Leave-Sensor-Board-Out CV ---
print("\n--- 3C: Leave-Sensor-Board-Out CV ---")
boards = np.array([m[2] for m in all_meta_info])
gkf_board = GroupKFold(n_splits=5)
scores_lsbo, yt_lsbo, yp_lsbo = run_cv(
    X, y, boards, gkf_board, "RF (LSBO)",
    RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
)

# --- 3D. Leave-Excitation-Pattern-Out CV ---
print("\n--- 3D: Leave-Excitation-Pattern-Out (LEPO) CV ---")
patterns = np.array([m[3] for m in all_meta_info])
def get_pattern_family(pat):
    if pat.startswith("50-100Hz") or pat.startswith("-Hz@"): return 0
    elif pat.startswith("50-200Hz"): return 1
    elif pat.startswith("50-1000Hz"): return 2
    elif pat.startswith("50-4000Hz"): return 3
    elif pat.startswith("100-200Hz"): return 4
    elif pat.startswith("100-1000Hz"): return 5
    else: return 6

pattern_families = np.array([get_pattern_family(p) for p in patterns])
gkf_pattern = GroupKFold(n_splits=6)
scores_lepo, yt_lepo, yp_lepo = run_cv(
    X, y, pattern_families, gkf_pattern, "RF (LEPO)",
    RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
)

# --- 3E. Linear Model Comparison ---
print("\n--- 3E: Linear Model Comparison (LEPO) ---")
# Using LogisticRegression for standard linear baseline on classification
scores_lr_lepo, yt_lr, yp_lr = run_cv(
    X, y, pattern_families, gkf_pattern, "LogReg (LEPO)",
    LogisticRegression(max_iter=2000, random_state=42)
)

# --- 3F. Strict Combinatorial Holdout ---
print("\n--- 3F: Strict Combinatorial Holdout ---")
unique_boards = np.unique(boards)
unique_patterns = np.unique(patterns)

# Hold out the last 2 boards and the '50-4000Hz' pattern completely
holdout_boards = unique_boards[-2:]
holdout_pattern = "50-4000Hz"

train_mask = ~(np.isin(boards, holdout_boards) | (patterns == holdout_pattern))
test_mask = np.isin(boards, holdout_boards) & (patterns == holdout_pattern)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

scaler_strict = StandardScaler()
X_train_s = scaler_strict.fit_transform(X_train)
X_test_s = scaler_strict.transform(X_test)

clf_strict = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
clf_strict.fit(X_train_s, y_train)
y_pred_strict = clf_strict.predict(X_test_s)

acc_strict = accuracy_score(y_test, y_pred_strict)
f1_strict = f1_score(y_test, y_pred_strict, average="macro")
print(f"  RF (Strict Holdout): Acc={acc_strict:.3f}, F1={f1_strict:.3f}")

# --- 3G. Final Feature Importance Analysis ---
print("\n--- 3G: Feature Importance Analysis ---")
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

clf_final = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
clf_final.fit(X_scaled, y)

importances = clf_final.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n  Top 15 most important features:")
for i in range(min(15, len(feature_names))):
    idx = sorted_idx[i]
    print(f"    {i+1:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")


# =============================================================================
# 4. SAVE RESULTS & PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Saving results and plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot 1: Model comparison bar chart across all splits
fig, ax = plt.subplots(figsize=(10, 6))
model_names = ["Naive (Random)", "LPTO (Test)", "LSBO (Board)", "LEPO (Pattern)", "LogReg (LEPO)", "RF (Strict Holdout)"]
accs = [
    np.mean([s["accuracy"] for s in scores_naive]),
    np.mean([s["accuracy"] for s in scores_lpto]),
    np.mean([s["accuracy"] for s in scores_lsbo]),
    np.mean([s["accuracy"] for s in scores_lepo]),
    np.mean([s["accuracy"] for s in scores_lr_lepo]),
    acc_strict
]
f1s = [
    np.mean([s["macro_f1"] for s in scores_naive]),
    np.mean([s["macro_f1"] for s in scores_lpto]),
    np.mean([s["macro_f1"] for s in scores_lsbo]),
    np.mean([s["macro_f1"] for s in scores_lepo]),
    np.mean([s["macro_f1"] for s in scores_lr_lepo])
]

x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, accs, width, label="Accuracy", color="#4A90D9", alpha=0.8)
ax.bar(x + width/2, f1s, width, label="Macro F1", color="#E8854A", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.set_title("Model Evaluation Across Validation Strategies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/split_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20
top_idx = sorted_idx[:top_n][::-1]
ax.barh(range(top_n), importances[top_idx], color="#4A90D9", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 20 Most Important Features (Random Forest)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Save numeric results
results = {
    "rf_naive": {"mean_acc": accs[0], "mean_f1": f1s[0]},
    "rf_lpto": {"mean_acc": accs[1], "mean_f1": f1s[1]},
    "rf_lsbo": {"mean_acc": accs[2], "mean_f1": f1s[2]},
    "rf_lepo": {"mean_acc": accs[3], "mean_f1": f1s[3]},
    "logreg_lepo": {"mean_acc": accs[4], "mean_f1": f1s[4]},
    "rf_strict_holdout": {"acc": acc_strict, "f1": f1_strict},
    "n_features": len(feature_names),
    "n_samples": len(all_features),
    "class_distribution": {"30NM": all_labels.count(0), "Loose": all_labels.count(1), "Mix-45": all_labels.count(2)}
}

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(f"{OUT}/X_features.npy", X)
np.save(f"{OUT}/y_labels.npy", y)

print(f"\n  All results saved to: {OUT}/")
print(f"  - split_comparison.png")
print(f"  - feature_importance.png")
print(f"  - results.json")
print(f"  - X_features.npy, y_labels.npy")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
