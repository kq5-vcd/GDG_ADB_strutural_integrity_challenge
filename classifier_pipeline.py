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

BASE = "/sessions/bold-magical-dirac/mnt/GDG-Hackaton/Structural.integrity.Seonsor.Board.data/Sensor Board Update Initial Test"
OUT = "/sessions/bold-magical-dirac/mnt/GDG-Hackaton/classifier_results"
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
# 3. CLASSIFICATION WITH PROPER SPLITS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Classification with rigorous evaluation")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
import json

# --- 3A. Leave-Excitation-Pattern-Out Cross-Validation ---
print("\n--- 3A: Leave-Excitation-Pattern-Out CV ---")
# This tests: can the classifier generalize to unseen excitation patterns?
# This is the HARDEST and most honest test.

patterns = np.array([m[3] for m in all_meta_info])
unique_patterns = np.unique(patterns)
print(f"  {len(unique_patterns)} unique excitation patterns")

# Assign pattern group IDs
pattern_to_id = {p: i for i, p in enumerate(unique_patterns)}
pattern_groups = np.array([pattern_to_id[p] for p in patterns])

# GroupKFold with 6 folds (grouping by excitation pattern families)
# Group patterns into 6 families by primary frequency band
def get_pattern_family(pat):
    """Group excitation patterns by primary sweep band."""
    if pat.startswith("50-100Hz") or pat.startswith("-Hz@"):
        return 0
    elif pat.startswith("50-200Hz"):
        return 1
    elif pat.startswith("50-1000Hz"):
        return 2
    elif pat.startswith("50-4000Hz"):
        return 3
    elif pat.startswith("100-200Hz"):
        return 4
    elif pat.startswith("100-1000Hz"):
        return 5
    else:
        return 6

pattern_families = np.array([get_pattern_family(p) for p in patterns])

gkf = GroupKFold(n_splits=6)
lepo_scores = []
lepo_reports = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=pattern_families)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    lepo_scores.append({"fold": fold, "accuracy": acc, "macro_f1": f1})
    
    test_patterns = set(patterns[test_idx])
    print(f"  Fold {fold}: acc={acc:.3f}, F1={f1:.3f} | test patterns: {len(test_patterns)}")
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

print(f"\n  LEPO Mean: acc={np.mean([s['accuracy'] for s in lepo_scores]):.3f} +/- {np.std([s['accuracy'] for s in lepo_scores]):.3f}")
print(f"  LEPO Mean: F1={np.mean([s['macro_f1'] for s in lepo_scores]):.3f} +/- {np.std([s['macro_f1'] for s in lepo_scores]):.3f}")

print("\n  Overall Classification Report (LEPO):")
print(classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES))

cm_lepo = confusion_matrix(all_y_true, all_y_pred)
print("  Confusion Matrix (LEPO):")
print(cm_lepo)

# --- 3B. Leave-Replicate-Out CV ---
print("\n--- 3B: Leave-Replicate-Out CV ---")
# Group by replicate number (last digit of test ID for standard tests)
replicate_groups = []
for m in all_meta_info:
    tid = m[1]
    if len(tid) >= 5:
        rep = int(tid[-1])  # last digit is replicate number (1-5)
    else:
        rep = 0  # special tests
    replicate_groups.append(rep)
replicate_groups = np.array(replicate_groups)

gkf_rep = GroupKFold(n_splits=5)
lro_scores = []
lro_y_true = []
lro_y_pred = []

for fold, (train_idx, test_idx) in enumerate(gkf_rep.split(X, y, groups=replicate_groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    lro_scores.append({"fold": fold, "accuracy": acc, "macro_f1": f1})
    print(f"  Fold {fold}: acc={acc:.3f}, F1={f1:.3f}")
    
    lro_y_true.extend(y_test)
    lro_y_pred.extend(y_pred)

print(f"\n  LRO Mean: acc={np.mean([s['accuracy'] for s in lro_scores]):.3f} +/- {np.std([s['accuracy'] for s in lro_scores]):.3f}")
print(f"  LRO Mean: F1={np.mean([s['macro_f1'] for s in lro_scores]):.3f} +/- {np.std([s['macro_f1'] for s in lro_scores]):.3f}")

print("\n  Overall Classification Report (LRO):")
print(classification_report(lro_y_true, lro_y_pred, target_names=LABEL_NAMES))

cm_lro = confusion_matrix(lro_y_true, lro_y_pred)
print("  Confusion Matrix (LRO):")
print(cm_lro)

# --- 3C. Train final model on all data for feature importance ---
print("\n--- 3C: Feature Importance Analysis ---")
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

clf_final = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
clf_final.fit(X_scaled, y)

importances = clf_final.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n  Top 25 most important features:")
for i in range(min(25, len(feature_names))):
    idx = sorted_idx[i]
    print(f"    {i+1:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")

# --- 3D. Gradient Boosting Comparison ---
print("\n--- 3D: Gradient Boosting Comparison ---")
gb_scores = []
gb_y_true = []
gb_y_pred = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=pattern_families)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf_gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    clf_gb.fit(X_train_s, y_train)
    y_pred = clf_gb.predict(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    gb_scores.append({"fold": fold, "accuracy": acc, "macro_f1": f1})
    gb_y_true.extend(y_test)
    gb_y_pred.extend(y_pred)

print(f"  GB LEPO Mean: acc={np.mean([s['accuracy'] for s in gb_scores]):.3f} +/- {np.std([s['accuracy'] for s in gb_scores]):.3f}")
print(f"  GB LEPO Mean: F1={np.mean([s['macro_f1'] for s in gb_scores]):.3f} +/- {np.std([s['macro_f1'] for s in gb_scores]):.3f}")

print("\n  GB Classification Report (LEPO):")
print(classification_report(gb_y_true, gb_y_pred, target_names=LABEL_NAMES))

cm_gb = confusion_matrix(gb_y_true, gb_y_pred)

# =============================================================================
# 4. PER-EXCITATION PATTERN ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Per-excitation pattern discriminative analysis")
print("=" * 60)

# Train on all data, then look at per-pattern accuracy
clf_final.fit(X_scaled, y)  # already done above
y_pred_all = clf_final.predict(X_scaled)

pattern_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
for i, (run, tid, board, pat) in enumerate(all_meta_info):
    pattern_accuracy[pat]["total"] += 1
    if y_pred_all[i] == y[i]:
        pattern_accuracy[pat]["correct"] += 1

print("\n  Per-excitation pattern accuracy (training set, for signal analysis):")
print(f"  {'Pattern':<50} {'Acc':>6} {'N':>6}")
print("  " + "-" * 65)
for pat in sorted(pattern_accuracy.keys()):
    d = pattern_accuracy[pat]
    acc = d["correct"] / d["total"]
    print(f"  {pat:<50} {acc:>6.1%} {d['total']:>6}")

# =============================================================================
# 5. PER-BOARD ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Per-board discriminative analysis")
print("=" * 60)

board_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
for i, (run, tid, board, pat) in enumerate(all_meta_info):
    board_accuracy[board]["total"] += 1
    if y_pred_all[i] == y[i]:
        board_accuracy[board]["correct"] += 1

print(f"\n  {'Board':<15} {'Acc':>6} {'N':>6}")
print("  " + "-" * 30)
for board in sorted(board_accuracy.keys()):
    d = board_accuracy[board]
    acc = d["correct"] / d["total"]
    print(f"  {board:<15} {acc:>6.1%} {d['total']:>6}")

# =============================================================================
# 6. SAVE RESULTS & PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Saving results and plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot 1: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cm, title in zip(axes, 
    [cm_lepo, cm_lro, cm_gb],
    ["Random Forest (Leave-Excitation-Out)", "Random Forest (Leave-Replicate-Out)", "Gradient Boosting (Leave-Excitation-Out)"]):
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Feature importance (top 20)
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

# Plot 3: Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
models = ["RF (LEPO)", "RF (LRO)", "GB (LEPO)"]
accs = [
    np.mean([s["accuracy"] for s in lepo_scores]),
    np.mean([s["accuracy"] for s in lro_scores]),
    np.mean([s["accuracy"] for s in gb_scores]),
]
f1s = [
    np.mean([s["macro_f1"] for s in lepo_scores]),
    np.mean([s["macro_f1"] for s in lro_scores]),
    np.mean([s["macro_f1"] for s in gb_scores]),
]
acc_stds = [
    np.std([s["accuracy"] for s in lepo_scores]),
    np.std([s["accuracy"] for s in lro_scores]),
    np.std([s["accuracy"] for s in gb_scores]),
]

x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, accs, width, yerr=acc_stds, label="Accuracy", color="#4A90D9", alpha=0.8, capsize=5)
ax.bar(x + width/2, f1s, width, label="Macro F1", color="#E8854A", alpha=0.8, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.set_title("Model Comparison Across Evaluation Strategies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Save numeric results
results = {
    "rf_lepo": {"mean_acc": float(np.mean([s["accuracy"] for s in lepo_scores])),
                "std_acc": float(np.std([s["accuracy"] for s in lepo_scores])),
                "mean_f1": float(np.mean([s["macro_f1"] for s in lepo_scores]))},
    "rf_lro": {"mean_acc": float(np.mean([s["accuracy"] for s in lro_scores])),
               "std_acc": float(np.std([s["accuracy"] for s in lro_scores])),
               "mean_f1": float(np.mean([s["macro_f1"] for s in lro_scores]))},
    "gb_lepo": {"mean_acc": float(np.mean([s["accuracy"] for s in gb_scores])),
                "std_acc": float(np.std([s["accuracy"] for s in gb_scores])),
                "mean_f1": float(np.mean([s["macro_f1"] for s in gb_scores]))},
    "n_features": len(feature_names),
    "n_samples": len(all_features),
    "class_distribution": {"30NM": all_labels.count(0), "Loose": all_labels.count(1), "Mix-45": all_labels.count(2)},
}

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save feature matrix for further experiments
np.save(f"{OUT}/X_features.npy", X)
np.save(f"{OUT}/y_labels.npy", y)

print(f"\n  All results saved to: {OUT}/")
print(f"  - confusion_matrices.png")
print(f"  - feature_importance.png")
print(f"  - model_comparison.png")
print(f"  - results.json")
print(f"  - feature_names.json")
print(f"  - X_features.npy, y_labels.npy")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
