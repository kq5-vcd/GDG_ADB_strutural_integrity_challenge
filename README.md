# Structural Integrity Monitoring for Airfield Lighting Fixtures

**GDG x KU Leuven Hackathon | ADB Safegate Case Study 1**

Vibration-based fault classifier that detects bolt loosening in airfield lighting fixtures using 3-axis accelerometer data. The system distinguishes three mounting conditions (Healthy 30NM, Loose, Mix-45 Deg) and is designed to integrate with ADB Safegate's CORTEX Service platform for predictive maintenance at scale.

## Results

A Random Forest classifier achieves **99.1% accuracy** on the hardest honest evaluation (Leave-Excitation-Pattern-Family-Out cross-validation, 6 folds), with no data leakage between train and test sets. Gradient Boosting reaches 96.2% under the same regime. The model uses 74 physics-grounded features extracted from raw 27 kHz tri-axial vibration signals.

## Repository Structure

```
GDG-Hackaton/
|
|-- GDG_ADB_strutural_integrity_challenge/    # Analysis & ML pipeline
|   |-- README.md                             # Original case study description
|   |-- Structural_Integrity_Student_Manual.docx
|   |-- eda_structural_integrity.py           # Exploratory data analysis (8 plots)
|   |-- step1_extract_features.py             # 74-feature extraction pipeline
|   |-- classifier_pipeline.py                # RF + GBM training & evaluation
|   |-- classifier_pipeline_feature_list.txt  # Exported feature names
|   +-- model/
|       +-- train_model.py                    # Standalone model training script
|
|-- poc_structural_integrity/                 # Streamlit PoC dashboard
|   |-- app.py                                # Main entry point
|   |-- core/
|   |   |-- feature_extraction.py             # Feature extraction module
|   |   |-- signal_loader.py                  # Raw CSV signal loader
|   |   +-- physics.py                        # Physics interpretation engine
|   |-- model/
|   |   +-- train_model.py                    # One-time model training
|   +-- pages/
|       |-- 1_Fleet_Overview.py               # Airport fleet health dashboard
|       |-- 2_Real_Time_Monitoring.py         # Live monitoring simulation
|       |-- 3_Fixture_Diagnostic.py           # Physics-grounded diagnostics
|       +-- 4_Model_Validation.py             # Model performance & gap analysis
|
|-- POC_SPECIFICATION.md                      # Full PoC build specification
|
+-- Structural.integrity.Seonsor.Board.data/  # Raw dataset
    +-- Sensor Board Update Initial Test/
        |-- Test.csv                          # Master metadata (648 tests)
        |-- A/                                # Run A: 30NM healthy bolts
        |-- B/                                # Run B: Loose bolts
        +-- C/                                # Run C: Mix-45 Deg condition
```

## Dataset

The dataset contains 648 vibration tests across 3 bolt conditions, 15 sensor boards, and 48 excitation patterns. Each capture records 8,192 samples at 27 kHz (~303 ms) on 3 axes (X, Y, Z). The data hierarchy is: `Run / Test ID / Board ID / timestamp.csv`.

Three "universal" boards present across all runs were used for fair comparison: `2523610003`, `2523610134`, `2546610759`, yielding 1,742 samples for classification.

## Feature Engineering

74 physics-grounded features per sample, grouped as follows:

- **Per-axis (21 each, 63 total):** RMS, std, mean, kurtosis, skewness, crest factor, peak-to-peak, zero-crossing rate, spectral centroid, spectral spread, peak frequency, spectral rolloff, spectral flatness, band energies (low/mid/high/total), energy ratios, half-power bandwidth, Q-factor
- **Cross-axis (9):** Pearson correlation, RMS ratio, spectral correlation for XY, XZ, YZ pairs
- **Magnitude (2):** magnitude RMS, magnitude kurtosis

The physics rationale: bolt loosening reduces mounting stiffness, shifting resonant frequencies lower and redistributing vibration energy from high-frequency bands to low-frequency bands. This is directly captured by spectral centroid, peak frequency, and the low/high energy ratio.

## Running the Analysis Pipeline

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# 2. Run EDA (generates 8 diagnostic plots)
cd GDG_ADB_strutural_integrity_challenge
python eda_structural_integrity.py

# 3. Extract features (produces features CSV)
python step1_extract_features.py

# 4. Train and evaluate classifiers
python classifier_pipeline.py
```

## Running the PoC Dashboard

```bash
# 1. Install Streamlit
pip install streamlit plotly

# 2. Train the model (one-time)
cd poc_structural_integrity
python -m model.train_model

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard provides four views: fleet-level health overview, real-time monitoring simulation, per-fixture physics-grounded diagnostics, and model validation with gap analysis.

## Key Findings

1. **Loose bolts are easy to detect** -- spectral peak frequency drops from ~1,480 Hz (healthy) to ~270 Hz, a 5x shift visible in raw FFT
2. **Mix-45 Deg is the hard case** -- subtle spectral changes require engineered features (energy ratios, Q-factor) rather than raw signal inspection
3. **Lateral axes (X, Y) are more discriminative** than the vertical axis (Z), despite Z carrying the highest absolute vibration energy
4. **Leave-excitation-out CV is essential** -- naive random splits inflate accuracy by leaking excitation-pattern information between folds

## Tech Stack

Python 3.10+, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Plotly

## Team

GDG x KU Leuven Hackathon -- ADB Safegate Structural Integrity Challenge