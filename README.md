
# UPACS — Unconscious Patient Autonomous Care System

**Purpose, Impact & Unique Selling Proposition (detailed)**
---
UPACS is a purpose-built research-to-demo system that argues for a practical and deployable alternative to EEG‑based unconsciousness / cognitive monitoring: **use ECG**. The pitch is simple but high-impact:
- **Accessibility:** ECG sensors are ubiquitous (wearables, monitors in wards/ICUs, portable single-lead devices). This lowers the barrier for continuous monitoring compared with EEG, which requires multiple electrodes, scalp prep and expert placement.
- **Robustness:** ECG signals are typically less noisy in non-ideal, movement-rich environments (wards, ambulances), enabling broader real-world data collection and easier integration into product pipelines.
- **Clinical utility:** Heart rate variability (HRV) and autonomic markers are established correlates of pain, stress, and sedation depth. UPACS leverages these relationships to provide useful proxy measures where EEG isn't available.
- **Product focus:** The system is designed to be *demoable* — a researcher or engineer can run the Streamlit demo, upload an ECG, and immediately get three clinically-relevant outputs: (1) mental-state detection, (2) unconsciousness depth (0/1/2), (3) pain/ANI-like score (0–100). This makes it ideal for hackathons, early-stage prototyping, and clinical research pipelines where rapid iteration matters.

**Why this matters:** If validated at scale, UPACS can reduce time-to-deploy monitoring solutions for perioperative wards, ambulances, telemedicine, and remote health scenarios. The combination of low-cost sensors + interpretable features (HRV/statistics) is the project’s core USP.

---
# Models — pipeline and full technical detail (start → end)
Below we describe each of the three sub-systems in UPACS. For each model we cover the data path (input → preprocessing → features → model → postprocessing), the training choices, and the inference-time behavior implemented in `app.py`.

## 1) Mental-State Detector (Stat7 → classifier)
**Goal:** Binary mental-state classification from a raw ECG snippet (single continuous trace).

**Input:** single-channel ECG (array of floats). Short traces (e.g., 30 s) are acceptable; longer also supported.

**Preprocessing (in inference):**
1. Basic sanitization: load numeric column from CSV/TXT/NPY/JSON/MAT (app supports many formats). fileciteturn1file5
2. Optional filtering: bandpass 0.5–40 Hz and a notch at mains frequency (50 Hz default) to remove baseline drift and mains hum. These filters are applied inside `bandpass()` and `notch()` helpers. fileciteturn1file5
3. No R-peak / HRV extraction is required for this model (it uses simple signal statistics).

**Feature extraction:** stat7 features — `mean, std, min, max, median, p25, p75`. Feature extraction is implemented as `stat7_features()` and is intentionally simple and fast. fileciteturn1file9

**Model architecture & training choices:** 
- We used a tree-based classifier (XGBoost / XGBClassifier in experiments). Trees are robust to feature scales and missingness and yield fast inference. The production demo expects a pre-trained classifier saved with `joblib`. `app.py` will call `model.predict()` or `model.predict_proba()` to display probabilities if supported. fileciteturn1file3
- Feature scaling: an optional scaler (StandardScaler/MinMaxScaler) can be used; if present the app applies it before prediction. Save the scaler with `joblib.dump()` using the path `models/mental_state_scaler.pkl`. fileciteturn1file0

**Inference output:** predicted class (Positive / Negative) and, where available, the probability of the positive class. The UI displays the stat7 JSON for inspection. fileciteturn1file3

---

## 2) Unconsciousness Level Analyzer (HRV10 → 3-class classifier)
**Goal:** Predict sedation / unconsciousness level with labels {0: Conscious, 1: Partial, 2: Deep} using HRV-derived features over sliding windows.

**Input:** continuous ECG (recommended ≥ window length). The UI exposes `win_s` (default 30 s) and `hop_s` (default 15 s) for sliding windows.

**Preprocessing pipeline (full detail):**
1. **Filtering & R-peak detection** — the raw ECG is bandpassed + notch filtered; R peaks are detected using `neurokit2` if available (preferred) and a robust envelope-based fallback if not. The fallback computes an envelope and finds peaks with a dynamic threshold and min distance. This makes detection resilient to a wide range of signals. fileciteturn1file5
2. **Windowing** — ECG is sliced into overlapping windows (`sliding_hrv_windows()`); each window is processed independently. If the entire trace is shorter than `win_s`, we compute HRV once across the entire trace. fileciteturn1file9
3. **HRV feature extraction** — for each window, compute time-domain and frequency-domain HRV features via `compute_hrv_from_peaks()`:
   - **Time domain:** `mean_hr`, `sdnn`, `rmssd`, `pnn50`, `sd1`, `sd2`. fileciteturn1file10
   - **Frequency domain (Welch PSD on interpolated RR):** `lf` (0.04–0.15 Hz), `hf` (0.15–0.4 Hz), `lf_hf`, and `total_power` (0.003–0.4 Hz band). The implementation uses an interpolated RR series and `scipy.signal.welch`. fileciteturn1file9

**Feature set:** the canonical 10 features (often called HRV10): `['mean_hr','sdnn','rmssd','pnn50','sd1','sd2','lf','hf','lf_hf','total_power']`. Missing values are handled via column medians before model input. fileciteturn1file12

**Model architecture & training choices:** 
- A tree-based classifier (XGBoost / XGBClassifier or equivalent) trained on per-window HRV rows. Trees provide robustness to partial missingness and non-linear interactions across HRV metrics.
- During inference, the app predicts per-window labels, then applies **majority voting** across windows to produce a single level for the uploaded trace. This stabilizes per-trace predictions and reduces spurious single-window flips. fileciteturn1file12

**Inference output & UI behavior:** 
- The UI shows a per-window table of HRV features + predicted label, the majority-vote label, and the badge color (green/yellow/red) for quick visual interpretation. CSV export is available for offline analysis. fileciteturn1file12

---

## 3) Pain Level Regressor (ANI-like, 0–100)
**Goal:** Predict continuous pain measure (analogous to ANI) in the 0–100 range using ECG-derived features and a multimodal *superset* feature vector approach so the model can be trained on datasets that include auxiliary signals (BVP, EDA, Temp, etc.) even when only ECG is available at inference time.

**Dataset used in experiments (example):** PMED / PMHDB pain dataset. In the training code you provided we load many per-subject CSV files, sanitize numeric columns and run LOSO (Leave-One-Group-Out) cross-validation across subject IDs. See the training snippet below. (This code is the training pipeline you pasted.)

**Important training snippet (pain model)** — *one small code pallet:*
```python
from xgboost import XGBRegressor
xgb_params = {
    "n_estimators": 2000, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5,
    "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 3.0,
    "gamma": 0.1, "random_state": 42, "verbosity": 0
}
model = XGBRegressor(**xgb_params)
model.fit(X_tr_sub, y_tr_sub,
          eval_set=[(X_val_sub, y_val_sub)],
          early_stopping_rounds=50,
          verbose=50)
preds = model.predict(X_test)
preds = np.clip(preds, 0.0, 100.0)
```
This training strategy uses **subject-level LOSO**, a large number of trees (2k) with small learning rate and early stopping — a sensible starting point for subject-generalizable regression. The per-fold output example shows validation RMSE settling (example: `validation_0-rmse:10.66` for a fold).

**Feature vector & inference-time adaptation:**  
- UPACS builds a **superset** pain feature vector in `build_pain_vector()` (a deterministic set: `['Seconds','Bvp','Eda_E4','Tmp','Ibi','Hr','Resp','Eda_RB','Ecg','Emg','Heater_C','COVAS','Heater_cleaned','Subject_ID']`). If your trained pain model expects fewer/more features, the app will **trim** or **pad** the vector automatically (removing `Subject_ID` first when trimming, then truncating, or padding with zeros), or you can provide `meta.json` with exact `candidate_cols` for exact alignment. This makes the demo robust to model feature-order mismatches. fileciteturn1file17

**Model architecture & training choices:** XGBoost regressor was used in experiments for its robustness and speed on tabular data. Prediction values are clipped to 0–100 during inference. The app attempts shape-resilient recovery if the model’s expected feature-count differs from the built vector. fileciteturn1file11

---
# Integration: how `app.py` ties everything together (complete runtime flow)
This section explains the Streamlit app end-to-end and how model selection triggers the correct processing pipeline.

**High-level UI controls:**
- **Model selection:** the sidebar provides a selectbox with options `mental_state`, `unconscious_level`, `pain`. The selection controls the downstream pipeline and UI panels. The `MODEL_REGISTRY` maps friendly labels to model file paths, scalers, and per-model metadata. fileciteturn1file0
- **Sampling rate input (`fs`)**: HRV-based models require a correct sampling rate. The app attempts automatic detection (candidate rates 125, 250, 360, 500) using peak-based heuristics (`estimate_fs_from_ecg()`) and will ask you to supply `fs` if it cannot auto-detect. fileciteturn1file13
- **Window / Hop controls:** for unconsciousness-level sliding windows. Defaults: `win_s=30`, `hop_s=15`. fileciteturn1file16
- **File upload / synthetic ECG:** app accepts CSV/TXT/NPY/JSON/MAT, or a 30s synthetic ECG for testing. fileciteturn1file5

**Model loading & safety:**  
- When the user presses **Run analysis**, the app loads the selected model (and scaler, if declared) via `safe_joblib_load()` and immediately validates the model file presence. If the model is missing the app stops with a helpful message. This ensures demo reliability and clear reproducibility steps. fileciteturn1file4

**Per-model execution flow (runtime):**  
- **Mental state:** compute `stat7_features()` → optional scaling → `model.predict()`/`predict_proba()` → display. fileciteturn1file3
- **Unconsciousness level:** compute sliding HRV windows (`sliding_hrv_windows()`) → fill NaNs with column medians → optional scaling → `model.predict()` per window → majority vote → display per-window table + aggregated label. fileciteturn1file12
- **Pain model:** compute `build_pain_vector()` with robust defaults → infer model expected feature count via `get_model_expected_n()` (supports `n_features_in_` and XGBoost booster introspection) → trim/pad if necessary → `model.predict()` → clip to [0,100] → display with visual badge and feature table. fileciteturn1file17

**Integration code (one small code pallet showing selection & load):**
```python
with st.sidebar:
    model_key = st.selectbox("Choose model", options=list(MODEL_REGISTRY.keys()),
                             format_func=lambda k: MODEL_REGISTRY[k]['label'])
# load model + scaler (safe loading)
model_info = MODEL_REGISTRY[model_key]
model = safe_joblib_load(model_info['model_file'])
scaler = safe_joblib_load(model_info['scaler_file']) if model_info.get('scaler_file') else None
```
This snippet is the entrypoint — once `model_key` is selected and `model`/`scaler` are loaded, the app enters the appropriate branch (mental_state / unconscious_level / pain) and runs the pipelines described above. fileciteturn1file16

**Robustness & failure modes:**  
- If R-peak detection fails in a given window, that window is skipped and flagged. If all windows fail, the app prompts for a longer or cleaner ECG. fileciteturn1file9  
- If pain model shape mismatches occur, the app first attempts an intelligent trim/pad, then a second recovery attempt; if prediction still fails, it asks the user to supply `meta.json` for exact alignment. fileciteturn1file11

---
# How to reproduce & run (quick)
1. Create environment and install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# requirements should include: streamlit, numpy, pandas, scipy, scikit-learn, xgboost, joblib, plotly, neurokit2 (optional)
```
2. Place models under `models/`:
- `models/mental_state_model.pkl` (+ optional `mental_state_scaler.pkl`)
- `models/unconscious_level_model.pkl` (+ optional `unconscious_level_scaler.pkl`)
- `models/pain_model.pkl` (+ optional `meta.json` describing `candidate_cols`) fileciteturn1file0
3. Run demo: `streamlit run app.py` and upload ECG (or use synthetic ECG).

---
# Training & evaluation recommendations
- Use **Leave-One-Subject-Out (LOSO)** for pain/regression to estimate subject generalization (the training snippet you provided uses scikit-learn's `LeaveOneGroupOut`). This is already implemented in your training loop skeleton.  
- Metrics to report:
  - **Mental state:** Accuracy, Precision, Recall, F1, ROC-AUC, per-subject performance. fileciteturn1file3
  - **Unconsciousness:** Macro-F1, per-class recall, confusion matrix, and Cohen’s Kappa. Also report per-window vs per-trace (after majority vote). fileciteturn1file12
  - **Pain:** MAE, RMSE, R², Bland–Altman plots and the percentage within ±10 units.

---
# Repo layout (recommended)
```
UPACS/
├─ app.py
├─ models/
│  ├─ mental_state_model.pkl
│  ├─ mental_state_scaler.pkl
│  ├─ unconscious_level_model.pkl
│  ├─ unconscious_level_scaler.pkl
│  ├─ pain_model.pkl
│  └─ meta.json   # optional (pain candidate_cols)
├─ notebooks/
│  ├─ mental-state-detection-ecg.ipynb
│  ├─ pain-classifier1 .ipynb
│  └─ unconscious-level-regression.ipynb
├─ requirements.txt
└─ README.md
```

---
# Final notes, limitations & next steps to highlight for recruiters
- This is a **research/demo** pipeline — clinical deployment requires rigorous trials and clear regulatory compliance. Always disclose this in the repo and demo.
- R-peak detection and HRV features are sensitive to severe noise; collecting a larger, diverse dataset and adding robust peak detection (template matching / ML-based) are strong next steps.
- Productizing: a small edge component that runs the HRV pipeline on-device and streams predictions to a dashboard would be a natural commercial follow-on.
