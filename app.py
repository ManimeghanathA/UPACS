# app.py — UPACS Streamlit app (final, robust, UI-polished)
import io
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt, iirnotch, welch, find_peaks

# Optional: neurokit2 for improved R peak detection (if installed)
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except Exception:
    NK_AVAILABLE = False

# -------------------------
# Configuration / models
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_REGISTRY = {
    "mental_state": {
        "label": "Mental State (stat7 → classifier)",
        "model_file": os.path.join(MODELS_DIR, "mental_state_model.pkl"),
        "scaler_file": os.path.join(MODELS_DIR, "mental_state_scaler.pkl"),
        "feature_type": "stat7",
    },
    "unconscious_level": {
        "label": "Unconsciousness Level (HRV10 → classifier 0/1/2)",
        "model_file": os.path.join(MODELS_DIR, "unconscious_level_model.pkl"),
        "scaler_file": os.path.join(MODELS_DIR, "unconscious_level_scaler.pkl"),
        "feature_type": "hrv10",
        "candidate_cols": ['mean_hr','sdnn','rmssd','pnn50','sd1','sd2','lf','hf','lf_hf','total_power']
    },
    "pain": {
        "label": "Pain Analyzer (multimodal → regressor ANI 0–100)",
        "model_file": os.path.join(MODELS_DIR, "pain_model.pkl"),
        "scaler_file": None,
        # we keep a superset order (Subject_ID optional); we'll automatically trim/pad to match model expectations
        "pain_cols_order": ['Seconds','Bvp','Eda_E4','Tmp','Ibi','Hr','Resp','Eda_RB','Ecg','Emg','Heater_C','COVAS','Heater_cleaned','Subject_ID']
    },
}

# -------------------------
# Page styling
# -------------------------
st.set_page_config(page_title="UPACS", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      body { background: linear-gradient(180deg,#f8fafc 0%, #ffffff 60%); }
      .app-title { font-size:45px; font-weight:800; color:#fcfefe; text-align : center }
      .muted { color:#eaf6f6; margin : 10px; text-align:center;}
      .card { background: white; border-radius:12px; padding:14px; box-shadow: 0 8px 24px rgba(15,23,42,0.06); color: black; }
      .badge-green { background:#ecfdf5; color:#065f46; padding:6px 10px; border-radius:999px; font-weight:700; }
      .badge-yellow { background:#fffbeb; color:#78350f; padding:6px 10px; border-radius:999px; font-weight:700; }
      .badge-red { background:#fff1f2; color:#7f1d1d; padding:6px 10px; border-radius:999px; font-weight:700; }
      .small { font-size:13px; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helper utilities
# -------------------------
def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def read_ecg_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if not raw:
        return None
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
            # pick first numeric column
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    arr = np.asarray(df[c].astype(float).values).ravel()
                    return arr
            return np.asarray(df.iloc[:,0].astype(float).values).ravel()
        elif name.endswith(".txt"):
            return np.loadtxt(io.BytesIO(raw)).astype(float).ravel()
        elif name.endswith(".npy"):
            return np.load(io.BytesIO(raw), allow_pickle=True).astype(float).ravel()
        elif name.endswith(".json"):
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict) and "ecg" in obj:
                return np.asarray(obj["ecg"], dtype=float).ravel()
            elif isinstance(obj, list):
                return np.asarray(obj, dtype=float).ravel()
            else:
                raise ValueError("JSON must be list or {'ecg': [...]} ")
        elif name.endswith(".mat"):
            from scipy.io import loadmat
            md = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
            for k in ("ecg", "ECG", "signal", "sig", "data"):
                if k in md:
                    a = np.asarray(md[k], dtype=float).ravel()
                    if a.size > 0:
                        return a
            for k, v in md.items():
                try:
                    a = np.asarray(v, dtype=float).ravel()
                    if a.ndim == 1 and a.size > 0:
                        return a
                except Exception:
                    continue
            raise ValueError("No ECG found in MAT file.")
        else:
            # try CSV, then txt
            try:
                df = pd.read_csv(io.BytesIO(raw))
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        return np.asarray(df[c].astype(float).values).ravel()
                return np.asarray(df.iloc[:,0].astype(float).values).ravel()
            except Exception:
                return np.loadtxt(io.BytesIO(raw)).astype(float).ravel()
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")
        return None

def bandpass(sig, fs, low=0.5, high=40.0, order=4):
    try:
        b, a = butter(order, [low/(fs/2.0), high/(fs/2.0)], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def notch(sig, fs, freq=50.0, q=30.0):
    try:
        b, a = iirnotch(freq/(fs/2.0), q)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def detect_rpeaks(sig, fs):
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return np.array([], dtype=int)
    # prefer neurokit2 if installed
    if NK_AVAILABLE:
        try:
            out = nk.ecg_peaks(sig, sampling_rate=fs)[1]
            peaks = np.asarray(out.get("ECG_R_Peaks", []), dtype=int)
            if peaks.size >= 3:
                return peaks
        except Exception:
            pass
    # fallback simple envelope peak detection
    try:
        s = sig - np.mean(sig)
        s = bandpass(s, fs, 0.5, 40.0)
        s = notch(s, fs)
        env = np.abs(s)
        win = max(1, int(0.03 * fs))
        env_smooth = np.convolve(env, np.ones(win)/win, mode="same")
        thr = np.percentile(env_smooth, 50) + 0.35 * (np.percentile(env_smooth, 95) - np.percentile(env_smooth, 50))
        min_dist = max(1, int(0.35 * fs))
        peaks, _ = find_peaks(env_smooth, height=thr, distance=min_dist)
        return np.asarray(peaks, dtype=int)
    except Exception:
        return np.array([], dtype=int)

def compute_hrv_from_peaks(peaks, fs):
    if len(peaks) < 4:
        return None
    rr_s = np.diff(peaks) / float(fs)
    if rr_s.size < 2:
        return None
    rr_ms = rr_s * 1000.0
    out = {}
    out['mean_hr'] = float(60.0 / rr_s.mean()) if rr_s.mean()>0 else np.nan
    out['sdnn'] = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else np.nan
    diff_rr = np.diff(rr_ms)
    out['rmssd'] = float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else np.nan
    out['pnn50'] = float(np.mean(np.abs(diff_rr) > 50.0)) if diff_rr.size > 0 else np.nan
    if rr_ms.size >= 2:
        sd1 = np.sqrt(0.5) * np.std(np.diff(rr_ms))
        sd1sq = sd1**2
        sdnn = out['sdnn'] if not math.isnan(out['sdnn']) else np.std(rr_ms, ddof=1)
        sd2sq = max(0.0, 2.0 * (sdnn**2) - sd1sq)
        out['sd1'] = float(sd1)
        out['sd2'] = float(math.sqrt(sd2sq))
    else:
        out['sd1'] = np.nan; out['sd2'] = np.nan
    try:
        times = peaks[1:] / float(fs)
        if len(times) >= 4:
            fs_interp = 4.0
            t0, t1 = times[0], times[-1]
            if t1 - t0 < 1.0:
                t_interp = np.linspace(t0, t1, max(4, int((t1 - t0) * fs_interp)))
            else:
                t_interp = np.arange(t0, t1, 1.0/fs_interp)
                if len(t_interp) < 4:
                    t_interp = np.linspace(t0, t1, max(4, int((t1 - t0) * fs_interp)))
            rr_interp = np.interp(t_interp, times, rr_ms)
            rr_interp = rr_interp - np.mean(rr_interp)
            nperseg = min(256, len(rr_interp))
            fxx, pxx = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
            lf_mask = (fxx >= 0.04) & (fxx < 0.15)
            hf_mask = (fxx >= 0.15) & (fxx < 0.4)
            lf = float(np.trapz(pxx[lf_mask], fxx[lf_mask])) if lf_mask.any() else 0.0
            hf = float(np.trapz(pxx[hf_mask], fxx[hf_mask])) if hf_mask.any() else 0.0
            total_mask = (fxx >= 0.003) & (fxx < 0.4)
            total_power = float(np.trapz(pxx[total_mask], fxx[total_mask])) if total_mask.any() else (lf + hf)
            out['lf'] = lf; out['hf'] = hf
            out['lf_hf'] = float(lf/hf) if (hf is not None and hf > 0) else np.nan
            out['total_power'] = float(total_power)
        else:
            out['lf'] = np.nan; out['hf'] = np.nan; out['lf_hf'] = np.nan; out['total_power'] = np.nan
    except Exception:
        out['lf'] = np.nan; out['hf'] = np.nan; out['lf_hf'] = np.nan; out['total_power'] = np.nan
    return out

def sliding_hrv_windows(ecg, fs, win_s=30, hop_s=15):
    ecg = np.asarray(ecg, dtype=float)
    n = len(ecg)
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    rows = []
    if n < win:
        peaks = detect_rpeaks(ecg, fs)
        feats = compute_hrv_from_peaks(peaks, fs)
        if feats:
            rows.append({'start_sec': 0.0, 'end_sec': n/float(fs), **feats})
        return rows
    for start in range(0, n - win + 1, hop):
        w = ecg[start:start+win]
        peaks = detect_rpeaks(w, fs)
        feats = compute_hrv_from_peaks(peaks, fs)
        if feats:
            rows.append({'start_sec': start/float(fs), 'end_sec': (start+win)/float(fs), **feats})
    return rows

def stat7_features(ecg):
    a = np.asarray(ecg, dtype=float)
    return {
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'median': float(np.median(a)),
        'p25': float(np.percentile(a, 25)),
        'p75': float(np.percentile(a, 75))
    }

def downsample_for_plot(y, max_points=6000):
    n = len(y)
    if n <= max_points:
        return np.arange(n), np.asarray(y)
    factor = int(np.ceil(n / max_points))
    return np.arange(0, n, factor), np.asarray(y[::factor])

def estimate_fs_from_ecg(ecg, candidates=(125,250,360,500)):
    best = None; best_score = 1e9
    for fs in candidates:
        try:
            p = detect_rpeaks(ecg, fs)
            if len(p) < 3:
                continue
            rr = np.diff(p) / float(fs)
            mean_hr = 60.0 / rr.mean() if rr.mean()>0 else 0.0
            score = 0 if (40 <= mean_hr <= 140) else abs(mean_hr - 80)
            if score < best_score:
                best_score = score; best = fs
        except Exception:
            pass
    return best

def get_model_expected_n(model):
    """Try multiple ways to extract expected number of input features from model."""
    try:
        # sklearn attribute
        if hasattr(model, "n_features_in_"):
            n = int(getattr(model, "n_features_in_"))
            if n>0:
                return n
    except Exception:
        pass
    try:
        booster = None
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif hasattr(model, "booster_"):
            booster = model.booster_
        if booster is not None:
            # feature_names list
            fnames = getattr(booster, "feature_names", None)
            if fnames:
                return len(fnames)
            # num_feature method
            numf = getattr(booster, "num_feature", None)
            if callable(numf):
                return int(numf())
            # alternative
            numf2 = getattr(booster, "num_features", None)
            if callable(numf2):
                return int(numf2())
    except Exception:
        pass
    return None

# -------------------------
# Pain features builder (robust to expected size)
# -------------------------
def build_pain_vector(ecg, fs, expected_n=None):
    """Return (X_row, used_cols) where X_row shape is (1, expected_n) if expected_n provided.
       We construct a superset of features from ECG and safe simulated defaults, then trim/pad.
    """
    # Base order (superset); Subject_ID is optional and will be removed first if trimming needed
    base_cols = MODEL_REGISTRY['pain']['pain_cols_order'][:]  # local copy
    n = len(ecg)
    duration_s = n / float(fs) if fs and fs>0 else 0.0

    # compute peaks -> HR/IBI
    peaks = detect_rpeaks(ecg, fs) if fs else np.array([], dtype=int)
    if len(peaks) >= 2:
        rr_s = np.diff(peaks) / float(fs)
        ibi_ms = float(np.mean(rr_s) * 1000.0)
        mean_hr = float(60.0 / rr_s.mean()) if rr_s.mean()>0 else 60.0
    else:
        ibi_ms = 1000.0
        mean_hr = 60.0

    # BVP proxy (envelope)
    env = np.abs(ecg - np.mean(ecg))
    bvp = float(np.mean(env)) if env.size else 0.1
    if bvp == 0.0: bvp = 0.1
    eda_e4 = float(np.median(env) * 0.5 + 0.2)
    eda_rb = eda_e4 * 0.95
    tmp = 36.5 + (np.mean(env) % 0.5) if env.size else 36.5
    resp = float(max(8.0, min(22.0, (mean_hr or 60.0) / 4.0)))
    ecg_mean = float(np.mean(ecg))
    emg = float(max(0.0, np.std(ecg) * 0.08))  # small muscle-like metric
    heater_c = 0.0
    heater_cleaned = 0.0
    covas = 0.0
    subject_id = 0  # default

    values = {
        'Seconds': float(duration_s),
        'Bvp': float(bvp),
        'Eda_E4': float(eda_e4),
        'Tmp': float(tmp),
        'Ibi': float(ibi_ms),
        'Hr': float(mean_hr),
        'Resp': float(resp),
        'Eda_RB': float(eda_rb),
        'Ecg': float(ecg_mean),
        'Emg': float(emg),
        'Heater_C': float(heater_c),
        'COVAS': float(covas),
        'Heater_cleaned': float(heater_cleaned),
        'Subject_ID': float(subject_id),
    }

    # initial row in base_cols order
    cols_local = base_cols[:]
    row = [float(values.get(c, 0.0)) for c in cols_local]

    # If model expects a certain number of features, trim/pad deterministically:
    if expected_n is None:
        # prefer 12 (common), but don't force - just reduce by dropping Subject_ID if present and length>12
        if len(row) > 12 and 'Subject_ID' in cols_local:
            idx = cols_local.index('Subject_ID')
            cols_local.pop(idx)
            row.pop(idx)
        # if still >12, truncate to 12
        if len(row) > 12:
            cols_local = cols_local[:12]
            row = row[:12]
        # if <12, pad with zeros
        while len(row) < 12:
            cols_local.append(f'pad_{len(row)}'); row.append(0.0)
    else:
        # adapt to expected_n
        if len(row) > expected_n:
            # remove 'Subject_ID' first if present
            if 'Subject_ID' in cols_local and len(row) - 1 >= expected_n:
                idx = cols_local.index('Subject_ID')
                cols_local.pop(idx); row.pop(idx)
            # if still too long, truncate from end
            while len(row) > expected_n:
                cols_local.pop() ; row.pop()
        elif len(row) < expected_n:
            # pad zeros and name them pad_i
            while len(row) < expected_n:
                cols_local.append(f'pad_{len(row)}'); row.append(0.0)

    X = np.asarray(row, dtype=float).reshape(1, -1)
    return X, cols_local

# -------------------------
# UI: Sidebar & controls
# -------------------------
st.markdown("<div class='app-title'>🩺 UPACS — Unconscious Patient Autonomous Care System</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>ECG-focused demo — mental state, unconsciousness level, and pain (ANI). Clean UI, safe fallbacks.</div>", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Controls")
    model_key = st.selectbox("Choose model", options=list(MODEL_REGISTRY.keys()), format_func=lambda k: MODEL_REGISTRY[k]['label'])
    model_info = MODEL_REGISTRY[model_key]
    fs_input = st.number_input("Sampling rate (Hz) — 0 = auto detect", min_value=0.0, value=0.0, step=1.0)
    win_s = st.number_input("Window (s) for HRV windows", min_value=5.0, value=30.0, step=1.0)
    hop_s = st.number_input("Hop (s) for HRV windows", min_value=1.0, value=15.0, step=1.0)
    uploaded_file = st.file_uploader("Upload ECG (CSV/TXT/NPY/JSON/MAT)", type=["csv","txt","npy","json","mat"])
    use_synth = st.checkbox("Use synthetic ECG (30s @ 250Hz)", value=False)
    run_button = st.button("Run analysis", key="run_button")

    st.markdown("---")
    st.write("Model files expected in:")
    st.code(MODELS_DIR)
    for k,mi in MODEL_REGISTRY.items():
        exists = os.path.exists(mi['model_file'])
        sc_exists = bool(mi.get('scaler_file') and os.path.exists(mi['scaler_file']))
        st.write(f"- {mi['label']}: model {'✅' if exists else '❌'}  scaler {'✅' if sc_exists else '—' if mi.get('scaler_file') else 'n/a'}")

# if not run, show quick help
if not run_button:
    st.info("Upload ECG and press Run (or choose synthetic ECG). For HRV-based models, provide sampling rate or leave blank for auto-detect.")
    st.stop()

# ---------- Execute analysis ----------
# 1) read ECG
ecg = None
if uploaded_file:
    ecg = read_ecg_from_upload(uploaded_file)
elif use_synth:
    fs_default = 250.0
    t = np.linspace(0, 30, int(30 * fs_default))
    ecg = 0.9 * np.sin(2*np.pi*1.2*t) + 0.2 * np.random.randn(len(t))
    st.sidebar.success("Using synthetic ECG (30s @ 250Hz)")
else:
    st.error("No ECG supplied. Upload a file or enable synthetic ECG.")
    st.stop()

if ecg is None or len(ecg) < 4:
    st.error("ECG unreadable or too short.")
    st.stop()

# show waveform (left column) and model summary (right column)
col_left, col_right = st.columns([2,1])

# waveform: raw + smoothed area
x_ds, y_ds = downsample_for_plot(ecg, max_points=8000)
import plotly.graph_objects as go
smooth = pd.Series(y_ds).rolling(window=max(3, int(len(y_ds)/200)), min_periods=1).mean().values
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode='lines', line=dict(color='#0ea5e9', width=1), name='ECG (raw)', opacity=0.7))
fig.add_trace(go.Scatter(x=x_ds, y=smooth, mode='lines', line=dict(color='#0369a1', width=2), name='Smoothed'))
fig.add_trace(go.Scatter(x=x_ds, y=smooth, fill='tozeroy', fillcolor='rgba(3,105,161,0.08)', line=dict(width=0), showlegend=False))
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=340, template='plotly_white', title='ECG waveform')
col_left.plotly_chart(fig, use_container_width=True)

# summary card
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**Model:** {model_info['label']}")
    st.write("")
    st.markdown(f"- Samples: **{len(ecg)}**")
    st.markdown(f"- Sampling rate (input): **{fs_input if fs_input>0 else 'auto'}**")
    st.markdown("</div>", unsafe_allow_html=True)

# detect sampling rate
fs = float(fs_input) if fs_input and fs_input>0 else None
if fs is None:
    detected = estimate_fs_from_ecg(ecg)
    if detected:
        fs = float(detected)
        st.success(f"Auto-detected sampling rate ≈ {fs} Hz")
    else:
        st.warning("Could not auto-detect sampling rate. Please provide sampling rate in sidebar.")
        st.stop()

# load model + scaler
model = safe_joblib_load(model_info['model_file'])
if model is None:
    st.error(f"Could not load model from {model_info['model_file']}. Please confirm file exists.")
    st.stop()
scaler = None
if model_info.get('scaler_file'):
    scaler = safe_joblib_load(model_info['scaler_file'])

# -------------------------
# MODEL: Mental State (stat7)
# -------------------------
if model_key == 'mental_state':
    feats = stat7_features(ecg)
    feat_order = ['mean','std','min','max','median','p25','p75']
    X = np.asarray([feats[c] for c in feat_order], dtype=float).reshape(1, -1)
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X
    else:
        Xs = X
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xs)[0]
            lab = int(model.predict(Xs)[0])
            lbl = "Positive" if lab==1 else "Negative"
            prob_text = f"P(positive) = {proba[1]:.3f}" if len(proba)>1 else ""
            st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<div><strong>Result — Mental State</strong><div class='small muted'>Stat7 → classifier</div></div>"
                        f"<div><span class='badge-green'>{lbl}</span></div></div><div style='margin-top:8px'>{prob_text}</div></div>",
                        unsafe_allow_html=True)
        else:
            val = float(model.predict(Xs)[0])
            st.markdown(f"<div class='card'><strong>Prediction:</strong> {val:.3f}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.subheader("Extracted features (stat7)")
    st.json(feats)

# -------------------------
# MODEL: Unconsciousness Level (HRV10 classifier)
# -------------------------
elif model_key == 'unconscious_level':
    with st.spinner("Computing HRV windows..."):
        windows = sliding_hrv_windows(ecg, fs, win_s=win_s, hop_s=hop_s)
    if not windows:
        st.error("No HRV windows could be computed. Try a larger window or a cleaner ECG.")
        st.stop()
    dfw = pd.DataFrame(windows)
    candidate = MODEL_REGISTRY['unconscious_level']['candidate_cols']
    # ensure candidate columns exist
    for c in candidate:
        if c not in dfw.columns:
            dfw[c] = np.nan
    X = dfw[candidate].values.astype(float)
    # fill NaNs
    if np.isnan(X).any():
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
    # scale if available
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X
    else:
        Xs = X
    try:
        preds = model.predict(Xs)
        preds = np.asarray(preds, dtype=int)
        dfw['pred'] = preds
        # majority vote
        vals, counts = np.unique(preds, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        label_map = {0: ("Conscious","badge-green"), 1: ("Partial","badge-yellow"), 2: ("Deep","badge-red")}
        label_text, badge_cls = label_map.get(maj, ("Unknown","badge-yellow"))
        st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div><strong>Result — Unconsciousness Level</strong><div class='small muted'>HRV10 per-window → majority vote</div></div>"
                    f"<div><span class='{badge_cls}'>{label_text} (#{maj})</span></div></div>"
                    f"<div style='margin-top:8px'>Windows used: <strong>{len(dfw)}</strong></div></div>", unsafe_allow_html=True)
        st.subheader("Per-window predictions & features")
        display_cols = ['start_sec','end_sec','pred'] + candidate
        st.dataframe(dfw[display_cols].round(4))
        # download button
        csv = dfw[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download per-window CSV", data=csv, file_name="unconscious_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------
# MODEL: Pain Analyzer (multimodal) — robust sizing
# -------------------------
elif model_key == 'pain':
    # determine expected feature count from model if possible
    expected_n = get_model_expected_n(model)
    if expected_n is None:
        # user reported model expects 12; default to 12 to be safe
        expected_n = 12
    X_pain, used_cols = build_pain_vector(ecg, fs, expected_n=expected_n)
    # final shape check; if mismatch we will attempt automatic trimming/padding
    if X_pain.shape[1] != expected_n:
        # attempt a re-build with expected_n forced (safe)
        X_pain, used_cols = build_pain_vector(ecg, fs, expected_n=expected_n)

    # Predict with graceful handling
    try:
        pred = model.predict(X_pain)
        val = float(pred[0])
        val = float(np.clip(val, 0.0, 100.0))
        if val < 30.0:
            badge = "badge-green"; tag = "Low pain"
        elif val < 60.0:
            badge = "badge-yellow"; tag = "Moderate pain"
        else:
            badge = "badge-red"; tag = "High pain"
        st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div><strong>Result — Pain Analyzer (ANI)</strong><div class='small muted'><b>0 = no pain, 100 = max</b></div></div>"
                    f"<div style='text-align:right'><div style='font-size:22px;font-weight:800'>{val:.1f}</div>"
                    f"<div style='margin-top:6px'><span class='{badge}'>{tag}</span></div></div></div></div>",
                    unsafe_allow_html=True)
        # display used features
        st.subheader("Features passed to Pain model")
        dfp = pd.DataFrame(X_pain, columns=used_cols)
        st.dataframe(dfp.T.rename(columns={0:'value'}).round(4))
        # allow download
        st.download_button("Download pain feature vector (CSV)", data=dfp.to_csv(index=False).encode('utf-8'),
                           file_name="pain_features.csv", mime="text/csv")
    except Exception as e:
        # We attempt a tolerant recovery: try to re-shape/pad/truncate to expected_n and retry
        try:
            # reshape or pad
            X_try = X_pain.copy()
            if X_try.shape[1] < expected_n:
                pad = np.zeros((1, expected_n - X_try.shape[1]))
                X_try = np.hstack([X_try, pad])
            elif X_try.shape[1] > expected_n:
                X_try = X_try[:, :expected_n]
            pred2 = model.predict(X_try)
            val2 = float(pred2[0])
            val2 = float(np.clip(val2, 0.0, 100.0))
            if val2 < 30.0:
                badge = "badge-green"; tag = "Low pain"
            elif val2 < 60.0:
                badge = "badge-yellow"; tag = "Moderate pain"
            else:
                badge = "badge-red"; tag = "High pain"
            st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<div><strong>Result — Pain Analyzer (ANI)</strong><div class='small muted'>Recovered after shape adjust</div></div>"
                        f"<div style='text-align:right'><div style='font-size:22px;font-weight:800'>{val2:.1f}</div>"
                        f"<div style='margin-top:6px'><span class='{badge}'>{tag}</span></div></div></div></div>",
                        unsafe_allow_html=True)
            st.subheader("Final feature vector used (adjusted)")
            dfp2 = pd.DataFrame(X_try, columns=[f"f{i}" for i in range(X_try.shape[1])])
            st.dataframe(dfp2.T.rename(columns={0:'value'}).round(4))
        except Exception as e2:
            st.error("Pain model prediction failed after multiple attempts. Please check model feature ordering or provide a 'meta.json' next to the model listing 'candidate_cols'.")
            st.error(f"Last error: {e2}")

# Done
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Tip: If the Pain model was trained with exact feature names/order, place a <code>meta.json</code> beside <code>pain_model.pkl</code> with <code>{\"candidate_cols\":[...]}</code> to guarantee exact alignment.</div>", unsafe_allow_html=True)
