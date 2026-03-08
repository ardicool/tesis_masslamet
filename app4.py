import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import io
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Kiln FCaOX System",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.stApp { background-color: #0d1117; color: #e6edf3; }

.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d; border-radius: 10px;
    padding: 1rem 1.5rem; text-align: center;
}
.metric-card h4 { color: #8b949e; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0; }
.metric-card p  { color: #f0883e; font-size: 1.7rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0; }

.pred-card-inc {
    background: linear-gradient(135deg, #0c1e30, #112233);
    border: 2px solid #1f6feb; border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.pred-card-abs {
    background: linear-gradient(135deg, #0d2818, #0f3020);
    border: 2px solid #238636; border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.pred-card-inc h4, .pred-card-abs h4 { color: #8b949e; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 8px 0; }
.pred-card-inc p  { color: #58a6ff; font-size: 2.2rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0; }
.pred-card-abs p  { color: #3fb950; font-size: 2.2rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0; }

.section-header {
    border-left: 4px solid #f0883e; padding-left: 12px;
    margin: 1.5rem 0 1rem 0; font-family: 'Space Mono', monospace; color: #e6edf3;
}
.success-box { background:#0d2818; border:1px solid #238636; border-radius:8px; padding:.75rem 1rem; color:#3fb950; font-family:'Space Mono',monospace; font-size:.85rem; }
.info-box    { background:#0c1e30; border:1px solid #1f6feb; border-radius:8px; padding:.75rem 1rem; color:#58a6ff; font-family:'Space Mono',monospace; font-size:.85rem; }
.warn-box    { background:#2d1f00; border:1px solid #d29922; border-radius:8px; padding:.75rem 1rem; color:#e3b341; font-family:'Space Mono',monospace; font-size:.85rem; }
.error-box   { background:#2d0f0f; border:1px solid #f85149; border-radius:8px; padding:.75rem 1rem; color:#f85149; font-family:'Space Mono',monospace; font-size:.85rem; }
div[data-testid="stDataFrame"] { border:1px solid #30363d; border-radius:8px; }

/* Nav pills */
div[data-testid="stHorizontalBlock"] button { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def compute_fcaox_inc(series):
    inc, last_valid = [], None
    for val in series:
        if pd.notna(val):
            inc.append(np.nan if last_valid is None else val - last_valid)
            last_valid = val
        else:
            inc.append(np.nan)
    return inc

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "compute_fcaox_inc":
            return compute_fcaox_inc
        return super().find_class(module, name)

def safe_load_joblib(file_obj):
    file_obj.seek(0)
    try:
        result = joblib.load(file_obj)
    except AttributeError:
        file_obj.seek(0)
        result = _SafeUnpickler(file_obj).load()
    if isinstance(result, dict) and "compute_fcaox_inc" in result:
        result["compute_fcaox_inc"] = compute_fcaox_inc
    return result

# ============================================================
# HELPER: rolling window engine
# ============================================================
def run_rolling_window(df_raw, pre, progress_bar=None):
    df = df_raw.copy()
    time_col     = pre["time_col"]
    process_cols = pre["process_cols"]
    lsf_col      = pre["lsf_col"]
    fcao_col     = pre["fcao_col"]
    winsor_param = pre["winsor_param"]
    lag_map      = pre["lag_map"]
    window_map   = pre["window_map"]

    for col in process_cols:
        p1, p99 = winsor_param[col]
        df[col] = df[col].clip(lower=p1, upper=p99)

    df[lsf_col] = df[lsf_col].ffill()
    df["FCaOX_Inc"] = compute_fcaox_inc(df[fcao_col])

    rows, last_quality_idx, last_mean_map = [], None, {}
    total = len(df)

    for idx in range(total):
        diff_vals = {}
        for col in process_cols:
            lag    = lag_map.get(col, 0)
            window = window_map.get(col, 1)
            end    = idx - lag
            start  = end - window + 1
            if end < 0:
                diff_vals[f"{col}_diff_mean"] = 0
                continue
            win    = df.loc[max(0, start):end, col]
            mean_T = win.mean()
            diff   = 0 if (last_quality_idx is None or col not in last_mean_map) else mean_T - last_mean_map[col]
            diff_vals[f"{col}_diff_mean"] = diff

        if pd.notna(df.loc[idx, fcao_col]):
            last_quality_idx = idx
            new_mean_map = {}
            for col in process_cols:
                lag    = lag_map.get(col, 0)
                window = window_map.get(col, 1)
                end    = idx - lag
                start  = end - window + 1
                if end < 0: continue
                new_mean_map[col] = df.loc[max(0, start):end, col].mean()
            last_mean_map = new_mean_map

        rows.append({time_col: df.loc[idx, time_col], **diff_vals,
                     "LSF": df.loc[idx, lsf_col],
                     "FCaOX_Inc": df.loc[idx, "FCaOX_Inc"],
                     "FCaOX": df.loc[idx, fcao_col]})

        if progress_bar and idx % max(1, total // 100) == 0:
            progress_bar.progress(int(idx / total * 100), text=f"Processing... {idx:,}/{total:,} rows")

    if progress_bar:
        progress_bar.progress(100, text="Done!")

    final_df = pd.DataFrame(rows)
    ordered  = [time_col] + [f"{c}_diff_mean" for c in process_cols] + ["LSF", "FCaOX_Inc", "FCaOX"]
    return final_df[[c for c in ordered if c in final_df.columns]]

# ============================================================
# HELPER: run batch prediction on step1 df
# ============================================================
def run_batch_prediction(step1_df, scaler, scale_cols, model_inc, model_abs):
    """Predict FCaOX untuk setiap baris di step1_df yang punya cukup data."""
    results = []

    # Tambah LSF_lag1 = LSF shift(1)
    df = step1_df.copy()
    df["LSF_lag1"]    = df["LSF"].shift(1)
    df["FCaOX_lag1"]  = df["FCaOX"].shift(1)

    feature_cols = [
        "Torsi Motor Kiln_diff_mean",
        "Arus Motor Kiln_diff_mean",
        "Nox IKGA_diff_mean",
        "Suhu Calciner_diff_mean",
        "LSF",
        "LSF_lag1"
    ]

    time_col = df.columns[0]

    for idx, row in df.iterrows():
        # Skip baris yang belum punya lag
        if pd.isna(row.get("LSF_lag1")) or pd.isna(row.get("FCaOX_lag1")):
            continue

        input_df = pd.DataFrame([row[scale_cols]])

        if input_df.isnull().any().any():
            continue

        try:
            X_scaled        = scaler.transform(input_df)
            fcaox_inc_pred  = model_inc.predict(X_scaled)[0]
            input_abs       = pd.DataFrame([{"FCaOX_Inc_pred": fcaox_inc_pred, "FCaOX_lag1": row["FCaOX_lag1"]}])
            fcaox_pred      = model_abs.predict(input_abs)[0]

            results.append({
                "timestamp":       row[time_col],
                "FCaOX_Inc_pred":  round(float(fcaox_inc_pred), 5),
                "FCaOX_pred":      round(float(fcaox_pred), 5),
                "FCaOX_actual":    row.get("FCaOX", np.nan),
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# ============================================================
# SESSION STATE INIT
# ============================================================
for key in ["step1_df", "pre", "prediction_log"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["prediction_log"] is None:
    st.session_state["prediction_log"] = pd.DataFrame()

# ============================================================
# NAVIGATION
# ============================================================
st.markdown("""
<h1 style='color:#f0883e; margin-bottom:4px'>🔥 Kiln FCaOX Prediction System</h1>
<p style='color:#8b949e; margin-top:0; font-size:0.9rem'>Preprocessing → Feature Engineering → Prediksi FCaOX</p>
<hr style='border-color:#30363d; margin:.75rem 0 1.25rem 0'>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#30363d; margin:.75rem 0 1.5rem 0'>", unsafe_allow_html=True)
col_l, col_r = st.columns(2)

with col_l:
    joblib_path = "window_preprocessor.joblib"
    window_preprocessor = joblib.load(joblib_path)
    st.success(f"Loaded: {joblib_path}")

with col_r:
    csv_path = "Data Kiln_step0.csv"
    df = pd.read_csv(csv_path)
    st.success(f"Loaded: {csv_path}")

df_input = df
pre = window_preprocessor

st.markdown("<h3 class='section-header'>⚙️ Processing...</h3>", unsafe_allow_html=True)
pbar = st.progress(0)
try:
    step1 = run_rolling_window(df_input, pre, progress_bar=pbar)
    st.session_state["step1_df"] = step1
    st.markdown(f"<div class='success-box'>✅ Step-1 selesai — {len(step1):,} rows. Lanjut ke tab Prediksi.</div>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error: {e}")

step1 = st.session_state["step1_df"]
st.markdown("<h3 class='section-header'>Output Step-1</h3>", unsafe_allow_html=True)

m2 = st.columns(4)
for i, (lbl, val) in enumerate([
    ("Rows",           f"{len(step1):,}"),
    ("Kolom",          f"{len(step1.columns)}"),
    ("FCaOX_Inc ≠ NaN",f"{step1['FCaOX_Inc'].notna().sum():,}"),
    ("FCaOX ≠ NaN",    f"{step1['FCaOX'].notna().sum():,}"),
]):
    with m2[i]:
        st.markdown(f"<div class='metric-card'><h4>{lbl}</h4><p>{val}</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(step1.head(50), use_container_width=True, height=340)

st.download_button(
            "⬇️ Download Data Kiln_step1.csv",
            data=step1.to_csv(index=False).encode("utf-8"),
            file_name="Data Kiln_step1.csv",
            mime="text/csv",
            use_container_width=True
        )

st.Page == "🔮 Step 2 — Prediksi"

st.markdown("<h3 class='section-header'>Upload Model Files</h3>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    scaler_file = st.file_uploader("📦 scaler_process_lsf.joblib", type=["joblib"])
with c2:
    model_inc_file = st.file_uploader("🤖 xgb_fcaox_increment.joblib", type=["joblib"])
with c3:
    model_abs_file = st.file_uploader("🤖 fcao_abs_model.joblib", type=["joblib"])

st.markdown("<hr style='border-color:#30363d; margin:1rem 0'>", unsafe_allow_html=True)

# Load models
scaler, scale_cols, model_inc, model_abs = None, None, None, None


try:
    bundle    = joblib.load("scaler_process_lsf.joblib")
    scaler    = bundle["scaler"]
    scale_cols = bundle["scale_cols"]
    model_inc = joblib.load("xgb_fcaox_increment.joblib")
    model_abs = joblib.load("fcao_abs_model.joblib")
    st.markdown("<div class='success-box'>✅ Semua model berhasil di-load.</div>", unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"<div class='warn-box'>⚠️ Gagal load model: {e}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>📂 Upload ketiga file model di atas.</div>", unsafe_allow_html=True)

models_ready = all([scaler, scale_cols, model_inc, model_abs])

# ---- MODE SELECTOR ----
st.markdown("<h3 class='section-header'>Mode Prediksi</h3>", unsafe_allow_html=True)
mode = st.radio("", ["📊 Batch dari Step-1"], horizontal=True, label_visibility="collapsed")

st.markdown("<hr style='border-color:#30363d; margin:1rem 0'>", unsafe_allow_html=True)

step1 = st.session_state.get("step1_df")

if step1 is None:
    # Allow manual upload of step1 CSV
    st.markdown("<div class='info-box'>ℹ️ Belum ada data Step-1 dari preprocessing. Anda bisa upload manual di bawah.</div>", unsafe_allow_html=True)
    uploaded_step1 = st.file_uploader("📄 Upload Data Kiln_step1.csv (opsional)", type=["csv"])
    if uploaded_step1:
        step1 = pd.read_csv(uploaded_step1)
        st.session_state["step1_df"] = step1

if step1 is not None:
    st.markdown(f"<div class='success-box'>✅ Step-1 tersedia: {len(step1):,} rows</div>", unsafe_allow_html=True)
    with st.expander("👁️ Preview Step-1"):
        st.dataframe(step1.head(20), use_container_width=True, height=250)

    batch_btn = st.button("🔮 Run Batch Prediction", type="primary", disabled=not models_ready, use_container_width=True)

    if not models_ready:
        st.markdown("<div class='info-box'>ℹ️ Upload ketiga file model untuk mengaktifkan prediksi.</div>", unsafe_allow_html=True)

    if batch_btn and models_ready:
        with st.spinner("Running batch prediction..."):
            try:
                result_df = run_batch_prediction(step1, scaler, scale_cols, model_inc, model_abs)

                st.markdown("<h3 class='section-header'>Hasil Batch Prediksi</h3>", unsafe_allow_html=True)

                m = st.columns(4)
                n_alarm = (result_df["FCaOX_pred"] > 2.0).sum()
                n_neg   = (result_df["FCaOX_pred"] < 0).sum()
                n_ok    = len(result_df) - n_alarm - n_neg
                for i, (lbl, val, color) in enumerate([
                    ("Total Prediksi", str(len(result_df)), "#f0883e"),
                    ("Normal ✅",       str(n_ok),          "#3fb950"),
                    ("Alarm 🚨",        str(n_alarm),       "#f85149"),
                    ("Negatif ⚠️",      str(n_neg),         "#e3b341"),
                ]):
                    with m[i]:
                        st.markdown(f"<div class='metric-card'><h4>{lbl}</h4><p style='color:{color}'>{val}</p></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True, height=380)

                # Chart
                if "FCaOX_actual" in result_df.columns:
                    with st.expander("📈 Chart Prediksi vs Aktual"):
                        chart_df = result_df[["FCaOX_pred", "FCaOX_actual"]].dropna()
                        st.line_chart(chart_df, height=280)

                # Append to log
                log_batch = result_df.copy()
                log_batch["mode"] = "batch"
                st.session_state["prediction_log"] = pd.concat(
                    [st.session_state["prediction_log"], log_batch], ignore_index=True
                )

                st.download_button(
                    "⬇️ Download Hasil Prediksi (.csv)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="prediction_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error batch prediction: {e}")
else:
    st.markdown("<div class='info-box'>ℹ️ Jalankan preprocessing di Step 1 terlebih dahulu, atau upload Step-1 CSV di atas.</div>", unsafe_allow_html=True)