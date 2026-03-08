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

# ============================================================
# HELPER: compute_fcaox_inc (lokal, anti pickle error)
# ============================================================
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

page = st.radio(
    "Navigasi",
    ["⚙️ Step 1 — Preprocessing", "🔮 Step 2 — Prediksi", "📋 Step 3 — Log Prediksi"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<hr style='border-color:#30363d; margin:.75rem 0 1.5rem 0'>", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — PREPROCESSING
# ============================================================
if page == "⚙️ Step 1 — Preprocessing":

    st.markdown("<h3 class='section-header'>Upload File</h3>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        joblib_file = st.file_uploader("📦 window_preprocessor.joblib", type=["joblib"])
    with col_r:
        csv_file = st.file_uploader("📄 Data Kiln_step0.csv", type=["csv"])

    st.markdown("<hr style='border-color:#30363d; margin:1rem 0'>", unsafe_allow_html=True)

    # Load preprocessor
    pre = None
    if joblib_file:
        try:
            pre = safe_load_joblib(joblib_file)
            st.session_state["pre"] = pre
            st.markdown(f"""
            <div class='success-box'>
                ✅ Preprocessor loaded<br>
                &nbsp;&nbsp;• Kolom proses : {', '.join(pre['process_cols'])}<br>
                &nbsp;&nbsp;• Lag map      : { {k: v for k, v in pre['lag_map'].items()} }<br>
                &nbsp;&nbsp;• Window map   : { {k: v for k, v in pre['window_map'].items()} }
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div class='warn-box'>⚠️ Gagal load preprocessor: {e}</div>", unsafe_allow_html=True)
    elif st.session_state["pre"]:
        pre = st.session_state["pre"]
        st.markdown("<div class='info-box'>ℹ️ Menggunakan preprocessor dari session sebelumnya.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>📂 Upload <b>window_preprocessor.joblib</b>.</div>", unsafe_allow_html=True)

    # Load CSV
    df_input = None
    if csv_file:
        try:
            df_input = pd.read_csv(csv_file)
            st.markdown(f"""
            <div class='success-box'>
                ✅ Data Step-0 loaded: <b>{csv_file.name}</b> — {len(df_input):,} rows × {len(df_input.columns)} kolom<br>
                &nbsp;&nbsp;• Kolom: {', '.join(df_input.columns.tolist())}
            </div>""", unsafe_allow_html=True)
            with st.expander("👁️ Preview Data Step-0"):
                st.dataframe(df_input.head(20), use_container_width=True, height=260)
        except Exception as e:
            st.markdown(f"<div class='warn-box'>⚠️ Gagal baca CSV: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>📂 Upload <b>Data Kiln_step0.csv</b>.</div>", unsafe_allow_html=True)

    # Metrics preview
    if df_input is not None:
        st.markdown("<h3 class='section-header'>Info Data</h3>", unsafe_allow_html=True)
        m = st.columns(4)
        for i, (lbl, val) in enumerate([
            ("Total Rows",    f"{len(df_input):,}"),
            ("FCaOX Samples", f"{df_input.iloc[:,6].notna().sum():,}"),
            ("LSF Coverage",  f"{df_input.iloc[:,5].notna().mean()*100:.1f}%"),
            ("Kolom",         f"{len(df_input.columns)}"),
        ]):
            with m[i]:
                st.markdown(f"<div class='metric-card'><h4>{lbl}</h4><p>{val}</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ready = (pre is not None) and (df_input is not None)
    run_btn = st.button("🚀 Generate Step-1", type="primary", disabled=not ready, use_container_width=True)

    if not ready:
        st.markdown("<div class='info-box'>ℹ️ Upload kedua file untuk mengaktifkan Generate.</div>", unsafe_allow_html=True)

    if run_btn and ready:
        st.markdown("<h3 class='section-header'>⚙️ Processing...</h3>", unsafe_allow_html=True)
        pbar = st.progress(0)
        try:
            step1 = run_rolling_window(df_input, pre, progress_bar=pbar)
            st.session_state["step1_df"] = step1
            st.markdown(f"<div class='success-box'>✅ Step-1 selesai — {len(step1):,} rows. Lanjut ke tab Prediksi.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # Show result + download if available
    if st.session_state["step1_df"] is not None:
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


# ============================================================
# PAGE 2 — PREDIKSI
# ============================================================
elif page == "🔮 Step 2 — Prediksi":

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

    if scaler_file and model_inc_file and model_abs_file:
        try:
            bundle    = joblib.load(scaler_file)
            scaler    = bundle["scaler"]
            scale_cols = bundle["scale_cols"]
            model_inc = joblib.load(model_inc_file)
            model_abs = joblib.load(model_abs_file)
            st.markdown("<div class='success-box'>✅ Semua model berhasil di-load.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div class='warn-box'>⚠️ Gagal load model: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>📂 Upload ketiga file model di atas.</div>", unsafe_allow_html=True)

    models_ready = all([scaler, scale_cols, model_inc, model_abs])

    # ---- MODE SELECTOR ----
    st.markdown("<h3 class='section-header'>Mode Prediksi</h3>", unsafe_allow_html=True)
    mode = st.radio("", ["📝 Input Manual", "📊 Batch dari Step-1"], horizontal=True, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#30363d; margin:1rem 0'>", unsafe_allow_html=True)

    # ==========================
    # MODE A: MANUAL INPUT
    # ==========================
    if mode == "📝 Input Manual":

        st.markdown("<h3 class='section-header'>Parameter Input</h3>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Process Features (dari Step-1)**")
            Torsi = st.number_input("Torsi Motor Kiln_diff_mean", value=0.0, format="%.5f")
            Arus  = st.number_input("Arus Motor Kiln_diff_mean",  value=0.0, format="%.5f")
            Nox   = st.number_input("Nox IKGA_diff_mean",         value=0.0, format="%.5f")
            Suhu  = st.number_input("Suhu Calciner_diff_mean",    value=0.0, format="%.5f")

        with col_b:
            st.markdown("**Quality & Reference**")
            LSF_current = st.number_input("LSF (saat ini)",            value=95.0, format="%.3f")
            LSF_last    = st.number_input("LSF_lag1 (LSF sebelumnya)", value=94.5, format="%.3f")
            FCaOX_last  = st.number_input("FCaOX_lag1 (FCaOX aktual terakhir)", value=1.2, format="%.5f")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict FCaOX", type="primary", disabled=not models_ready, use_container_width=True)

        if not models_ready:
            st.markdown("<div class='info-box'>ℹ️ Upload ketiga file model untuk mengaktifkan prediksi.</div>", unsafe_allow_html=True)

        if predict_btn and models_ready:
            if Suhu < 0:
                st.markdown("<div class='warn-box'>⚠️ Suhu tidak boleh negatif.</div>", unsafe_allow_html=True)
            else:
                input_dict = {
                    "Torsi Motor Kiln_diff_mean": Torsi,
                    "Arus Motor Kiln_diff_mean":  Arus,
                    "Nox IKGA_diff_mean":          Nox,
                    "Suhu Calciner_diff_mean":     Suhu,
                    "LSF":                         LSF_current,
                    "LSF_lag1":                    LSF_last
                }
                input_df = pd.DataFrame([input_dict]).reindex(columns=scale_cols)

                if input_df.isnull().any().any():
                    st.markdown(f"<div class='warn-box'>⚠️ Kolom mismatch.<br>Expected: {scale_cols}<br>Got: {list(input_dict.keys())}</div>", unsafe_allow_html=True)
                else:
                    X_scaled       = scaler.transform(input_df)
                    fcaox_inc_pred = model_inc.predict(X_scaled)[0]
                    input_abs      = pd.DataFrame([{"FCaOX_Inc_pred": fcaox_inc_pred, "FCaOX_lag1": FCaOX_last}])
                    fcaox_pred     = model_abs.predict(input_abs)[0]

                    # Display result
                    st.markdown("<h3 class='section-header'>Hasil Prediksi</h3>", unsafe_allow_html=True)
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"""
                        <div class='pred-card-inc'>
                            <h4>FCaOX Increment</h4>
                            <p>{round(float(fcaox_inc_pred), 5)}</p>
                        </div>""", unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div class='pred-card-abs'>
                            <h4>FCaOX Absolut</h4>
                            <p>{round(float(fcaox_pred), 5)}</p>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if fcaox_pred > 2.0:
                        st.markdown("<div class='error-box'>🚨 FCaOX Melebihi Batas Operasi (> 2.0)</div>", unsafe_allow_html=True)
                    elif fcaox_pred < 0:
                        st.markdown("<div class='warn-box'>⚠️ Prediksi Negatif – Periksa Input</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='success-box'>✅ FCaOX Dalam Rentang Normal</div>", unsafe_allow_html=True)

                    # Logging
                    log_row = pd.DataFrame([{
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": "manual",
                        **input_dict,
                        "FCaOX_lag1":      FCaOX_last,
                        "FCaOX_Inc_pred":  round(float(fcaox_inc_pred), 5),
                        "FCaOX_pred":      round(float(fcaox_pred), 5),
                        "FCaOX_actual":    np.nan
                    }])
                    st.session_state["prediction_log"] = pd.concat(
                        [st.session_state["prediction_log"], log_row], ignore_index=True
                    )

    # ==========================
    # MODE B: BATCH FROM STEP-1
    # ==========================
    else:
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


# ============================================================
# PAGE 3 — LOG
# ============================================================
elif page == "📋 Step 3 — Log Prediksi":

    st.markdown("<h3 class='section-header'>Log Prediksi</h3>", unsafe_allow_html=True)

    log = st.session_state.get("prediction_log")

    if log is not None and len(log) > 0:
        m = st.columns(4)
        for i, (lbl, val) in enumerate([
            ("Total Prediksi",  f"{len(log):,}"),
            ("Alarm (>2.0)",    f"{(log['FCaOX_pred'] > 2.0).sum():,}"),
            ("Negatif",         f"{(log['FCaOX_pred'] < 0).sum():,}"),
            ("Normal",          f"{((log['FCaOX_pred'] >= 0) & (log['FCaOX_pred'] <= 2.0)).sum():,}"),
        ]):
            with m[i]:
                st.markdown(f"<div class='metric-card'><h4>{lbl}</h4><p>{val}</p></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(log, use_container_width=True, height=420)

        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            st.download_button(
                "⬇️ Download Log (.csv)",
                data=log.to_csv(index=False).encode("utf-8"),
                file_name="prediction_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear Log", use_container_width=True):
                st.session_state["prediction_log"] = pd.DataFrame()
                st.rerun()

        if "FCaOX_pred" in log.columns:
            with st.expander("📈 Trend FCaOX Prediksi"):
                st.line_chart(log["FCaOX_pred"].reset_index(drop=True), height=260)

    else:
        st.markdown("<div class='info-box'>ℹ️ Belum ada prediksi. Jalankan prediksi di Step 2.</div>", unsafe_allow_html=True)