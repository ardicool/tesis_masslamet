import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import pickle
import types

# ======================================================
# CUSTOM UNPICKLER — fix "Can't get attribute 'compute_fcaox_inc'"
# Fungsi didefinisikan dulu di sini, lalu di-inject ke pre setelah load
# ======================================================
def compute_fcaox_inc(series):
    inc = []
    last_valid = None
    for val in series:
        if pd.notna(val):
            if last_valid is None:
                inc.append(np.nan)
            else:
                inc.append(val - last_valid)
            last_valid = val
        else:
            inc.append(np.nan)
    return inc

class _SafeUnpickler(pickle.Unpickler):
    """Ganti fungsi yang tidak bisa di-resolve dengan versi lokal."""
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
    # Selalu override fungsinya agar pakai versi lokal
    result["compute_fcaox_inc"] = compute_fcaox_inc
    return result


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Kiln Step-1 Generator",
    page_icon="🔥",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-card h4 { color: #8b949e; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0; }
    .metric-card p { color: #f0883e; font-size: 1.8rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0; }
    .section-header {
        border-left: 4px solid #f0883e;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Space Mono', monospace;
        color: #e6edf3;
    }
    .success-box {
        background: #0d2818; border: 1px solid #238636;
        border-radius: 8px; padding: 0.75rem 1rem;
        color: #3fb950; font-family: 'Space Mono', monospace; font-size: 0.85rem;
    }
    .info-box {
        background: #0c1e30; border: 1px solid #1f6feb;
        border-radius: 8px; padding: 0.75rem 1rem;
        color: #58a6ff; font-family: 'Space Mono', monospace; font-size: 0.85rem;
    }
    .warn-box {
        background: #2d1f00; border: 1px solid #d29922;
        border-radius: 8px; padding: 0.75rem 1rem;
        color: #e3b341; font-family: 'Space Mono', monospace; font-size: 0.85rem;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


# ======================================================
# ROLLING WINDOW ENGINE
# ======================================================
def run_rolling_window(df_raw, pre):
    df = df_raw.copy()
    time_col     = pre["time_col"]
    process_cols = pre["process_cols"]
    lsf_col      = pre["lsf_col"]
    fcao_col     = pre["fcao_col"]
    winsor_param = pre["winsor_param"]
    lag_map      = pre["lag_map"]
    window_map   = pre["window_map"]
    compute_fn   = pre["compute_fcaox_inc"]

    # Apply winsorizing
    for col in process_cols:
        p1, p99 = winsor_param[col]
        df[col] = df[col].clip(lower=p1, upper=p99)

    # LSF forward fill
    df[lsf_col] = df[lsf_col].ffill()

    # FCaOX increment
    df["FCaOX_Inc"] = compute_fn(df[fcao_col])

    rows = []
    last_quality_idx = None
    last_mean_map = {}

    total = len(df)
    progress = st.progress(0, text="Processing rows...")

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

            if last_quality_idx is None or col not in last_mean_map:
                diff = 0
            else:
                diff = mean_T - last_mean_map[col]

            diff_vals[f"{col}_diff_mean"] = diff

        # Update reference mean hanya saat ada kualitas (FCaOX tidak NaN)
        if pd.notna(df.loc[idx, fcao_col]):
            last_quality_idx = idx
            new_mean_map = {}
            for col in process_cols:
                lag    = lag_map.get(col, 0)
                window = window_map.get(col, 1)
                end    = idx - lag
                start  = end - window + 1
                if end < 0:
                    continue
                win = df.loc[max(0, start):end, col]
                new_mean_map[col] = win.mean()
            last_mean_map = new_mean_map

        row = {
            time_col: df.loc[idx, time_col],
            **diff_vals,
            "LSF":       df.loc[idx, lsf_col],
            "FCaOX_Inc": df.loc[idx, "FCaOX_Inc"],
            "FCaOX":     df.loc[idx, fcao_col]
        }
        rows.append(row)

        if idx % max(1, total // 100) == 0:
            progress.progress(int(idx / total * 100), text=f"Processing... {idx:,}/{total:,} rows")

    progress.progress(100, text="Done!")

    final_df = pd.DataFrame(rows)
    ordered_cols = (
        [time_col] +
        [f"{c}_diff_mean" for c in process_cols] +
        ["LSF", "FCaOX_Inc", "FCaOX"]
    )
    final_df = final_df[[c for c in ordered_cols if c in final_df.columns]]
    return final_df


# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style='color:#f0883e; margin-bottom:0'>🔥 Kiln Step-1 Generator</h1>
<p style='color:#8b949e; font-family:DM Sans; margin-top:4px'>
    Load trained <code>window_preprocessor.joblib</code> + Data Step-0 → Generate Step-1
</p>
<hr style='border-color:#30363d; margin:1rem 0'>
""", unsafe_allow_html=True)

# ======================================================
# UPLOAD SECTION
# ======================================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("<h3 class='section-header'>1️⃣ Upload Preprocessor</h3>", unsafe_allow_html=True)
    joblib_file = st.file_uploader(
        "window_preprocessor.joblib",
        type=["joblib"],
        help="File .joblib hasil training preprocessor"
    )

with col_right:
    st.markdown("<h3 class='section-header'>2️⃣ Upload Data Step-0</h3>", unsafe_allow_html=True)
    csv_file = st.file_uploader(
        "Data Kiln_step0.csv",
        type=["csv"],
        help="Data mentah kiln: Start Time, Torsi Motor Kiln, Arus Motor Kiln, Nox IKGA, Suhu Calciner, LSF, FCaOX"
    )

st.markdown("<hr style='border-color:#30363d; margin:1rem 0'>", unsafe_allow_html=True)

# ======================================================
# STATUS CHECK & LOAD
# ======================================================
pre = None
df_input = None

if joblib_file:
    try:
        pre = safe_load_joblib(joblib_file)
        st.markdown(f"""
        <div class='success-box'>
            ✅ Preprocessor loaded<br>
            &nbsp;&nbsp;• Kolom proses : {', '.join(pre['process_cols'])}<br>
            &nbsp;&nbsp;• Lag map      : { {k: v for k, v in pre['lag_map'].items()} }<br>
            &nbsp;&nbsp;• Window map   : { {k: v for k, v in pre['window_map'].items()} }
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='warn-box'>⚠️ Gagal load preprocessor: {e}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>📂 Upload <b>window_preprocessor.joblib</b> (file hasil training).</div>", unsafe_allow_html=True)

if csv_file:
    try:
        df_input = pd.read_csv(csv_file)
        st.markdown(f"""
        <div class='success-box'>
            ✅ Data Step-0 loaded: <b>{csv_file.name}</b> — {len(df_input):,} rows × {len(df_input.columns)} kolom<br>
            &nbsp;&nbsp;• Kolom: {', '.join(df_input.columns.tolist())}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("👁️ Preview Data Step-0", expanded=False):
            st.dataframe(df_input.head(20), use_container_width=True, height=280)
    except Exception as e:
        st.markdown(f"<div class='warn-box'>⚠️ Gagal baca CSV: {e}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>📂 Upload <b>Data Kiln_step0.csv</b>.</div>", unsafe_allow_html=True)

# ======================================================
# RUN BUTTON
# ======================================================
st.markdown("<br>", unsafe_allow_html=True)
ready = (pre is not None) and (df_input is not None)

run_btn = st.button(
    "🚀 Generate Step-1",
    type="primary",
    disabled=not ready,
    use_container_width=True
)

if not ready:
    st.markdown("<div class='info-box'>ℹ️ Upload kedua file di atas untuk mengaktifkan tombol Generate.</div>", unsafe_allow_html=True)

if run_btn and ready:
    st.markdown("<h3 class='section-header'>⚙️ Processing...</h3>", unsafe_allow_html=True)
    try:
        final_df = run_rolling_window(df_input, pre)
        st.session_state["final_df"] = final_df
        st.markdown("<div class='success-box'>✅ Rolling window selesai! Data Step-1 siap di-download.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error saat processing: {e}")
        st.stop()

# ======================================================
# RESULTS & DOWNLOAD
# ======================================================
if "final_df" in st.session_state:
    final_df = st.session_state["final_df"]

    st.markdown("<hr style='border-color:#30363d; margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header'>📊 Output Step-1</h3>", unsafe_allow_html=True)

    m_cols = st.columns(4)
    for i, (label, val) in enumerate([
        ("Total Rows",         f"{len(final_df):,}"),
        ("Total Kolom",        f"{len(final_df.columns)}"),
        ("FCaOX_Inc Non-null", f"{final_df['FCaOX_Inc'].notna().sum():,}"),
        ("FCaOX Non-null",     f"{final_df['FCaOX'].notna().sum():,}"),
    ]):
        with m_cols[i]:
            st.markdown(f"""<div class='metric-card'><h4>{label}</h4><p>{val}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(final_df.head(50), use_container_width=True, height=380)

    with st.expander("📈 Visualisasi Kolom", expanded=False):
        diff_cols = [c for c in final_df.columns if "_diff_mean" in c]
        sel_col = st.selectbox("Pilih kolom:", diff_cols + ["LSF", "FCaOX_Inc", "FCaOX"])
        st.line_chart(final_df[sel_col].dropna(), height=250)

    st.markdown("<h3 class='section-header'>💾 Download</h3>", unsafe_allow_html=True)
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Data Kiln_step1.csv",
        data=csv_bytes,
        file_name="Data Kiln_step1.csv",
        mime="text/csv",
        use_container_width=True
    )