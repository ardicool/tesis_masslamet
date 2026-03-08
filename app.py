import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Kiln Rolling Window Engine", layout="wide")
st.title("Kiln Rolling Window Engine (Interactive)")

# ======================================================
# LOCAL FUNCTION
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


# ======================================================
# LOAD PREPROCESSOR
# ======================================================
@st.cache_resource
def load_preprocessor():
    return joblib.load("window_preprocessor.joblib")

pre = load_preprocessor()

time_col     = pre["time_col"]
process_cols = pre["process_cols"]
lsf_col      = pre["lsf_col"]
fcao_col     = pre["fcao_col"]
winsor_param = pre["winsor_param"]
lag_map_orig = pre["lag_map"]
window_map_orig = pre["window_map"]

# ======================================================
# MODE SELECTION
# ======================================================
mode = st.radio(
    "Mode Operasi",
    ["Industrial Mode (Lag Asli)", "Demo Mode (Lag Adaptif)"]
)

# ======================================================
# INITIAL DATA (SET ONCE ONLY)
# ======================================================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame({
        time_col: pd.date_range("2025-01-01", periods=10, freq="H"),
        process_cols[0]: np.zeros(10),
        process_cols[1]: np.zeros(10),
        process_cols[2]: np.zeros(10),
        process_cols[3]: np.zeros(10),
        lsf_col: np.zeros(10),
        fcao_col: np.nan
    })

st.subheader("Input Data")

edited_df = st.data_editor(
    st.session_state.data,
    key="kiln_editor",
    num_rows="dynamic",
    use_container_width=True
)

# ======================================================
# PROCESS BUTTON
# ======================================================
if st.button("Run Rolling Window Engine"):

    df = edited_df.copy()

    # ===============================
    # ADAPTIVE LAG
    # ===============================
    lag_map = lag_map_orig.copy()
    window_map = window_map_orig.copy()

    max_lag = max(lag_map.values())

    if mode == "Industrial Mode (Lag Asli)":
        if len(df) <= max_lag:
            st.error(f"Industrial Mode membutuhkan > {max_lag} baris data.")
            st.stop()
    else:
        for col in process_cols:
            lag_map[col] = min(1, len(df)-1)
            window_map[col] = min(2, len(df))
        st.warning("Demo Mode aktif: Lag disesuaikan untuk data kecil.")

    # ===============================
    # WINSOR
    # ===============================
    for col in process_cols:
        p1, p99 = winsor_param[col]
        df[col] = df[col].clip(lower=p1, upper=p99)

    # ===============================
    # LSF FORWARD FILL
    # ===============================
    df[lsf_col] = df[lsf_col].ffill()

    # ===============================
    # FCaOX INC
    # ===============================
    df["FCaOX_Inc"] = compute_fcaox_inc(df[fcao_col])

    # ===============================
    # ROLLING ENGINE
    # ===============================
    rows = []
    last_quality_idx = None
    last_mean_map = {}

    for idx in range(len(df)):

        diff_vals = {}

        for col in process_cols:

            lag = lag_map[col]
            window = window_map[col]

            end = idx - lag
            start = end - window + 1

            if end < 0:
                diff_vals[f"{col}_diff_mean"] = 0
                continue

            if start < 0:
                win = df.loc[:end, col]
            else:
                win = df.loc[start:end, col]

            mean_T = win.mean()

            if last_quality_idx is None or col not in last_mean_map:
                diff = 0
            else:
                diff = mean_T - last_mean_map[col]

            diff_vals[f"{col}_diff_mean"] = diff

        # UPDATE BASELINE
        if pd.notna(df.loc[idx, fcao_col]):

            last_quality_idx = idx
            new_mean_map = {}

            for col in process_cols:

                lag = lag_map[col]
                window = window_map[col]

                end = idx - lag
                start = end - window + 1

                if end < 0:
                    continue

                if start < 0:
                    win = df.loc[:end, col]
                else:
                    win = df.loc[start:end, col]

                new_mean_map[col] = win.mean()

            last_mean_map = new_mean_map

        row = {
            time_col: df.loc[idx, time_col],
            **diff_vals,
            "LSF": df.loc[idx, lsf_col],
            "FCaOX_Inc": df.loc[idx, "FCaOX_Inc"],
            "FCaOX": df.loc[idx, fcao_col]
        }

        rows.append(row)

    final_df = pd.DataFrame(rows)

    ordered_cols = (
        [time_col] +
        [f"{c}_diff_mean" for c in process_cols] +
        ["LSF", "FCaOX_Inc", "FCaOX"]
    )

    final_df = final_df[ordered_cols]

    st.success("Processing selesai")
    st.dataframe(final_df, use_container_width=True)

    # ===============================
    # DOWNLOAD
    # ===============================
    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Data Kiln_step1.csv",
        data=csv_buffer.getvalue(),
        file_name="Data Kiln_step1.csv",
        mime="text/csv"
    )