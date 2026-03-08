import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from scipy.optimize import minimize
import openai


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Kiln FCaOX System",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# GOOGLE SHEET URL
# ============================================================
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1WZcow7B0i8ZXXp7jai_D_tFA-eRCqj3tViZGoKheojQ/export?format=csv&gid=1798580501"


# ============================================================
# LOAD GOOGLE SHEET
# ============================================================
@st.cache_data(ttl=10)
def load_kiln_sheet():

    conn = st.connection("gsheets", type=GSheetsConnection)

    data = conn.read(spreadsheet=GSHEET_URL)

    df = pd.DataFrame(data)

    # convert time
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")

    numeric_cols = [
        "Torsi Motor Kiln",
        "Arus Motor Kiln",
        "Nox IKGA",
        "Suhu Calciner",
        "LSF",
        "FCaOX"
    ]

    for col in numeric_cols:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(" ", "")
            .str.replace(",", ".", regex=False)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Start Time")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
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


# ============================================================
# ROLLING WINDOW ENGINE
# ============================================================
def run_rolling_window(df_raw, pre):

    df = df_raw.copy()

    time_col = pre["time_col"]
    process_cols = pre["process_cols"]
    lsf_col = pre["lsf_col"]
    fcao_col = pre["fcao_col"]
    winsor_param = pre["winsor_param"]
    lag_map = pre["lag_map"]
    window_map = pre["window_map"]

    for col in process_cols:

        p1, p99 = winsor_param[col]

        df[col] = df[col].clip(lower=p1, upper=p99)

    df[lsf_col] = df[lsf_col].ffill()

    df["FCaOX_Inc"] = compute_fcaox_inc(df[fcao_col])

    rows = []
    last_quality_idx = None
    last_mean_map = {}

    total = len(df)

    for idx in range(total):

        diff_vals = {}

        for col in process_cols:

            lag = lag_map.get(col, 0)
            window = window_map.get(col, 1)

            end = idx - lag
            start = end - window + 1

            if end < 0:
                diff_vals[f"{col}_diff_mean"] = 0
                continue

            win = df.loc[max(0, start):end, col]

            mean_T = win.mean()

            diff = 0 if (last_quality_idx is None or col not in last_mean_map) else mean_T - last_mean_map[col]

            diff_vals[f"{col}_diff_mean"] = diff

        if pd.notna(df.loc[idx, fcao_col]):

            last_quality_idx = idx

            new_mean_map = {}

            for col in process_cols:

                lag = lag_map.get(col, 0)
                window = window_map.get(col, 1)

                end = idx - lag
                start = end - window + 1

                if end < 0:
                    continue

                new_mean_map[col] = df.loc[max(0, start):end, col].mean()

            last_mean_map = new_mean_map

        rows.append({
            time_col: df.loc[idx, time_col],
            **diff_vals,
            "LSF": df.loc[idx, lsf_col],
            "FCaOX_Inc": df.loc[idx, "FCaOX_Inc"],
            "FCaOX": df.loc[idx, fcao_col]
        })

    final_df = pd.DataFrame(rows)

    ordered = [time_col] + [f"{c}_diff_mean" for c in process_cols] + ["LSF", "FCaOX_Inc", "FCaOX"]

    return final_df[[c for c in ordered if c in final_df.columns]]


# ============================================================
# UI HEADER
# ============================================================
st.title("🔥 Kiln FCaOX Prediction System")


# ============================================================
# LOAD DATA
# ============================================================
st.subheader("Load Data Google Sheet")

df = load_kiln_sheet()

st.success(f"Data loaded : {len(df):,} rows")

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(50), use_container_width=True)

with st.expander("Dtype Check"):
    st.write(df.dtypes)

#fungsi preskriptif
def prescribe_control(last_row, scaler, scale_cols, model_inc, model_abs, target):

    # validasi lag FCaOX
    #if pd.isna(last_row["FCaOX"]):
     #   raise ValueError("FCaOX terakhir masih NaN. Prescriptive tidak dapat dijalankan.")

    #current_fcaox = last_row["FCaOX"]
    # ambil FCaOX aktual terakhir yang tersedia
    current_fcaox = step1["FCaOX"].dropna().iloc[-1]

    X0 = last_row[scale_cols].values.astype(float)

    # isi NaN feature dengan 0
    X0 = np.nan_to_num(X0)

    def objective(x):

        X = pd.DataFrame([x], columns=scale_cols)

        X = X.fillna(0)

        Xs = scaler.transform(X)

        inc_pred = model_inc.predict(Xs)[0]

        if np.isnan(inc_pred):
            return 999

        abs_input = pd.DataFrame([{
            "FCaOX_Inc_pred": inc_pred,
            "FCaOX_lag1": current_fcaox
        }])

        if abs_input.isnull().values.any():
            return 999

        pred = model_abs.predict(abs_input)[0]

        return abs(pred - target)

    result = minimize(
        objective,
        X0,
        method="Nelder-Mead"
    )

    recommended = result.x

    rec_df = pd.DataFrame({
        "Variable": scale_cols,
        "Current": X0,
        "Recommended": recommended,
        "Delta": recommended - X0
    })

    return rec_df

#preskriptif proses
def prescribe_from_process(df_raw, pre, scaler, scale_cols, model_inc, model_abs, target):

    # ambil FCaOX aktual terakhir
    current_fcaox = df_raw["FCaOX"].dropna().iloc[-1]

    # state proses terakhir
    last_row = df_raw.iloc[-1].copy()

    process_cols = [
        "Torsi Motor Kiln",
        "Arus Motor Kiln",
        "Nox IKGA",
        "Suhu Calciner",
        "LSF"
    ]

    X0 = last_row[process_cols].values

    def objective(x):

        temp_df = df_raw.copy()

        for i, col in enumerate(process_cols):
            temp_df.loc[temp_df.index[-1], col] = x[i]

        step1 = run_rolling_window(temp_df, pre)

        last_feat = step1.iloc[-1]

        X = pd.DataFrame([last_feat[scale_cols]])

        Xs = scaler.transform(X)

        inc_pred = model_inc.predict(Xs)[0]

        abs_input = pd.DataFrame([{
            "FCaOX_Inc_pred": inc_pred,
            "FCaOX_lag1": current_fcaox
        }])

        pred = model_abs.predict(abs_input)[0]

        return abs(pred - target)

    result = minimize(
        objective,
        X0,
        method="Nelder-Mead"
    )

    rec = result.x

    rec_df = pd.DataFrame({
        "Variable": process_cols,
        "Current": X0,
        "Recommended": rec,
        "Delta": rec - X0
    })

    return rec_df

#fungsi LLM agent

def explain_prescriptive(rec_df):

    client = openai.OpenAI(
        base_url="https://api.llm7.io/v1",
        api_key=["1RRDNXKFj2nmyiNm0hGAH9nCu8UoGQ+5G0/qZ+7MFUiFupCiig7nzcE/O2fkmEoGK1bTOorNGssH/2zR71yYDDfYZq5DK366JeoMR+quDacy9PHITK8BFXuJBFnpG+bQSkjYlQ=="]
    )

    table_text = rec_df.to_string(index=False)

    prompt = f"""
You are a cement kiln process expert.

Explain the following prescriptive recommendation for controlling FCaOX.

Table:
{table_text}

Explain:
1. What process changes are recommended
2. Why these changes reduce or increase FCaOX
3. A simple instruction list for the kiln operator
4. Following point need to be explained
    "Torsi Motor Kiln" is torque of the kiln motor, higher torque can indicate higher load on the kiln which can affect the reduction process and thus FCaOX levels.
    "Arus Motor Kiln" is current of the kiln motor, which can also indicate load and energy input to the kiln, influencing the chemical reactions and FCaOX.
    "Nox IKGA" is the nitrogen oxide level in the kiln gas, which can affect the combustion efficiency and FCaOX levels.
    "Suhu Calciner" is the temperature in the calciner, which affects the decomposition of calcium carbonate and FCaOX levels.
    "LSF" is the Lime Saturation Factor, which indicates the amount of lime available for reaction and affects FCaOX levels.

Keep explanation concise and technical.
"""

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
# ============================================================
# LOAD PREPROCESSOR
# ============================================================
preprocessor_path = "window_preprocessor.joblib"

try:

    window_preprocessor = joblib.load(preprocessor_path)

    st.success("Window preprocessor loaded")

except Exception as e:

    st.error(f"Preprocessor error : {e}")
    st.stop()


# ============================================================
# PROCESSING
# ============================================================
st.subheader("⚙️ Processing Step-1")

try:

    step1 = run_rolling_window(df, window_preprocessor)

    st.session_state["step1_df"] = step1

    st.success(f"Step-1 completed : {len(step1):,} rows")

except Exception as e:

    st.error(f"Processing error : {e}")
    st.stop()


# ============================================================
# STEP1 OUTPUT
# ============================================================
step1 = st.session_state.get("step1_df")

if step1 is None:

    st.error("Step-1 failed")
    st.stop()

st.subheader("Step-1 Output")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", len(step1))
col2.metric("Columns", len(step1.columns))
col3.metric("FCaOX Inc valid", step1["FCaOX_Inc"].notna().sum())
col4.metric("FCaOX valid", step1["FCaOX"].notna().sum())

st.dataframe(step1.head(50), use_container_width=True)

st.download_button(
    "Download Step1 CSV",
    step1.to_csv(index=False),
    "kiln_step1.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================
st.subheader("🔮 Prediction")

try:

    bundle = joblib.load("scaler_process_lsf.joblib")

    scaler = bundle["scaler"]
    scale_cols = bundle["scale_cols"]

    model_inc = joblib.load("xgb_fcaox_increment.joblib")

    model_abs = joblib.load("fcao_abs_model.joblib")

    st.success("Models loaded")
    st.markdown("""model scaler : `scaler_process_lsf.joblib`  
model increment : `xgb_fcaox_increment.joblib`  
model absolute : `fcao_abs_model.joblib`""")
    

except Exception as e:

    st.warning(f"Model load failed : {e}")
    st.stop()


# ============================================================
# BATCH PREDICTION
# ============================================================
def run_batch_prediction(step1_df):

    df = step1_df.copy()

    df["LSF_lag1"] = df["LSF"].shift(1)
    df["FCaOX_lag1"] = df["FCaOX"].shift(1)

    results = []

    for _, row in df.iterrows():

        if pd.isna(row["LSF_lag1"]) or pd.isna(row["FCaOX_lag1"]):
            continue

        X = pd.DataFrame([row[scale_cols]])

        if X.isnull().any().any():
            continue

        Xs = scaler.transform(X)

        inc_pred = model_inc.predict(Xs)[0]

        abs_input = pd.DataFrame([{
            "FCaOX_Inc_pred": inc_pred,
            "FCaOX_lag1": row["FCaOX_lag1"]
        }])

        abs_pred = model_abs.predict(abs_input)[0]

        results.append({
            "time": row.iloc[0],
            "FCaOX_pred": abs_pred,
            "FCaOX_actual": row["FCaOX_lag1"],
            "FCaOX Accuracy": abs_pred - row["FCaOX_lag1"]
        })

    return pd.DataFrame(results)


if st.button("Re-Run Batch Prediction"):

    result = run_batch_prediction(step1)

    st.dataframe(result)

    st.line_chart(result[["FCaOX_pred", "FCaOX_actual"]])

    st.download_button(
        "Download Prediction",
        result.to_csv(index=False),
        "prediction.csv"
    )
else:
    result = run_batch_prediction(step1)

    st.dataframe(result)

    st.line_chart(result[["FCaOX_pred", "FCaOX_actual"]])

    st.download_button(
        "Download Prediction",
        result.to_csv(index=False),
        "prediction.csv"
    )

st.subheader("🧠 Prescriptive Control Recommendation")

target = st.number_input(
    "Target FCaOX",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.05
)

if st.button("Generate Recommendation"):

    last_row = step1.iloc[-1]
    # ambil timestamp
    time_col = step1.columns[0]
    data_time = last_row[time_col]
    data_time_str = pd.to_datetime(data_time).strftime("%Y-%m-%d %H:%M")

    rec = prescribe_control(
        last_row,
        scaler,
        scale_cols,
        model_inc,
        model_abs,
        target
    )

    st.success("Recommendation generated")

    st.metric("Data Timestamp", data_time_str)

    st.dataframe(rec, use_container_width=True)

    alarm = []

    for _, r in rec.iterrows():

        if abs(r["Delta"]) > 0.05:

            alarm.append(
                f"{r['Variable']} adjust {round(r['Delta'],3)}"
            )

    if alarm:

        st.warning("Recommended adjustments:")

        for a in alarm:
            st.write("-", a)

        explanation = explain_prescriptive(rec)

        st.subheader("AI Process Explanation")

        st.write(explanation)



if st.button("🔄 Refresh Page"):
    st.cache_data.clear()
    st.rerun()