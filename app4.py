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
# FCAOX STATUS HELPER
# ============================================================
def get_fcaox_status(value):
    """Return (label, color_hex, emoji) based on FCaOX value."""
    if value is None or np.isnan(value):
        return "N/A", "#888888", "❓"
    if value < 0.6:
        return "OVERBURN", "#FF6B35", "🔶"
    elif value <= 1.5:
        return "NORMAL", "#2ECC71", "✅"
    else:
        return "HIGH FCaOX", "#E74C3C", "🔴"


def render_fcaox_card(label, value, status_label, color, emoji):
    """Render a styled FCaOX status card using st.markdown."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 2px solid {color};
            border-radius: 16px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 4px 20px {color}33;
        ">
            <div style="font-size: 13px; color: #aaa; font-weight: 600;
                        letter-spacing: 1.5px; text-transform: uppercase;
                        margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-size: 52px; font-weight: 800;
                        color: {color}; line-height: 1; margin-bottom: 8px;">
                {round(value, 3) if value is not None and not np.isnan(value) else "—"}
            </div>
            <div style="font-size: 18px; font-weight: 700;
                        color: {color}; letter-spacing: 1px;">
                {emoji} {status_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    process_vars = scale_cols.copy()

    # state awal
    X0 = last_row[process_vars].values.astype(float)

    bounds = []

    for var, val in zip(process_vars, X0):

        if var == "LSF":
            bounds.append((val, val))  # dikunci

        elif "Torsi Motor Kiln" in var:
            low = val * 0.95
            high = val * 1.05
            bounds.append((min(low, high), max(low, high)))

        elif "Arus Motor Kiln" in var:
            low = val * 0.95
            high = val * 1.05
            bounds.append((min(low, high), max(low, high)))

        elif "Nox IKGA" in var:
            low = val * 0.90
            high = val * 1.10
            bounds.append((min(low, high), max(low, high)))

        elif "Suhu Calciner" in var:
            low = val - 20
            high = val + 20
            bounds.append((min(low, high), max(low, high)))

        else:
            low = val * 0.95
            high = val * 1.05
            bounds.append((min(low, high), max(low, high)))

    current_fcaox = last_row["FCaOX"]

    def objective(x):

        X = pd.DataFrame([x], columns=process_vars)

        Xs = scaler.transform(X)

        inc_pred = model_inc.predict(Xs)[0]

        abs_input = pd.DataFrame([{
            "FCaOX_Inc_pred": inc_pred,
            "FCaOX_lag1": current_fcaox
        }])

        pred = model_abs.predict(abs_input)[0]

        return (pred - target)**2

    result = minimize(
        objective,
        X0,
        method="Powell",
        bounds=bounds
    )

    rec = result.x

    rec_df = pd.DataFrame({
        "Variable": process_vars,
        "Current": X0,
        "Recommended": rec,
        "Delta": rec - X0
    })
    last_actual_fcaox = step1["FCaOX"].dropna().iloc[-1]

    # menentukan mode kontrol
    if target > last_actual_fcaox:
        mode = "increase"
    else:
        mode = "decrease"

    last_actual_fcaox = step1["FCaOX"].dropna().iloc[-1]

    if mode == "increase":

        direction_rules = {
        "Torsi Motor Kiln_diff_mean": -1,
        "Arus Motor Kiln_diff_mean": -1,
        "Nox IKGA_diff_mean": -1,
        "Suhu Calciner_diff_mean": -1,
        "LSF": 0
    }

    else:

        direction_rules = {
            "Torsi Motor Kiln_diff_mean": 1,
            "Arus Motor Kiln_diff_mean": 1,
            "Nox IKGA_diff_mean": 1,
            "Suhu Calciner_diff_mean": 1,
            "LSF": 0
        }
    for i, row in rec_df.iterrows():

        var = row["Variable"]
        delta = row["Delta"]

        rule = direction_rules.get(var)

        if rule == -1 and delta > 0:
            rec_df.loc[i, "Recommended"] = row["Current"]
            rec_df.loc[i, "Delta"] = 0

        if rule == 1 and delta < 0:
            rec_df.loc[i, "Recommended"] = row["Current"]
            rec_df.loc[i, "Delta"] = 0

        if rule == 0:
            rec_df.loc[i, "Recommended"] = row["Current"]
            rec_df.loc[i, "Delta"] = 0

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
Anda adalah seorang ahli proses tungku semen.

Jelaskan rekomendasi preskriptif berikut untuk mengendalikan FCaOX.

Table:
{table_text}

Penjelasan:
pertimbangkan Delta Point, Delta point adalah Arus vs Rekomendasi, jika positif berarti rekomendasi lebih tinggi dari kondisi saat ini, jika negatif berarti rekomendasi lebih rendah dari kondisi saat ini.
"Torsi Motor Kiln" adalah torsi motor kiln, torsi yang lebih tinggi dapat menunjukkan beban yang lebih tinggi pada kiln yang dapat memengaruhi proses reduksi dan dengan demikian tingkat FCaOX.
"Arus Motor Kiln" adalah arus motor kiln, yang juga dapat menunjukkan beban dan masukan energi ke kiln, memengaruhi reaksi kimia dan FCaOX.
"Nox IKGA" adalah tingkat nitrogen oksida dalam gas kiln, yang dapat memengaruhi efisiensi pembakaran dan tingkat FCaOX.
"Suhu Calciner" adalah suhu di dalam kalsinator, yang memengaruhi dekomposisi kalsium karbonat dan tingkat FCaOX.

"LSF" adalah Faktor Saturasi Kapur, yang menunjukkan jumlah kapur yang tersedia untuk reaksi dan memengaruhi tingkat FCaOX.

Jaga agar penjelasan tetap ringkas dan teknis.
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

with st.expander("Preview Step-1 Data"):
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

    # --------------------------------------------------------
    # FCaOX STATUS CARDS — setelah chart
    # --------------------------------------------------------
    st.subheader("📊 FCaOX Status Terkini")

    last_actual_val = result["FCaOX_actual"].dropna().iloc[-1] if not result["FCaOX_actual"].dropna().empty else None
    last_pred_val   = result["FCaOX_pred"].dropna().iloc[-1]   if not result["FCaOX_pred"].dropna().empty   else None

    act_label, act_color, act_emoji = get_fcaox_status(last_actual_val)
    pred_label, pred_color, pred_emoji = get_fcaox_status(last_pred_val)

    card_col1, card_col2, card_col3 = st.columns([2, 2, 1])

    with card_col1:
        render_fcaox_card("FCaOX Sekarang (Aktual)", last_actual_val, act_label, act_color, act_emoji)

    with card_col2:
        render_fcaox_card("FCaOX Prediksi", last_pred_val, pred_label, pred_color, pred_emoji)

    with card_col3:
        st.markdown(
            """
            <div style="background:#1a1a2e; border-radius:12px; padding:16px;
                        font-size:12px; color:#ccc; line-height:1.8;">
                <b style="color:#fff;">Keterangan</b><br>
                <span style="color:#2ECC71;">■</span> 0.6 – 1.5 : Normal<br>
                <span style="color:#FF6B35;">■</span> &lt; 0.6 : Overburn<br>
                <span style="color:#E74C3C;">■</span> &gt; 1.5 : FCaOX Tinggi
            </div>
            """,
            unsafe_allow_html=True,
        )
    # --------------------------------------------------------

    st.download_button(
        "Download Prediction",
        result.to_csv(index=False),
        "prediction.csv"
    )

else:
    result1 = run_batch_prediction(step1)
    result = result1.sort_values("time", ascending=False).reset_index(drop=True)
    st.dataframe(result, use_container_width=True, height=200)

    st.line_chart(result[["FCaOX_pred", "FCaOX_actual"]])

    # --------------------------------------------------------
    # FCaOX STATUS CARDS — setelah chart
    # --------------------------------------------------------
    st.subheader("📊 FCaOX Status Terkini")

    last_actual_val = result["FCaOX_actual"].dropna().iloc[-1] if not result["FCaOX_actual"].dropna().empty else None
    last_pred_val   = result["FCaOX_pred"].dropna().iloc[-1]   if not result["FCaOX_pred"].dropna().empty   else None

    act_label, act_color, act_emoji = get_fcaox_status(last_actual_val)
    pred_label, pred_color, pred_emoji = get_fcaox_status(last_pred_val)

    card_col1, card_col2, card_col3 = st.columns([2, 2, 1])

    with card_col1:
        render_fcaox_card("FCaOX Sekarang (Aktual)", last_actual_val, act_label, act_color, act_emoji)

    with card_col2:
        render_fcaox_card("FCaOX Prediksi", last_pred_val, pred_label, pred_color, pred_emoji)

    with card_col3:
        st.markdown(
            """
            <div style="background:#1a1a2e; border-radius:12px; padding:16px;
                        font-size:12px; color:#ccc; line-height:1.8;">
                <b style="color:#fff;">Keterangan</b><br>
                <span style="color:#2ECC71;">■</span> 0.6 – 1.5 : Normal<br>
                <span style="color:#FF6B35;">■</span> &lt; 0.6 : Overburn<br>
                <span style="color:#E74C3C;">■</span> &gt; 1.5 : FCaOX Tinggi
            </div>
            """,
            unsafe_allow_html=True,
        )
    # --------------------------------------------------------

    st.download_button(
        "Download Prediction",
        result.to_csv(index=False),
        "prediction.csv"
    )

st.subheader("🧠 Prescriptive Control Recommendation")

last_actual_fcaox = step1["FCaOX"].dropna().iloc[-1]

delta_target = st.number_input(
    "Δ Target FCaOX (relative to last actual)",
    min_value=-1.0,
    max_value=1.0,
    value=0.10,
    step=0.05
)

target = last_actual_fcaox + delta_target

st.info(
    f"Last Actual FCaOX : {round(last_actual_fcaox,3)} → Target FCaOX : {round(target,3)}"
)

if st.button("Generate Recommendation"):

    last_row = step1.iloc[-1]

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

    st.metric("Data Timestamp", str(data_time_str))
    st.metric("Last Actual FCaOX", round(last_actual_fcaox,3))
    st.metric("Target FCaOX", round(target,3))

    st.dataframe(rec, use_container_width=True)
    explanation = explain_prescriptive(rec)
    st.subheader("AI Process Explanation")
    st.write(explanation)


if st.button("🔄 Refresh Page"):
    st.cache_data.clear()
    st.rerun()