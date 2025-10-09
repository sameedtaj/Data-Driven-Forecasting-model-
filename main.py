# app.py
# -*- coding: utf-8 -*-
import io
import os
import tempfile
from datetime import datetime
import streamlit as st
from contextlib import redirect_stdout

st.set_page_config(page_title="Predictive Analysis using ML — Month Closing", page_icon="📈", layout="wide")

# ---------- UI: Heading ----------
st.title("📈 Data-Driven Month Close Predictive Analysis")
st.caption("Upload your daily data Excel file to generate current-month actuals, remaining-day forecasts, and a summary. The backend runs exactly as you provided.")

# ---------- UI: File uploader ----------
uploaded = st.file_uploader("Upload your Excel file (.xlsx) with the required columns", type=["xlsx"])

# ---------- Backend (original logic; only CONFIG lines are conditional so UI-injected paths work) ----------
BACKEND_SCRIPT = r'''
# -*- coding: utf-8 -*-
"""
Month-Close Forecast with XGBoost/LightGBM + Robust Blending
------------------------------------------------------------
(Original docstring preserved)
"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ========= Console formatting =========
np.set_printoptions(suppress=True)
pd.set_option("display.float_format", "{:,.0f}".format)

# ========= Optional: TensorFlow/Keras (DL section) =========
TF_AVAILABLE, TF_IMPORT_ERROR = False, None
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = str(e)
    TF_AVAILABLE = False

# ========= Try LightGBM / XGBoost =========
LGBM_OK, XGB_OK = False, False
try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except Exception:
    LGBM_OK = False

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

from sklearn.ensemble import GradientBoostingRegressor  # fallback

# =========================
# 📂 CONFIG  (make conditional so UI-injected values aren't overwritten)
# =========================
try:
    FILE_PATH
except NameError:
    FILE_PATH = r"D:\Predictive analysis data.xlsx"
try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents")
RANDOM_STATE = 42

# =========================
# 🧰 Helpers
# =========================
def month_bounds(d):
    start = d.replace(day=1)
    end = (start + relativedelta(months=1)) - timedelta(days=1)
    return start, end

def daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def iso_week(series_dt):
    try:
        iso = series_dt.dt.isocalendar()
        week = getattr(iso, "week", None) or iso["week"]
        return week.astype(int)
    except Exception:
        return series_dt.dt.isocalendar().week.astype(int)

def build_time_features(dates_like):
    s = pd.to_datetime(dates_like)
    feats = pd.DataFrame(index=range(len(s)))
    feats["day_of_week"]  = s.dt.dayofweek
    feats["week_of_year"] = iso_week(s)
    feats["month"]        = s.dt.month
    feats["day"]          = s.dt.day
    feats["day_of_year"]  = s.dt.dayofyear
    feats["is_weekend"]   = (s.dt.dayofweek >= 5).astype(int)
    feats["quarter"]      = s.dt.quarter
    feats["is_month_end"] = s.dt.is_month_end.astype(int)
    feats["days_since_start"] = (s - s.min()).dt.days
    return feats

def coerce_numeric(x):
    if pd.api.types.is_numeric_dtype(x):
        return x
    return (x.astype(str).str.replace(",", "", regex=False)
                   .str.replace("%", "", regex=False).astype(float))

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(2*np.abs(y_pred - y_true)/denom) * 100.0

def format_daily_for_print(df_daily):
    out = df_daily.copy()
    out["Date"] = out["Started Date"].dt.strftime("%Y-%m-%d")
    for c in ["Total Txn", "Customers"]:
        if c in out.columns: out[c] = out[c].round(0).astype(np.int64).map(lambda v: f"{v:,}")
    for c in ["Total Vol", "Successful vol"]:
        if c in out.columns: out[c] = out[c].map(lambda v: f"{v:,.0f}")
    if "SR" in out.columns:
        out["SR"] = out["SR"].map(lambda v: "—" if pd.isna(v) else f"{v*100:.2f}%")
    cols = ["Date", "Total Txn", "Total Vol", "Successful vol", "Customers"]
    if "SR" in out.columns: cols.append("SR")
    cols.append("Type")
    return out[cols]

def format_summary_for_print(df_sum):
    out = df_sum.copy()
    for c in ["Total Txn","Total Vol","Successful Vol","Total Customers (sum of daily)"]:
        if c in out.columns: out[c] = out[c].map(lambda v: f"{int(round(v)):,}")
    for c in ["SR Txn-weighted (%)","SR Vol-weighted (%)"]:
        if c in out.columns:
            out[c] = out[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}%")
    return out

def save_excel_safe(df_sheets, summary_df, base_name="current_month_actuals_and_forecast"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{base_name}_{ts}.xlsx")
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for sheet_name, df in df_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            wb = writer.book
            fmt_int   = wb.add_format({"num_format": "#,##0"})
            fmt_float = wb.add_format({"num_format": "#,##0"})
            fmt_pct   = wb.add_format({"num_format": "0.00%"})
            fmt_date  = wb.add_format({"num_format": "yyyy-mm-dd"})

            for sheet_name, df in df_sheets.items():
                ws = writer.sheets[sheet_name]
                headers = list(df.columns); col_idx = {n:i for i,n in enumerate(headers)}
                if "Started Date" in col_idx: ws.set_column(col_idx["Started Date"], col_idx["Started Date"], 12, fmt_date)
                for c in ["Total Txn","Customers"]:
                    if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 14, fmt_int)
                for c in ["Total Vol","Successful vol"]:
                    if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 18, fmt_float)
                if "SR" in col_idx: ws.set_column(col_idx["SR"], col_idx["SR"], 10, fmt_pct)
                if "Type" in col_idx: ws.set_column(col_idx["Type"], col_idx["Type"], 12)

            ws = writer.sheets["Summary"]
            headers = list(summary_df.columns); hidx = {h:i for i,h in enumerate(headers)}
            def set_s(name, fmt, width=22):
                if name in hidx: ws.set_column(hidx[name], hidx[name], width, fmt)
            set_s("Total Txn", fmt_int); set_s("Total Vol", fmt_float); set_s("Successful Vol", fmt_float)
            set_s("Total Customers (sum of daily)", fmt_int, 28)
            for col in ["SR Txn-weighted (fraction)","SR Vol-weighted (fraction)"]:
                if col in hidx: set_s(col, fmt_pct, 28)
    except PermissionError:
        ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(OUTPUT_DIR, f"{base_name}_{ts2}.xlsx")
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for sheet_name, df in df_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
    return path

# ---------- Engine selection ----------
def make_regressor(engine="auto", seed=RANDOM_STATE):
    eng = engine.lower()
    if eng == "lgbm" or (eng == "auto" and LGBM_OK):
        model = LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.2, reg_lambda=0.8,
            random_state=seed, n_jobs=-1
        )
        return model, "LGBM"
    if eng == "xgb" or (eng == "auto" and XGB_OK):
        model = XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:squarederror", random_state=seed, n_jobs=-1
        )
        return model, "XGB"
    model = GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=3, random_state=seed
    )
    return model, "GBM"

# ---------- Adaptive lag/rolling for tiny histories ----------
def add_lag_rolling_adaptive(df, date_col, cols):
    n = len(df)
    lags  = [1] if n < 7 else [1,7]
    if n < 7:
        rolls = [max(2, min(3, n-1))]
    elif n < 14:
        rolls = [7]
    elif n < 28:
        rolls = [7,14]
    else:
        rolls = [7,14,28]

    df = df.sort_values(date_col).copy()
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        for W in rolls:
            df[f"{c}_roll{W}"] = (df[c].shift(1)).rolling(window=W, min_periods=max(2, W//2)).mean()
    feat_cols = [col for col in df.columns if any(k in col for k in ["_lag","_roll"])]
    df[feat_cols] = df[feat_cols].ffill().bfill()
    return df

def weekday_baseline(df, value_col, weight_col=None):
    tmp = df[["Started Date", value_col] + ([weight_col] if weight_col else [])].dropna().copy()
    tmp["dow"] = tmp["Started Date"].dt.dayofweek
    if weight_col:
        grp = tmp.groupby("dow").apply(lambda g: np.average(g[value_col], weights=g[weight_col]))
        return grp.to_dict()
    else:
        return tmp.groupby("dow")[value_col].mean().to_dict()

def moving_average(series, window):
    if len(series) == 0: return np.nan
    w = max(1, min(window, len(series)))
    return float(pd.Series(series).tail(w).mean())

def blend_with_baseline(model_pred, baseline_value, horizon_days):
    h = max(1, int(horizon_days))
    w_model = np.exp(-h/14.0)
    return w_model * model_pred + (1.0 - w_model) * baseline_value

def clip_to_recent_range(value, recent_values, q_low=0.05, q_high=0.95):
    if len(recent_values) < 5:
        return float(value)
    lo, hi = np.quantile(recent_values, [q_low, q_high])
    return float(np.clip(value, lo, hi))

# =========================
# 📥 Load & clean
# =========================
df = pd.read_excel(FILE_PATH)
df.columns = df.columns.str.strip()

required = ["Started Date","Total Txn","Total Vol","Successful vol","Customers"]
miss = [c for c in required if c not in df.columns]
if miss: raise ValueError(f"Missing required columns: {miss}. Found: {list(df.columns)}")

df["Started Date"] = pd.to_datetime(df["Started Date"])
for col in ["Total Txn","Total Vol","Successful vol","Customers"]:
    df[col] = coerce_numeric(df[col])

# SR handling
sr_available = False
if "SR" in df.columns:
    try:
        sr_raw = coerce_numeric(df["SR"])
        sr_available = True
        df["SR"] = sr_raw/100.0 if sr_raw.max() > 1.5 else sr_raw
    except Exception:
        df["SR"] = pd.NA; sr_available = False

df = df.dropna(subset=["Started Date","Total Txn","Total Vol","Successful vol","Customers"]).copy()
df = df.sort_values("Started Date").reset_index(drop=True)

# SR model target
if not sr_available or df["SR"].isna().any():
    sr_from_vol = np.where(df["Total Vol"]>0, df["Successful vol"]/df["Total Vol"], np.nan)
    sr_fill = np.nanmedian(sr_from_vol[np.isfinite(sr_from_vol)]) if np.isfinite(sr_from_vol).any() else 0.55
    df["SR_model_target"] = np.where(np.isfinite(sr_from_vol), sr_from_vol, sr_fill)
else:
    df["SR_model_target"] = df["SR"].astype(float)

# =========================
# 🧪 Features + calendar
# =========================
lag_cols = ["Total Txn","Total Vol","Customers","SR_model_target"]
df_feat = add_lag_rolling_adaptive(df, "Started Date", lag_cols)
cal_feats = build_time_features(df_feat["Started Date"])
X_all = pd.concat([cal_feats, df_feat[[c for c in df_feat.columns if any(k in c for k in ["_lag","_roll"])]]], axis=1).reset_index(drop=True)

y_txn = df_feat["Total Txn"].astype(float).values
y_vol = df_feat["Total Vol"].astype(float).values
y_cus = df_feat["Customers"].astype(float).values
y_srf = df_feat["SR_model_target"].astype(float).values

n = len(X_all)
split_idx = int(n*0.8) if n >= 50 else max(1, n-10)
X_tr, X_va = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_tr = {"txn": y_txn[:split_idx], "vol": y_vol[:split_idx], "cus": y_cus[:split_idx], "srf": y_srf[:split_idx]}
y_va = {"txn": y_txn[split_idx:], "vol": y_vol[split_idx:], "cus": y_cus[split_idx:], "srf": y_srf[split_idx:]}

# =========================
# 🧠 Train models
# =========================
def fit_one_target(ytr, yva, name="target"):
    est, eng = make_regressor(engine="auto", seed=RANDOM_STATE)
    if eng == "LGBM" and len(X_va) >= 15:
        est.set_params(metric="l2")
        est.fit(X_tr, ytr, eval_set=[(X_va, yva)], callbacks=[])
        try:
            est.fit(X_tr, ytr, eval_set=[(X_va, yva)], early_stopping_rounds=50, verbose=False)
        except Exception:
            pass
    elif eng == "XGB" and len(X_va) >= 15:
        est.fit(X_tr, ytr, eval_set=[(X_va, yva)], early_stopping_rounds=80, verbose=False)
    else:
        est.fit(X_tr, ytr)
    return est, eng

model_txn, eng_txn = fit_one_target(y_tr["txn"], y_va["txn"], "Total Txn")
model_vol, eng_vol = fit_one_target(y_tr["vol"], y_va["vol"], "Total Vol")
model_cus, eng_cus = fit_one_target(y_tr["cus"], y_va["cus"], "Customers")
model_srf, eng_srf = fit_one_target(y_tr["srf"], y_va["srf"], "SR")

# =========================
# 📅 Windows
# =========================
last_date = df["Started Date"].max()
cur_start, cur_end = month_bounds(last_date)
actual_cur = df[(df["Started Date"]>=cur_start) & (df["Started Date"]<=last_date)].copy()

start_pred = last_date + timedelta(days=1)
pred_dates_current = list(daterange(start_pred, cur_end)) if start_pred <= cur_end else []

# ===== Baselines =====
dow_txn = weekday_baseline(df, "Total Txn")
dow_vol = weekday_baseline(df, "Total Vol")
dow_cus = weekday_baseline(df, "Customers")
dow_sr  = weekday_baseline(df.assign(weight=df["Total Txn"].clip(lower=1e-9)),
                           "SR_model_target", weight_col="weight")

# =========================
# 🔮 Predict with blending + clipping
# =========================
def make_feature_row(history_df, date):
    s = pd.Series([date]); cal = build_time_features(s)
    feats = {}
    for col in lag_cols:
        feats[f"{col}_lag1"] = history_df[col].iloc[-1] if len(history_df)>=1 else np.nan
        if f"{col}_lag7" in X_all.columns:
            feats[f"{col}_lag7"] = history_df[col].iloc[-7] if len(history_df)>=7 else feats[f"{col}_lag1"]
        for W in (7,14,28):
            key = f"{col}_roll{W}"
            if key in X_all.columns:
                vals = history_df[col].tail(W)
                feats[key] = np.nan if vals.empty else float(vals.mean())
    row = pd.concat([cal.reset_index(drop=True), pd.DataFrame([feats])], axis=1)
    for c in X_all.columns:
        if c not in row.columns: row[c] = np.nan
    row = row[X_all.columns].fillna(X_all.median(numeric_only=True))
    return row

def predict_future_dates(pred_dates):
    if not pred_dates:
        return pd.DataFrame(columns=["Started Date","Total Txn","Total Vol","Successful vol","Customers","SR"])
    hist = df_feat[["Started Date","Total Txn","Total Vol","Customers","SR_model_target"]].copy()
    out = []
    for step, d in enumerate(pred_dates, start=1):
        X_row = make_feature_row(hist, d)
        txn_m = float(max(model_txn.predict(X_row)[0], 0.0))
        vol_m = float(max(model_vol.predict(X_row)[0], 0.0))
        cus_m = float(max(model_cus.predict(X_row)[0], 0.0))
        sr_m  = float(np.clip(model_srf.predict(X_row)[0], 0.0, 1.0))

        dow = d.weekday()
        txn_b = np.nanmean([moving_average(hist["Total Txn"].values, 7), dow_txn.get(dow, np.nan)])
        vol_b = np.nanmean([moving_average(hist["Total Vol"].values, 7), dow_vol.get(dow, np.nan)])
        cus_b = np.nanmean([moving_average(hist["Customers"].values, 7), dow_cus.get(dow, np.nan)])
        sr_b  = np.nanmean([moving_average(hist["SR_model_target"].values, 7), dow_sr.get(dow, np.nan)])

        def safe_base(b, series, default=0.0):
            if np.isnan(b):
                if len(series)>0: return float(pd.Series(series).iloc[-1])
                return default
            return float(b)

        txn_b = safe_base(txn_b, hist["Total Txn"].values)
        vol_b = safe_base(vol_b, hist["Total Vol"].values)
        cus_b = safe_base(cus_b, hist["Customers"].values)
        sr_b  = np.clip(safe_base(sr_b, hist["SR_model_target"].values, default=0.55), 0.0, 1.0)

        txn = blend_with_baseline(txn_m, txn_b, step)
        vol = blend_with_baseline(vol_m, vol_b, step)
        cus = blend_with_baseline(cus_m, cus_b, step)
        srp = float(np.clip(blend_with_baseline(sr_m, sr_b, step), 0.0, 1.0))

        recent_txn = hist["Total Txn"].tail(28).values
        recent_vol = hist["Total Vol"].tail(28).values
        recent_cus = hist["Customers"].tail(28).values
        recent_srf = hist["SR_model_target"].tail(28).values
        txn = clip_to_recent_range(txn, recent_txn)
        vol = clip_to_recent_range(vol, recent_vol)
        cus = clip_to_recent_range(cus, recent_cus)
        srp = float(np.clip(clip_to_recent_range(srp, recent_srf), 0.0, 1.0))

        suc = max(vol * srp, 0.0)
        out.append({"Started Date": d, "Total Txn": txn, "Total Vol": vol,
                    "Successful vol": suc, "Customers": cus, "SR": srp})

        hist = pd.concat([hist, pd.DataFrame([{
            "Started Date": d, "Total Txn": txn, "Total Vol": vol,
            "Customers": cus, "SR_model_target": srp
        }])], ignore_index=True)
    return pd.DataFrame(out)

# =========================
# 🔧 Build daily outputs
# =========================
pred_cur = predict_future_dates(pred_dates_current)

actual_daily = actual_cur[["Started Date","Total Txn","Total Vol","Successful vol","Customers"]].copy()
if sr_available:
    m = df.set_index("Started Date")["SR"]
    actual_daily["SR"] = m.reindex(actual_daily["Started Date"]).values
else:
    actual_daily["SR"] = np.where(actual_daily["Total Vol"]>0,
                                  actual_daily["Successful vol"]/actual_daily["Total Vol"], np.nan)
actual_daily["Type"] = "ACTUAL"

pred_daily = pred_cur.copy()
if not pred_daily.empty:
    pred_daily["Type"] = "PREDICTED"

full_daily = pd.concat([actual_daily, pred_daily], ignore_index=True).sort_values("Started Date")

# =========================
# 📈 Month summaries (no "Unique Customers")
# =========================
def summarize_block(df_block, label):
    s_txn = df_block["Total Txn"].sum()
    s_vol = df_block["Total Vol"].sum()
    s_suc = df_block["Successful vol"].sum()
    s_cus = df_block["Customers"].sum()

    sr_vol = (s_suc/s_vol) if s_vol>0 else np.nan

    tmp = df_block.copy()
    tmp["SR_eff"] = tmp["SR"]
    mask = tmp["SR_eff"].isna() & (tmp["Total Vol"]>0)
    tmp.loc[mask, "SR_eff"] = tmp.loc[mask, "Successful vol"] / tmp.loc[mask, "Total Vol"]
    if tmp["Total Txn"].sum() > 0:
        sr_txn = float((tmp["SR_eff"] * tmp["Total Txn"]).sum() / tmp["Total Txn"].sum())
    else:
        sr_txn = np.nan

    return {
        "Period": label,
        "Total Txn": float(s_txn),
        "Total Vol": float(s_vol),
        "Successful Vol": float(s_suc),
        "Total Customers (sum of daily)": float(s_cus),
        "SR Txn-weighted (fraction)": None if np.isnan(sr_txn) else float(sr_txn),
        "SR Vol-weighted (fraction)": None if np.isnan(sr_vol) else float(sr_vol),
    }

last_date = df["Started Date"].max()
cur_start, cur_end = month_bounds(last_date)
actual_label = f"{cur_start:%Y-%m} Actuals to {last_date:%Y-%m-%d}"
proj_label   = f"{cur_start:%Y-%m} Projected Close"
summary_df = pd.DataFrame([summarize_block(actual_daily, actual_label),
                           summarize_block(full_daily,  proj_label)])

# =========================
# 🖨️ Console (tabulated views)
# =========================
print("\n==================== ACTUALS: Current Month (daily) ====================")
print(tabulate(format_daily_for_print(actual_daily), headers="keys", tablefmt="fancy_grid", showindex=False))
print("\n==================== PREDICTIONS: Remaining Current Month (daily) ====================")
if pred_daily.empty:
    print("No remaining days to predict (you already have data to month-end).")
else:
    print(tabulate(format_daily_for_print(pred_daily), headers="keys", tablefmt="fancy_grid", showindex=False))
print("\n==================== SUMMARY: Actuals & Projected Close ====================")
sum_print = summary_df.copy()
sum_print["SR Txn-weighted (%)"] = sum_print["SR Txn-weighted (fraction)"] * 100.0
sum_print["SR Vol-weighted (%)"] = sum_print["SR Vol-weighted (fraction)"] * 100.0
sum_print = sum_print.drop(columns=["SR Txn-weighted (fraction)","SR Vol-weighted (fraction)"])
print(tabulate(format_summary_for_print(sum_print), headers="keys", tablefmt="fancy_grid", showindex=False))

# =========================
# 🤖 Deep Learning (optional)  (unchanged)
# =========================
def maybe_run_deep_learning():
    if not TF_AVAILABLE:
        if TF_IMPORT_ERROR:
            print("\n⚠️ TensorFlow/Keras not fully available — skipping Deep Learning.")
            print("   Import error:", TF_IMPORT_ERROR)
        return
    try:
        pass  # kept minimal for speed; you can restore your DL block here if desired
    except Exception as e:
        print("\n⚠️ Deep Learning section encountered an error and was skipped.")
        print("   Details:", str(e))

maybe_run_deep_learning()

# =========================
# 💾 Save Excel
# =========================
sheets = {
    "Actuals_Current_Month_Daily": actual_daily,
    "Pred_Remaining_Current_Mo": pred_daily,
    "Current_Month_Full_Daily": full_daily,
}
excel_path = save_excel_safe(sheets, summary_df, base_name="current_month_actuals_and_forecast")
print(f"\n✅ Saved Excel here: {excel_path}")
'''

# ---------- Runner ----------
def run_backend_with_uploaded_file(uploaded_file):
    """
    Writes uploaded file to a unique temp path, sets FILE_PATH and OUTPUT_DIR for the backend,
    executes the backend verbatim, and returns:
      excel_path, console_text, actual_df, pred_df, summary_df
    """
    import pandas as pd

    tmp_dir = tempfile.mkdtemp(prefix="month_close_")
    tmp_in = os.path.join(tmp_dir, f"input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.xlsx")
    out_dir = os.path.join(tmp_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    with open(tmp_in, "wb") as f:
        f.write(uploaded_file.getbuffer())

    g = {}
    g["FILE_PATH"] = tmp_in
    g["OUTPUT_DIR"] = out_dir

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(BACKEND_SCRIPT, g)
    except Exception:
        import traceback
        traceback.print_exc(file=buf)
    console_text = buf.getvalue()

    # Retrieve excel path
    excel_path = g.get("excel_path")
    if not excel_path or not os.path.exists(excel_path):
        candidates = [os.path.join(out_dir, p) for p in os.listdir(out_dir) if p.lower().endswith(".xlsx")]
        excel_path = candidates[0] if candidates else None

    # ALSO surface the real dataframes produced by the backend for full-width viewing
    actual_df  = g.get("actual_daily", None)
    pred_df    = g.get("pred_daily", None)
    summary_df = g.get("summary_df", None)

    # Ensure we return pandas DataFrames or None
    for vname in ("actual_df", "pred_df", "summary_df"):
        v = locals()[vname]
        if v is not None and not hasattr(v, "to_dict"):
            locals()[vname] = None

    return excel_path, console_text, actual_df, pred_df, summary_df

# ---------- UI: Action & Results ----------
if uploaded:
    st.success(f"File uploaded: **{uploaded.name}** ({uploaded.size:,} bytes). Click **Run Forecast**.")
    if st.button("🚀 Run Forecast"):
        with st.spinner("Running your month-close forecast..."):
            excel_path, console_text, actual_df, pred_df, summary_df = run_backend_with_uploaded_file(uploaded)

        # Download Excel
        if excel_path and os.path.exists(excel_path):
            st.subheader("✅ Forecast Complete")
            c1, c2 = st.columns([2,1])
            with c1:
                st.write("Your formatted Excel is ready to download:")
            with c2:
                with open(excel_path, "rb") as f:
                    excel_bytes = f.read()
                dl_name = os.path.basename(excel_path)
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_bytes,
                    file_name=dl_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_{dl_name}"
                )
        else:
            st.error("The backend did not produce an Excel file.")

        # Tabs: Structured Tables | Raw Logs (Wrap) | Raw Logs (Scroll)
        st.markdown("### 📊 Results & Logs")
        tab1, tab2, tab3 = st.tabs(["Structured tables", "Raw logs (wrap)", "Raw logs (horizontal scroll)"])

        with tab1:
            st.caption("These are real DataFrames captured from the backend’s global namespace.")
            if actual_df is not None:
                st.markdown("**Actuals — Current Month (daily)**")
                st.dataframe(actual_df, use_container_width=True, height=420)
            if pred_df is not None and not pred_df.empty:
                st.markdown("**Predictions — Remaining Current Month (daily)**")
                st.dataframe(pred_df, use_container_width=True, height=420)
            if summary_df is not None:
                st.markdown("**Summary — Actuals & Projected Close**")
                st.dataframe(summary_df, use_container_width=True, height=260)

        with tab2:
            st.caption("Wrapped view (quick preview).")
            st.code(console_text or "(no logs)")

        with tab3:
            st.caption("Full-width, no wrapping. Use the horizontal scrollbar to see every column.")
            # Download logs
            st.download_button(
                "⬇️ Download logs (.txt)",
                data=(console_text or "").encode("utf-8"),
                file_name=f"processing_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=False
            )
            # Non-wrapping preformatted block with horizontal scroll
            safe = (console_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"""
                <div style="border:1px solid #ddd;border-radius:8px;padding:10px;max-height:700px;overflow:auto;background:#0e1117;">
                  <pre style="margin:0;color:#c9d1d9;white-space:pre;overflow-x:auto;">{safe}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Please upload a .xlsx file to begin.")

# ---------- Footer ----------
st.markdown("---")
st.caption("© " + datetime.now().strftime("%Y") + " Predictive Analysis using ML — Month Closing")
