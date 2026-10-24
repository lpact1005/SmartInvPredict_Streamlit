"""
Smart Inventory Forecast - Streamlit Version
Uses the original XGBoost model to provide a publicly deployable web interface.
Deploy to Streamlit Community Cloud to obtain a public link.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_model.joblib"
MODEL_DOMAIN_MEAN = 20000.0


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_cols = []
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feature_cols = [str(x) for x in model.feature_names_in_]
    elif hasattr(model, "get_booster"):
        feature_cols = [str(x) for x in (model.get_booster().feature_names or [])]
    if not feature_cols:
        raise RuntimeError("Model has no feature names")
    return model, feature_cols


# ---------------------------------------------------------------------------
# Helper functions (from original ml_server.py)
# ---------------------------------------------------------------------------
def _hash_id(s: str, lo: int = 1, hi: int = 45) -> int:
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return lo + (h % (hi - lo + 1))


def season_enc(season: str) -> int:
    return {"none": 0, "peak": 2, "low": 1, "holiday": 3}.get(season or "none", 0)


def lag_value(series: list[float], lag: int) -> float:
    if not series:
        return 0.0
    if len(series) >= lag:
        return float(series[-lag])
    return float(series[0])


def roll_mean(series: list[float], w: int) -> float:
    if not series:
        return 0.0
    chunk = series[-w:] if len(series) >= w else series
    return float(np.mean(chunk))


def roll_std(series: list[float], w: int) -> float:
    if not series:
        return 0.0
    chunk = series[-w:] if len(series) >= w else series
    if len(chunk) < 2:
        return 0.0
    return float(np.std(chunk, ddof=0))


def build_row(
    *,
    history_scaled: list[float],
    base_date: date,
    step: int,
    granularity: str,
    season: str,
    has_promotion: bool,
    store_id: int,
    department: int,
    store_size: float,
    feature_cols: list[str],
) -> dict[str, float]:
    if granularity == "month":
        d = base_date + timedelta(days=30 * (step + 1))
    else:
        d = base_date + timedelta(days=7 * (step + 1))

    iso = d.isocalendar()
    is_holiday = 1.0 if season == "holiday" else 0.0
    md = 500.0 if has_promotion else 0.0

    row: dict[str, float] = {
        "store_id": float(store_id),
        "department": float(department),
        "store_size": float(store_size),
        "year": float(d.year),
        "month": float(d.month),
        "week": float(iso.week),
        "dayofweek": float(d.weekday()),
        "quarter": float((d.month - 1) // 3 + 1),
        "day_of_year": float(d.timetuple().tm_yday),
        "is_weekend": 1.0 if d.weekday() >= 5 else 0.0,
        "is_month_start": 1.0 if d.day <= 3 else 0.0,
        "is_month_end": 1.0 if d.day >= 28 else 0.0,
        "temperature": 70.0,
        "fuel_price": 3.5,
        "cpi": 220.0,
        "unemployment": 6.0,
        "markdown_1": md * 0.4,
        "markdown_2": md * 0.2,
        "markdown_3": md * 0.15,
        "markdown_4": md * 0.15,
        "markdown_5": md * 0.1,
        "total_markdown": md,
        "is_holiday": is_holiday,
        "store_type_enc": 1.0,
        "region_enc": 1.0,
        "season_enc": float(season_enc(season)),
        "holiday_name_enc": 1.0 if season == "holiday" else 0.0,
        "sales_lag_1": lag_value(history_scaled, 1),
        "sales_lag_2": lag_value(history_scaled, 2),
        "sales_lag_4": lag_value(history_scaled, 4),
        "sales_lag_8": lag_value(history_scaled, 8),
        "sales_roll_mean_4": roll_mean(history_scaled, 4),
        "sales_roll_std_4": roll_std(history_scaled, 4),
        "sales_roll_mean_8": roll_mean(history_scaled, 8),
    }
    for c in feature_cols:
        row.setdefault(c, 0.0)
    return {c: float(row.get(c, 0.0)) for c in feature_cols}


def predict_one(model, row: dict[str, float], feature_cols: list[str]) -> float:
    X = pd.DataFrame([row], columns=feature_cols)
    pred = float(model.predict(X)[0])
    if not math.isfinite(pred):
        return 0.0
    return max(0.0, pred)


def multi_step_forecast(
    model,
    feature_cols: list[str],
    past_sales: list[float],
    periods: int,
    granularity: str,
    season: str,
    has_promotion: bool,
    sku: str,
    product_name: str,
) -> dict:
    if len(past_sales) < 3:
        raise ValueError("At least 3 historical sales values are required")

    periods = max(3, min(12, periods))
    store_id = _hash_id(sku, 1, 45)
    department = _hash_id(product_name, 1, 99)
    store_size = 150000.0

    user_recent = past_sales[-min(4, len(past_sales)):]
    user_mean = float(np.mean(user_recent))
    user_mean = max(user_mean, 1e-6)
    scale = MODEL_DOMAIN_MEAN / user_mean

    history_scaled = [v * scale for v in past_sales]
    base_date = date.today()

    preds_user: list[float] = []
    hist = list(history_scaled)
    user_hist = list(past_sales)

    for step in range(periods):
        row = build_row(
            history_scaled=hist,
            base_date=base_date,
            step=step,
            granularity=granularity,
            season=season,
            has_promotion=has_promotion,
            store_id=store_id,
            department=department,
            store_size=store_size,
            feature_cols=feature_cols,
        )
        y_scaled = predict_one(model, row, feature_cols)
        baseline = max(roll_mean(hist, 4), 1e-6)
        rel = y_scaled / baseline
        rel = max(0.75, min(1.35, rel))
        last_u = user_hist[-1] if user_hist else user_mean
        y_user = 0.55 * last_u + 0.45 * (user_mean * rel)
        band_hi = max(past_sales) * 1.6 + user_mean * 0.25
        band_lo = max(0.0, min(past_sales) * 0.55)
        y_user = max(band_lo, min(band_hi, y_user))
        preds_user.append(y_user)

        hist.append(y_user * scale)
        user_hist.append(y_user)

    preds_int = [int(round(p)) for p in preds_user]
    recent_avg = user_mean
    forecast_avg = float(np.mean(preds_int)) if preds_int else 0.0
    change = (forecast_avg - recent_avg) / max(recent_avg, 1e-6)
    change_pct = round(change * 1000) / 10

    return {
        "forecast": preds_int,
        "changePercent": change_pct,
        "recentAvg": round(recent_avg, 1),
        "forecastAvg": round(forecast_avg, 1),
        "storeId": store_id,
        "department": department,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Inventory Forecast",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Smart Inventory Forecast")
st.caption("Multi-step demand forecast demo powered by XGBoost | Deploy to Streamlit Cloud for a public link")

try:
    model, feature_cols = load_model()
    st.success(f"XGBoost model loaded (features: {len(feature_cols)})")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

with st.sidebar:
    st.header("Input Parameters")
    product_name = st.text_input("Product Name", value="Classic Black Coffee Beans 500g")
    sku = st.text_input("SKU", value="COF-500-BLK")
    current_stock = st.number_input("Current Stock", min_value=0, value=48, step=1)
    past_sales_raw = st.text_area(
        "Historical Sales (comma-separated)",
        value="22, 25, 24, 28, 30, 27, 32, 35",
        height=100,
        help="At least 3 numbers, separated by commas",
    )
    granularity = st.selectbox(
        "Time Granularity",
        ["week", "month"],
        format_func=lambda x: "Week" if x == "week" else "Month",
    )
    periods = st.slider("Forecast Periods", min_value=3, max_value=12, value=6)
    season = st.selectbox(
        "Season Factor",
        ["none", "peak", "low", "holiday"],
        format_func=lambda x: {
            "none": "None",
            "peak": "Peak Season",
            "low": "Low Season",
            "holiday": "Holiday",
        }[x],
    )
    has_promotion = st.checkbox("Has Promotion", value=False)
    lead_time = st.number_input("Lead Time (days)", min_value=0, value=7, step=1)
    unit_cost = st.number_input("Unit Cost", min_value=0.0, value=22.5, step=0.5)

    run = st.button("Generate Forecast", type="primary", use_container_width=True)

if run:
    try:
        past = [
            max(0.0, float(x.strip()))
            for x in past_sales_raw.replace("，", ",").split(",")
            if x.strip()
        ]
        result = multi_step_forecast(
            model,
            feature_cols,
            past,
            periods,
            granularity,
            season,
            has_promotion,
            sku,
            product_name,
        )

        st.subheader("Forecast Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recent Average Sales", f"{result['recentAvg']}")
        col2.metric("Forecast Average Sales", f"{result['forecastAvg']}")
        col3.metric("Change", f"{result['changePercent']:+.1f}%")

        # Chart
        labels = [f"Period {i+1}" for i in range(len(result["forecast"]))]
        chart_df = pd.DataFrame({
            "Period": labels,
            "Forecasted Sales": result["forecast"],
        })
        st.line_chart(chart_df.set_index("Period"))

        st.write("**Forecast values:**", ", ".join(str(v) for v in result["forecast"]))

        # Simple restock suggestion
        total_forecast = sum(result["forecast"])
        if current_stock < total_forecast * 0.5:
            st.warning(
                f"Current stock ({current_stock}) is relatively low. "
                f"Estimated total demand over {periods} periods is approximately {total_forecast}."
            )
        else:
            st.info(
                f"Current stock ({current_stock}) appears sufficient. "
                f"Estimated total demand over {periods} periods is approximately {total_forecast}."
            )

        st.caption(f"Internal reference: Store ID {result['storeId']} | Department {result['department']}")

    except Exception as e:
        st.error(f"Forecast failed: {e}")

st.divider()
st.markdown("""
### Deployment Instructions
1. Upload this folder (containing `app.py`, `models/xgb_model.joblib`, and `requirements.txt`) to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app** and select your repository.
4. Set Main file path to `app.py`.
5. Click **Deploy**.
6. Once deployment completes, you will receive a public URL (e.g., `https://your-app-name.streamlit.app`).

**Note:** Forecast results are for reference only. Model features have been relatively scaled for this demo.
""")
