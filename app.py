"""
Smart Inventory Forecast - Enhanced Streamlit Version
Closer to the original SmartInvPredict features:
- Home / Forecast / History pages
- Full input form with CSV import
- Historical vs forecast chart
- Smart restock / promotion alerts
- Session-based history
Uses the original XGBoost model.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_model.joblib"
MODEL_DOMAIN_MEAN = 20000.0


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_cols: list[str] = []
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feature_cols = [str(x) for x in model.feature_names_in_]
    elif hasattr(model, "get_booster"):
        feature_cols = [str(x) for x in (model.get_booster().feature_names or [])]
    if not feature_cols:
        raise RuntimeError("Model has no feature names")
    return model, feature_cols


# ---------------------------------------------------------------------------
# Helpers (same logic as original ml_server.py)
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
) -> dict[str, Any]:
    if len(past_sales) < 3:
        raise ValueError("At least 3 historical sales values are required")

    periods = max(3, min(12, periods))
    store_id = _hash_id(sku, 1, 45)
    department = _hash_id(product_name, 1, 99)
    store_size = 150000.0

    user_recent = past_sales[-min(4, len(past_sales)) :]
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
    total_demand = sum(preds_int)
    peak_idx = int(np.argmax(preds_int)) + 1 if preds_int else 1

    # Simple confidence based on history length and volatility
    volatility = float(np.std(past_sales)) / max(user_mean, 1e-6) if len(past_sales) > 1 else 0.5
    confidence = max(0.55, min(0.92, 0.85 - volatility * 0.3 + min(len(past_sales), 12) * 0.015))
    level = "high" if confidence >= 0.78 else "medium" if confidence >= 0.62 else "low"

    return {
        "forecast": preds_int,
        "changePercent": change_pct,
        "recentAvg": round(recent_avg, 1),
        "forecastAvg": round(forecast_avg, 1),
        "totalDemand": total_demand,
        "peakPeriod": peak_idx,
        "confidence": round(confidence, 2),
        "confidenceLevel": level,
        "storeId": store_id,
        "department": department,
        "pastSales": past_sales,
    }


def generate_alerts(result: dict, current_stock: int, lead_time: int, has_promotion: bool) -> list[dict]:
    alerts = []
    total = result["totalDemand"]
    change = result["changePercent"]
    coverage = current_stock / max(result["forecastAvg"], 1) if result["forecastAvg"] else 0

    if current_stock < total * 0.6 or coverage < lead_time / 7:
        alerts.append(
            {
                "type": "restock",
                "title": "Restock Recommended",
                "message": f"Current stock ({current_stock}) may not cover the forecasted demand of {total} units. "
                           f"Consider placing an order, especially with lead time of {lead_time} days.",
            }
        )
    if change > 15 and not has_promotion:
        alerts.append(
            {
                "type": "promote",
                "title": "Demand Rising",
                "message": f"Forecast shows a {change:+.1f}% increase. A promotion could help capture higher demand.",
            }
        )
    if change < -15:
        alerts.append(
            {
                "type": "caution",
                "title": "Demand Softening",
                "message": f"Forecast indicates a {change:.1f}% decline. Review inventory levels to avoid overstock.",
            }
        )
    if not alerts:
        alerts.append(
            {
                "type": "ok",
                "title": "Stable Outlook",
                "message": "Demand appears stable relative to current stock. Continue monitoring.",
            }
        )
    return alerts


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "current_input" not in st.session_state:
        st.session_state.current_input = None


def save_to_history(input_data: dict, result: dict):
    entry = {
        "id": len(st.session_state.history) + 1,
        "timestamp": date.today().isoformat(),
        "input": input_data,
        "result": result,
    }
    st.session_state.history.insert(0, entry)
    # Keep last 20
    st.session_state.history = st.session_state.history[:20]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("📦 Smart Inventory Forecast")
    st.caption("Intelligent Inventory Forecast & Restock Alerts · Demo")

    st.markdown("""
    ### Welcome
    This application uses an **XGBoost model** to generate multi-step demand forecasts from your historical sales data.
    Enter product information and past sales, then receive charts, smart alerts, and restock suggestions.

    **How it works**
    1. **Data Input** — Product details, current stock, and historical sales (manual or CSV).
    2. **XGBoost Forecast** — Multi-step prediction mapped to your sales scale.
    3. **Charts** — Historical actuals vs. forecasted demand.
    4. **Smart Alerts** — Restock or promotion recommendations based on stock coverage and trend.
    5. **History** — Session-based record of previous forecasts for review and comparison.

    Use the sidebar to navigate to **Forecast** or **History**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Forecasting", type="primary", use_container_width=True):
            st.session_state.page = "Forecast"
            st.rerun()
    with col2:
        if st.button("View History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()


def page_forecast(model, feature_cols):
    st.title("Forecast")
    st.caption("Enter product and sales data to generate an XGBoost demand forecast")

    with st.form("forecast_form"):
        c1, c2 = st.columns(2)
        with c1:
            product_name = st.text_input("Product Name", value="Classic Black Coffee Beans 500g")
            current_stock = st.number_input("Current Stock (units)", min_value=0, value=48, step=1)
        with c2:
            sku = st.text_input("SKU", value="COF-500-BLK")
            unit_cost = st.number_input("Unit Cost", min_value=0.0, value=22.5, step=0.5)

        past_sales_raw = st.text_area(
            "Historical Sales (oldest → newest, comma-separated)",
            value="22, 25, 24, 28, 30, 27, 32, 35",
            height=100,
            help="At least 3 numbers. Commas, spaces, or newlines accepted.",
        )

        uploaded = st.file_uploader("Or import past sales from CSV", type=["csv"], help="First column or column named 'sales' will be used.")

        c3, c4, c5, c6 = st.columns(4)
        with c3:
            granularity = st.selectbox("Granularity", ["week", "month"], format_func=lambda x: "Week" if x == "week" else "Month")
        with c4:
            periods = st.slider("Forecast Periods", 3, 12, 6)
        with c5:
            season = st.selectbox(
                "Season Factor",
                ["none", "peak", "low", "holiday"],
                format_func=lambda x: {"none": "None", "peak": "Peak", "low": "Low", "holiday": "Holiday"}[x],
            )
        with c6:
            lead_time = st.number_input("Lead Time (days)", min_value=0, value=7, step=1)

        has_promotion = st.checkbox("Currently running a promotion", value=False)

        submitted = st.form_submit_button("Generate Forecast", type="primary", use_container_width=True)

    if submitted or st.session_state.get("rerun_data"):
        # Handle CSV if uploaded
        sales_text = past_sales_raw
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if "sales" in df.columns:
                    vals = df["sales"].dropna().tolist()
                else:
                    vals = df.iloc[:, 0].dropna().tolist()
                sales_text = ", ".join(str(v) for v in vals)
                st.info(f"Loaded {len(vals)} periods from CSV.")
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        try:
            past = [
                max(0.0, float(x.strip()))
                for x in sales_text.replace("，", ",").replace("\n", ",").split(",")
                if x.strip()
            ]
            result = multi_step_forecast(
                model, feature_cols, past, periods, granularity, season, has_promotion, sku, product_name
            )
            input_data = {
                "productName": product_name,
                "sku": sku,
                "currentStock": current_stock,
                "pastSalesRaw": sales_text,
                "granularity": granularity,
                "periods": periods,
                "season": season,
                "hasPromotion": has_promotion,
                "leadTime": lead_time,
                "unitCost": unit_cost,
            }
            st.session_state.current_result = result
            st.session_state.current_input = input_data
            save_to_history(input_data, result)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            return

    if st.session_state.current_result:
        result = st.session_state.current_result
        inp = st.session_state.current_input

        st.subheader("Results")
        st.write(f"**{inp['productName']}** · {inp['sku']}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Change", f"{result['changePercent']:+.1f}%")
        m2.metric("Total Demand", f"{result['totalDemand']}")
        m3.metric("Avg Forecast", f"{result['forecastAvg']}")
        m4.metric("Peak Period", f"#{result['peakPeriod']}")
        m5.metric("Confidence", f"{int(result['confidence']*100)}% ({result['confidenceLevel']})")

        # Chart: history + forecast
        hist_len = len(result["pastSales"])
        labels = [f"H{i+1}" for i in range(hist_len)] + [f"F{i+1}" for i in range(len(result["forecast"]))]
        values = result["pastSales"] + result["forecast"]
        chart_df = pd.DataFrame({"Period": labels, "Sales": values})
        st.line_chart(chart_df.set_index("Period"), height=350)
        st.caption("H = Historical · F = Forecast")

        # Alerts
        alerts = generate_alerts(result, inp["currentStock"], inp["leadTime"], inp["hasPromotion"])
        st.subheader("Smart Alerts")
        for a in alerts:
            if a["type"] == "restock":
                st.warning(f"**{a['title']}** — {a['message']}")
            elif a["type"] == "promote":
                st.info(f"**{a['title']}** — {a['message']}")
            elif a["type"] == "caution":
                st.error(f"**{a['title']}** — {a['message']}")
            else:
                st.success(f"**{a['title']}** — {a['message']}")

        with st.expander("Forecast values"):
            st.write(", ".join(str(v) for v in result["forecast"]))


def page_history():
    st.title("History")
    st.caption("Session-based forecast history (cleared when the app restarts)")

    if not st.session_state.history:
        st.info("No forecasts yet. Go to the Forecast page to generate one.")
        return

    for entry in st.session_state.history:
        with st.expander(
            f"#{entry['id']} · {entry['input']['productName']} · {entry['timestamp']} · "
            f"Change {entry['result']['changePercent']:+.1f}%"
        ):
            r = entry["result"]
            st.write(f"SKU: {entry['input']['sku']} | Stock: {entry['input']['currentStock']}")
            st.write(f"Total demand: {r['totalDemand']} | Avg: {r['forecastAvg']} | Peak: period {r['peakPeriod']}")
            st.write("Forecast:", ", ".join(str(v) for v in r["forecast"]))
            if st.button("Delete", key=f"del_{entry['id']}"):
                st.session_state.history = [h for h in st.session_state.history if h["id"] != entry["id"]]
                st.rerun()

    if st.button("Clear all history"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Inventory Forecast", page_icon="📦", layout="wide")
    init_session()

    try:
        model, feature_cols = load_model()
    except Exception as e:
        st.error(f"Failed to load XGBoost model: {e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Smart Inventory Forecast")
    st.sidebar.caption("XGBoost-powered demo")
    page = st.sidebar.radio("Navigate", ["Home", "Forecast", "History"], index=["Home", "Forecast", "History"].index(st.session_state.get("page", "Home")))
    st.session_state.page = page

    st.sidebar.divider()
    st.sidebar.success(f"Model loaded · {len(feature_cols)} features")

    if page == "Home":
        page_home()
    elif page == "Forecast":
        page_forecast(model, feature_cols)
    else:
        page_history()

    st.sidebar.divider()
    st.sidebar.markdown("**Note:** Forecasts are for reference only.")


if __name__ == "__main__":
    main()
