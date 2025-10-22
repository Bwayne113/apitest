# main.py
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
from io import BytesIO

# -------------------------------
# CONFIG
# -------------------------------
MODEL_ROOT = "/dbfs/models/dbcu_forecast/"  # Must match your save path
NEXT_RESERVATION_START = datetime(2025, 10, 28)
RESERVATION_DAYS = 365
DBU_COST_PER_UNIT = 1.0
RESERVATION_DISCOUNT = 0.30

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="DBCU Forecast API", version="1.0")

# -------------------------------
# Pydantic Models
# -------------------------------
class ForecastRequest(BaseModel):
    meter_group: str

class ForecastResponse(BaseModel):
    meter_group: str
    model_used: str
    total_dbcu_forecast: float
    recommended_reservation: float
    expected_utilization_pct: float
    daily_forecast: List[Dict[str, Any]]
    plot_base64: str
    generated_at: str

# -------------------------------
# Helper: Load model + metadata
# -------------------------------
def load_model_and_meta(meter_group: str):
    safe_name = meter_group.replace(" ", "_").replace("/", "_")
    model_dir = os.path.join(MODEL_ROOT, safe_name)
    model_path = os.path.join(model_dir, "model.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for {meter_group}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta

# -------------------------------
# Helper: Recursive forecast (ML models)
# -------------------------------
def recursive_forecast(model, last_features: pd.Series, steps: int, start_date: datetime):
    preds = []
    current = last_features.to_dict()
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
        'rolling_7', 'rolling_30', 'day_of_week', 'day_of_month', 'month'
    ]

    for i in range(steps):
        X = pd.DataFrame([current])
        if hasattr(model, "predict"):  # scikit-learn style
            pred = model.predict(X)[0]
        else:  # GAM or others
            pred = model.predict(X.values)[0]
        preds.append(max(0, pred))

        # Update lags
        for lag in [1, 2, 3, 7, 14, 30]:
            key = f'lag_{lag}'
            if lag == 1:
                current[key] = pred
            elif f'lag_{lag-1}' in current:
                current[key] = current[f'lag_{lag-1}']

        # Update rolling
        recent = preds[-7:] if len(preds) >= 7 else preds
        current['rolling_7'] = np.mean(recent) if recent else pred
        recent30 = preds[-30:] if len(preds) >= 30 else preds
        current['rolling_30'] = np.mean(recent30) if recent30 else pred

        # Update date features
        date = start_date + timedelta(days=i)
        current['day_of_week'] = date.weekday()
        current['day_of_month'] = date.day
        current['month'] = date.month

    return np.array(preds), [start_date + timedelta(days=i) for i in range(steps)]

# -------------------------------
# Helper: Prophet forecast (if best model was Prophet)
# -------------------------------
def prophet_forecast(model, start_date: datetime, steps: int):
    future = pd.DataFrame({
        'ds': [start_date + timedelta(days=i) for i in range(steps)]
    })
    forecast = model.predict(future)
    return forecast['yhat'].values, future['ds'].tolist()

# -------------------------------
# Helper: Generate Plot (base64)
# -------------------------------
def generate_plot_base64(historical_df: pd.DataFrame, bridge_dates, bridge_vals,
                         res_dates, res_vals, meter_group: str, model_name: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(historical_df['usagedate'], historical_df['usedquantity'],
            label='Historical', color='blue', linewidth=1.5)
    if len(bridge_vals) > 0:
        ax.plot(bridge_dates, bridge_vals, label='Bridge', color='orange', linestyle='--')
    ax.plot(res_dates, res_vals, label='Forecast (365d)', color='green', linestyle='-', linewidth=2)

    ax.axvline(NEXT_RESERVATION_START, color='purple', linestyle='-', alpha=0.7, label='Next Res. Start')
    ax.set_title(f"DBCU Forecast: {meter_group} | Model: {model_name}")
    ax.set_ylabel("Daily DBCU")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/forecast", response_model=ForecastResponse)
def forecast_dbcu(request: ForecastRequest):
    meter_group = request.meter_group.strip()

    # 1. Load model + meta
    try:
        model, meta = load_model_and_meta(meter_group)
    except HTTPException as e:
        raise e

    model_name = meta["model_name"]
    model_type = meta["model_type"]

    # 2. Load historical data (same query logic — simplified)
    #     We assume you have the same Spark table available
    spark = SparkSession.builder.getOrCreate()
    sql = f"""
    SELECT to_date(`Date`) AS usagedate, SUM(Total_Quantity) AS usedquantity
    FROM prd_slz_enrich_azcost.tp_az_cost.azcost_ti_infra_delta
    WHERE `MeterSubCategory` = '{meter_group}'
      AND year(`Date`) >= 2024
    GROUP BY `Date`
    ORDER BY `Date`
    """
    try:
        df = spark.sql(sql).toPandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {meter_group}")

    df['usagedate'] = pd.to_datetime(df['usagedate'])
    df = df.sort_values('usagedate').reset_index(drop=True)

    # 3. Clean anomalies (reuse function if defined, else skip)
    try:
        from pyspark.sql import functions as F
        # If detect_and_clean_anomalies is in scope, use it
        _, df_clean, _ = detect_and_clean_anomalies(df, method="Isolation Forest", contamination=0.10)
        df = df_clean
    except:
        pass  # skip if not available

    # 4. Feature engineering: recreate last feature row
    df_ml = df.copy()
    for lag in [1,2,3,7,14,30]:
        df_ml[f'lag_{lag}'] = df_ml['usedquantity'].shift(lag)
    df_ml['rolling_7'] = df_ml['usedquantity'].rolling(7).mean()
    df_ml['rolling_30'] = df_ml['usedquantity'].rolling(30).mean()
    df_ml['day_of_week'] = df_ml['usagedate'].dt.dayofweek
    df_ml['day_of_month'] = df_ml['usagedate'].dt.day
    df_ml['month'] = df_ml['usagedate'].dt.month
    df_ml = df_ml.dropna().reset_index(drop=True)

    if len(df_ml) == 0:
        raise HTTPException(status_code=500, detail="Not enough data after cleaning")

    last_features = df_ml.iloc[-1]
    last_historical_date = df['usagedate'].iloc[-1]

    # 5. Forecast
    bridge_days = max(0, (NEXT_RESERVATION_START - last_historical_date).days)
    total_steps = bridge_days + RESERVATION_DAYS

    if model_type == "ML":
        full_pred, full_dates = recursive_forecast(
            model, last_features, total_steps, last_historical_date + timedelta(days=1)
        )
    elif model_name == "Prophet":
        full_pred, full_dates = prophet_forecast(model, last_historical_date + timedelta(days=1), total_steps)
    else:  # TS models (ARIMA, etc.)
        try:
            full_pred = model.forecast(steps=total_steps)
            full_dates = [last_historical_date + timedelta(days=i+1) for i in range(total_steps)]
        except:
            raise HTTPException(status_code=500, detail="TS model forecast failed")

    full_pred = np.maximum(full_pred, 0)

    bridge_pred = full_pred[:bridge_days] if bridge_days > 0 else np.array([])
    bridge_dates = full_dates[:bridge_days] if bridge_days > 0 else []
    res_pred = full_pred[bridge_days:bridge_days + RESERVATION_DAYS]
    res_dates = full_dates[bridge_days:bridge_days + RESERVATION_DAYS]

    total_dbcu = float(np.sum(res_pred))
    recommended = total_dbcu  # 100% target
    utilization_pct = 100.0

    # 6. Build daily forecast list
    daily_forecast = [
        {"date": d.strftime("%Y-%m-%d"), "dbcus": round(float(p), 2)}
        for d, p in zip(res_dates, res_pred)
    ]

    # 7. Generate plot
    plot_b64 = generate_plot_base64(
        historical_df=df,
        bridge_dates=bridge_dates,
        bridge_vals=bridge_pred,
        res_dates=res_dates,
        res_vals=res_pred,
        meter_group=meter_group,
        model_name=model_name
    )

    # 8. Response
    return ForecastResponse(
        meter_group=meter_group,
        model_used=model_name,
        total_dbcu_forecast=round(total_dbcu, 2),
        recommended_reservation=round(recommended, 0),
        expected_utilization_pct=utilization_pct,
        daily_forecast=daily_forecast,
        plot_base64=plot_b64,
        generated_at=datetime.now().isoformat()
    )

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def root():
    return {"message": "DBCU Forecast API is running", "docs": "/docs"}
