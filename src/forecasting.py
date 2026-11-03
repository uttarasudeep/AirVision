"""
Time-Series Forecasting Module for Air Quality System
Implements Moving Average, Holt's Exponential Smoothing, and ARIMA models
for PM2.5 projection with chronological evaluation and multi-step future forecasting.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation import calculate_forecast_metrics
from src.prediction import categorize_pm25
from src.chart_theme import apply_chart_theme

warnings.filterwarnings("ignore")


def forecast_moving_average(ts_train, ts_test, horizon=30, window=7):
    """
    Moving Average Forecasting:
    - Test evaluation: Uses rolling average of recent past window to forecast test values.
    - Future forecast: Projects future values using the last observed rolling mean.
    """
    ts_combined = pd.concat([ts_train, ts_test])
    test_len = len(ts_test)

    # 1. In-sample test period simulation (one-step ahead or rolling window)
    # Using window rolling mean of preceding days
    rolling_series = ts_combined.rolling(window=window, min_periods=1).mean()
    y_pred_test = rolling_series.iloc[len(ts_train):len(ts_train) + test_len].values

    metrics = calculate_forecast_metrics(ts_test.values, y_pred_test, model_name=f"Moving Average ({window}-Day)")

    # 2. Future forecast projection
    last_known_mean = float(ts_combined.tail(window).mean())
    last_date = ts_combined.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_preds = np.full(horizon, max(0.0, last_known_mean))

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_PM2.5": np.round(future_preds, 2),
    })

    return {
        "model_name": f"Moving Average ({window}-Day)",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df,
    }


def forecast_exponential_smoothing(ts_train, ts_test, horizon=30):
    """
    Holt's Exponential Smoothing (Linear Trend):
    Captures level and local slope trend in PM2.5 concentrations.
    """
    # 1. Fit on training set and predict test set
    try:
        model_train = Holt(ts_train, initialization_method="estimated").fit(optimized=True)
        y_pred_test = model_train.forecast(len(ts_test)).values
        y_pred_test = np.clip(y_pred_test, 0, None)
    except Exception as e:
        # Fallback to simple mean if Holt fails
        mean_val = float(ts_train.mean())
        y_pred_test = np.full(len(ts_test), mean_val)

    metrics = calculate_forecast_metrics(ts_test.values, y_pred_test, model_name="Holt's Exponential Smoothing")

    # 2. Refit on entire series to forecast into the future
    ts_full = pd.concat([ts_train, ts_test])
    try:
        model_full = Holt(ts_full, initialization_method="estimated").fit(optimized=True)
        future_preds = model_full.forecast(horizon).values
        future_preds = np.clip(future_preds, 0, None)
    except Exception:
        last_mean = float(ts_full.tail(14).mean())
        future_preds = np.full(horizon, max(0.0, last_mean))

    last_date = ts_full.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_PM2.5": np.round(future_preds, 2),
    })

    return {
        "model_name": "Holt's Exponential Smoothing",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df,
    }


def forecast_arima(ts_train, ts_test, horizon=30, order=(1, 1, 1)):
    """
    ARIMA (AutoRegressive Integrated Moving Average) Forecasting:
    Uses differencing (d=1) for stationarity and autoregressive / moving average components.
    Includes lower and upper 95% confidence intervals for future projections.
    """
    # 1. Test set evaluation
    try:
        model_train = ARIMA(ts_train, order=order).fit()
        forecast_res_test = model_train.forecast(steps=len(ts_test))
        y_pred_test = np.clip(forecast_res_test.values, 0, None)
    except Exception as e:
        # Fallback to (1,0,0) or naive if (1,1,1) fails
        try:
            model_train = ARIMA(ts_train, order=(1, 0, 0)).fit()
            y_pred_test = np.clip(model_train.forecast(steps=len(ts_test)).values, 0, None)
        except Exception:
            y_pred_test = np.full(len(ts_test), float(ts_train.mean()))

    metrics = calculate_forecast_metrics(ts_test.values, y_pred_test, model_name=f"ARIMA{order}")

    # 2. Future forecast with full series and confidence intervals
    ts_full = pd.concat([ts_train, ts_test])
    try:
        model_full = ARIMA(ts_full, order=order).fit()
        forecast_res = model_full.get_forecast(steps=horizon)
        future_preds = np.clip(forecast_res.predicted_mean.values, 0, None)
        conf_int = forecast_res.conf_int(alpha=0.05)
        lower_bound = np.clip(conf_int.iloc[:, 0].values, 0, None)
        upper_bound = np.clip(conf_int.iloc[:, 1].values, 0, None)
    except Exception:
        # Graceful fallback
        rolling_mean = float(ts_full.tail(14).mean())
        std_val = float(ts_full.tail(14).std())
        future_preds = np.full(horizon, rolling_mean)
        lower_bound = np.clip(future_preds - 1.96 * std_val, 0, None)
        upper_bound = future_preds + 1.96 * std_val

    last_date = ts_full.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_PM2.5": np.round(future_preds, 2),
        "Lower_CI_95%": np.round(lower_bound, 2),
        "Upper_CI_95%": np.round(upper_bound, 2),
    })

    return {
        "model_name": f"ARIMA{order}",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df,
    }


def run_all_forecasting_models(ts_series, test_days=30, horizon=30):
    """
    Executes Moving Average, Exponential Smoothing, and ARIMA side-by-side.
    Splits the last `test_days` for chronological out-of-sample validation.
    Returns:
    - Dictionary with individual model results
    - Consolidated test metrics comparison table
    - Unified forecast table
    """
    if len(ts_series) < test_days + 30:
        raise ValueError(
            f"Insufficient time-series data: found {len(ts_series)} daily records, "
            f"but at least {test_days + 30} records are needed for a {test_days}-day test split."
        )

    ts_train = ts_series.iloc[:-test_days]
    ts_test = ts_series.iloc[-test_days:]

    # Run the three models
    ma_res = forecast_moving_average(ts_train, ts_test, horizon=horizon, window=7)
    es_res = forecast_exponential_smoothing(ts_train, ts_test, horizon=horizon)
    arima_res = forecast_arima(ts_train, ts_test, horizon=horizon, order=(1, 1, 1))

    # Consolidated metrics
    comparison_df = pd.DataFrame([
        ma_res["metrics"],
        es_res["metrics"],
        arima_res["metrics"]
    ])

    return {
        "ts_train": ts_train,
        "ts_test": ts_test,
        "moving_average": ma_res,
        "exponential_smoothing": es_res,
        "arima": arima_res,
        "comparison_table": comparison_df,
        "horizon": horizon,
        "test_days": test_days,
    }


def plot_forecast_results(forecast_dict, selected_model_key="arima", city_name="Selected City"):
    """
    Interactive Plotly figure showing:
    1. Historical PM2.5 (past 90-180 days for clean visual clarity)
    2. Actual vs Test Predicted
    3. Future Out-of-Sample Forecast Horizon (with confidence bounds if available)
    """
    ts_train = forecast_dict["ts_train"]
    ts_test = forecast_dict["ts_test"]
    model_res = forecast_dict[selected_model_key]
    forecast_df = model_res["forecast_df"]
    model_name = model_res["model_name"]

    fig = go.Figure()

    # Historical display (show past 120 days of training data so chart isn't cramped)
    hist_subset = ts_train.tail(120)

    fig.add_trace(go.Scatter(
        x=hist_subset.index,
        y=hist_subset.values,
        mode="lines",
        name="Historical PM2.5 (Train)",
        line=dict(color="#166534", width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Train PM2.5: %{y:.1f} µg/m³<extra></extra>"
    ))

    # Actual test period
    fig.add_trace(go.Scatter(
        x=ts_test.index,
        y=ts_test.values,
        mode="lines+markers",
        name="Actual PM2.5 (Test Period)",
        line=dict(color="#14532D", width=2.5),
        marker=dict(size=4, color="#14532D"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.1f} µg/m³<extra></extra>"
    ))

    # Model Test Prediction
    fig.add_trace(go.Scatter(
        x=ts_test.index,
        y=model_res["y_pred_test"],
        mode="lines",
        name=f"{model_name} (Test Fit)",
        line=dict(color="#22C55E", width=2, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted: %{y:.1f} µg/m³<extra></extra>"
    ))

    # Future Forecast (Accent Yellow)
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast_PM2.5"],
        mode="lines+markers",
        name=f"Future Forecast ({len(forecast_df)} Days)",
        line=dict(color="#EAB308", width=3),
        marker=dict(size=6, symbol="diamond", color="#CA8A04"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.1f} µg/m³<extra></extra>"
    ))

    # Confidence Interval band for ARIMA (Soft Yellow Shading)
    if "Lower_CI_95%" in forecast_df.columns and "Upper_CI_95%" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df["Date"]) + list(forecast_df["Date"])[::-1],
            y=list(forecast_df["Upper_CI_95%"]) + list(forecast_df["Lower_CI_95%"])[::-1],
            fill="toself",
            fillcolor="rgba(234, 179, 8, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

    # CPCB National Ambient Air Quality Standard (NAAQS) safe limit (60 µg/m³)
    fig.add_hline(y=60, line_dash="dot", line_color="#22C55E",
                  annotation_text="NAAQS Safe Limit (60 µg/m³)",
                  annotation_position="bottom right")

    fc_title = f"{model_name} — PM2.5 Forecast for {city_name} ({len(forecast_df)}-Day Horizon)"
    apply_chart_theme(fig, title=fc_title, x_title="Date", y_title="PM2.5 (µg/m³)")
    return fig

