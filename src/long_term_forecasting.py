"""
Long-Term Air Quality Forecasting Module
Performs multi-month time-series forecasting for PM2.5 and AQI using:
- Seasonal Naive (baseline)
- Holt-Winters Exponential Smoothing (additive trend & seasonality, period=12)
- Seasonal ARIMA / SARIMA (p,d,q) x (P,D,Q,12) with confidence intervals
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation import calculate_forecast_metrics
from src.chart_theme import apply_chart_theme

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Indian Central Pollution Control Board (CPCB) AQI Categorization
# ---------------------------------------------------------------------
def categorize_aqi(aqi_val):
    """
    Categorizes AQI value according to official Indian National AQI thresholds:
    - 0–50: Good (Minimal impact)
    - 51–100: Satisfactory (Minor breathing discomfort to sensitive people)
    - 101–200: Moderate (Breathing discomfort to people with asthma/heart disease)
    - 201–300: Poor (Breathing discomfort to most people on prolonged exposure)
    - 301–400: Very Poor (Respiratory illness to people on prolonged exposure)
    - 401–500: Severe (Affects healthy people and seriously impacts sensitive groups)
    """
    if np.isnan(aqi_val):
        return "Unknown", "#94A3B8", "No data available."
    val = float(aqi_val)
    if val <= 50:
        return "Good", "#22C55E", "Minimal health impact."
    elif val <= 100:
        return "Satisfactory", "#84CC16", "Minor breathing discomfort to sensitive individuals."
    elif val <= 200:
        return "Moderate", "#EAB308", "Breathing discomfort to people with lungs, asthma, and heart diseases."
    elif val <= 300:
        return "Poor", "#F97316", "Breathing discomfort to most people on prolonged exposure."
    elif val <= 400:
        return "Very Poor", "#EF4444", "Respiratory illness on prolonged exposure."
    else:
        return "Severe", "#7F1D1D", "Affects healthy people and seriously impacts those with existing conditions."


def categorize_target(val, target_col="PM2.5"):
    """Categorizes either PM2.5 or AQI based on standard CPCB thresholds."""
    if target_col == "AQI":
        return categorize_aqi(val)
    # PM2.5 thresholds (CPCB 24-hr standard)
    if np.isnan(val):
        return "Unknown", "#94A3B8", "No data available."
    v = float(val)
    if v <= 30:
        return "Good", "#22C55E", "Air quality is good."
    elif v <= 60:
        return "Satisfactory", "#84CC16", "Acceptable air quality."
    elif v <= 90:
        return "Moderate", "#EAB308", "Moderate air quality."
    elif v <= 120:
        return "Poor", "#F97316", "Poor air quality."
    elif v <= 250:
        return "Very Poor", "#EF4444", "Very poor air quality."
    else:
        return "Severe", "#7F1D1D", "Severe hazard."


# ---------------------------------------------------------------------
# Monthly Series Preparation
# ---------------------------------------------------------------------
def prepare_monthly_series(df, city, target_col="PM2.5"):
    """
    Extracts and aggregates daily observations into continuous monthly averages (MS frequency).
    Handles missing months via time-based interpolation and forward/backward filling.
    """
    city_data = df[df["City"] == city].copy() if "City" in df.columns else df.copy()

    if "Date" not in city_data.columns or target_col not in city_data.columns:
        raise ValueError(f"Required columns 'Date' and '{target_col}' not found.")

    city_data["Date"] = pd.to_datetime(city_data["Date"], errors="coerce")
    city_data = city_data.dropna(subset=["Date"]).sort_values(by="Date")

    # Filter out rows where target is invalid
    city_data[target_col] = pd.to_numeric(city_data[target_col], errors="coerce")
    city_data = city_data.set_index("Date")

    # Resample to Month Start (MS) using mean
    monthly = city_data[target_col].resample("MS").mean()
    valid_count = monthly.dropna().count()

    if valid_count < 18:
        # Need at least 18 months for meaningful seasonal split
        return None

    # Continuous monthly index from min to max date
    full_range = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq="MS")
    monthly_reindexed = monthly.reindex(full_range)

    # Interpolate missing months
    monthly_imputed = monthly_reindexed.interpolate(method="time").ffill().bfill()
    monthly_imputed.name = target_col
    return monthly_imputed


# ---------------------------------------------------------------------
# 1. Seasonal Naive Forecasting Model
# ---------------------------------------------------------------------
def forecast_seasonal_naive(train_series, test_series, horizon_months=6):
    """
    Seasonal Naive Forecast:
    Projects future months using the observed value from the same month 12 months prior.
    Serves as an essential baseline for seasonal time-series evaluation.
    """
    full_series = pd.concat([train_series, test_series])

    # 1. Test set backtesting
    y_pred_test = []
    for dt in test_series.index:
        past_dt = dt - pd.DateOffset(years=1)
        if past_dt in full_series.index:
            y_pred_test.append(float(full_series.loc[past_dt]))
        else:
            y_pred_test.append(float(train_series.iloc[-1]))
    y_pred_test = np.clip(np.array(y_pred_test), 0, None)

    metrics = calculate_forecast_metrics(test_series.values, y_pred_test, model_name="Seasonal Naive (12M Baseline)")

    # 2. Future forecast projection
    last_date = full_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

    future_preds = []
    for dt in future_dates:
        past_dt = dt - pd.DateOffset(years=1)
        if past_dt in full_series.index:
            future_preds.append(float(full_series.loc[past_dt]))
        else:
            future_preds.append(float(full_series.tail(12).mean()))

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecast_Value": np.round(np.clip(future_preds, 0, None), 2)
    })

    return {
        "model_name": "Seasonal Naive (12M Baseline)",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df
    }


# ---------------------------------------------------------------------
# 2. Holt-Winters Exponential Smoothing Model
# ---------------------------------------------------------------------
def forecast_holt_winters(train_series, test_series, horizon_months=6):
    """
    Holt-Winters Exponential Smoothing:
    Models level, trend, and seasonal components (seasonal_periods=12).
    """
    # 1. Backtesting on test period
    try:
        hw_train = ExponentialSmoothing(
            train_series,
            seasonal_periods=12,
            trend="add",
            seasonal="add",
            initialization_method="estimated"
        ).fit(optimized=True)
        y_pred_test = np.clip(hw_train.forecast(len(test_series)).values, 0, None)
    except Exception:
        # Fallback to simple trend without seasonal if series is short
        try:
            hw_train = ExponentialSmoothing(train_series, trend="add").fit(optimized=True)
            y_pred_test = np.clip(hw_train.forecast(len(test_series)).values, 0, None)
        except Exception:
            y_pred_test = np.full(len(test_series), float(train_series.tail(6).mean()))

    metrics = calculate_forecast_metrics(test_series.values, y_pred_test, model_name="Holt-Winters Exponential Smoothing")

    # 2. Future projection using full historical series
    full_series = pd.concat([train_series, test_series])
    try:
        hw_full = ExponentialSmoothing(
            full_series,
            seasonal_periods=12,
            trend="add",
            seasonal="add",
            initialization_method="estimated"
        ).fit(optimized=True)
        future_preds = np.clip(hw_full.forecast(horizon_months).values, 0, None)
    except Exception:
        rolling_mean = float(full_series.tail(12).mean())
        future_preds = np.full(horizon_months, max(0.0, rolling_mean))

    last_date = full_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecast_Value": np.round(future_preds, 2)
    })

    return {
        "model_name": "Holt-Winters Exponential Smoothing",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df
    }


# ---------------------------------------------------------------------
# 3. Seasonal ARIMA (SARIMA) Model
# ---------------------------------------------------------------------
def forecast_sarima(train_series, test_series, horizon_months=6, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)):
    """
    Seasonal ARIMA (SARIMA):
    Captures non-seasonal autoregression, differencing, moving average, and seasonal lag components.
    Provides 95% forecast confidence intervals.
    """
    # 1. Backtesting on test set
    try:
        sarima_train = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        forecast_res_test = sarima_train.get_forecast(len(test_series))
        y_pred_test = np.clip(forecast_res_test.predicted_mean.values, 0, None)
    except Exception:
        # Fallback to (1,0,1)(0,0,0,0) if seasonal fitting diverges
        try:
            sarima_train = SARIMAX(train_series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            y_pred_test = np.clip(sarima_train.get_forecast(len(test_series)).predicted_mean.values, 0, None)
        except Exception:
            y_pred_test = np.full(len(test_series), float(train_series.tail(6).mean()))

    metrics = calculate_forecast_metrics(test_series.values, y_pred_test, model_name=f"SARIMA{order}x{seasonal_order}")

    # 2. Future projection with full series and confidence bounds
    full_series = pd.concat([train_series, test_series])
    try:
        sarima_full = SARIMAX(
            full_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        forecast_res = sarima_full.get_forecast(horizon_months)
        future_preds = np.clip(forecast_res.predicted_mean.values, 0, None)
        conf_int = forecast_res.conf_int(alpha=0.05)
        lower_bound = np.clip(conf_int.iloc[:, 0].values, 0, None)
        upper_bound = np.clip(conf_int.iloc[:, 1].values, 0, None)
    except Exception:
        mean_val = float(full_series.tail(12).mean())
        std_val = float(full_series.tail(12).std())
        future_preds = np.full(horizon_months, mean_val)
        lower_bound = np.clip(future_preds - 1.96 * std_val, 0, None)
        upper_bound = future_preds + 1.96 * std_val

    last_date = full_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecast_Value": np.round(future_preds, 2),
        "Lower_CI_95%": np.round(lower_bound, 2),
        "Upper_CI_95%": np.round(upper_bound, 2)
    })

    return {
        "model_name": f"SARIMA{order}x{seasonal_order}",
        "metrics": metrics,
        "y_pred_test": y_pred_test,
        "forecast_df": forecast_df
    }


# ---------------------------------------------------------------------
# Unified Long-Term Pipeline
# ---------------------------------------------------------------------
def run_all_long_term_models(monthly_series, test_months=6, horizon_months=6, target_col="PM2.5"):
    """
    Executes Seasonal Naive, Holt-Winters, and SARIMA side-by-side on monthly data.
    Splits the last `test_months` for chronological out-of-sample validation.
    """
    if len(monthly_series) < test_months + 12:
        raise ValueError(
            f"Insufficient monthly data: found {len(monthly_series)} months, "
            f"but at least {test_months + 12} months are needed for valid seasonal evaluation."
        )

    train_series = monthly_series.iloc[:-test_months]
    test_series = monthly_series.iloc[-test_months:]

    # Run the three models
    naive_res = forecast_seasonal_naive(train_series, test_series, horizon_months=horizon_months)
    hw_res = forecast_holt_winters(train_series, test_series, horizon_months=horizon_months)
    sarima_res = forecast_sarima(train_series, test_series, horizon_months=horizon_months)

    comparison_df = pd.DataFrame([
        naive_res["metrics"],
        hw_res["metrics"],
        sarima_res["metrics"]
    ])

    return {
        "train_series": train_series,
        "test_series": test_series,
        "seasonal_naive": naive_res,
        "holt_winters": hw_res,
        "sarima": sarima_res,
        "comparison_table": comparison_df,
        "horizon_months": horizon_months,
        "test_months": test_months,
        "target_col": target_col,
        "last_observed_date": monthly_series.index.max().strftime("%B %Y")
    }


# ---------------------------------------------------------------------
# Visualization (Green & Yellow Palette)
# ---------------------------------------------------------------------
def plot_long_term_forecast(results_dict, selected_model_key="sarima", city_name="Selected City"):
    """
    Plotly visualization for long-term forecasting:
    - Historical monthly trend: Forest Green (#14532D / #166534)
    - Test period actual: Secondary Green (#166534)
    - Test period model prediction: Soft Green dashed line
    - Future forecast: Accent Yellow (#EAB308) with markers
    - Confidence Interval: Translucent Yellow shaded band
    """
    train_series = results_dict["train_series"]
    test_series = results_dict["test_series"]
    model_res = results_dict[selected_model_key]
    forecast_df = model_res["forecast_df"]
    target_col = results_dict["target_col"]
    unit_str = "µg/m³" if target_col == "PM2.5" else "AQI Index"

    fig = go.Figure()

    # 1. Historical monthly data (last 36 months for clear visualization)
    hist_subset = train_series.tail(36)
    fig.add_trace(go.Scatter(
        x=hist_subset.index,
        y=hist_subset.values,
        mode="lines+markers",
        name="Historical Monthly (Train)",
        line=dict(color="#166534", width=2),
        marker=dict(size=4, color="#14532D"),
        hovertemplate="<b>%{x|%b %Y}:</b> %{y:.1f} " + unit_str + "<extra></extra>"
    ))

    # 2. Test Period Actuals
    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series.values,
        mode="lines+markers",
        name="Actual (Test Period)",
        line=dict(color="#14532D", width=2.5),
        marker=dict(size=6, color="#14532D"),
        hovertemplate="<b>Actual %{x|%b %Y}:</b> %{y:.1f} " + unit_str + "<extra></extra>"
    ))

    # 3. Test Period Predicted
    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=model_res["y_pred_test"],
        mode="lines",
        name=f"{model_res['model_name']} (Test Fit)",
        line=dict(color="#22C55E", width=2, dash="dash"),
        hovertemplate="<b>Backtest %{x|%b %Y}:</b> %{y:.1f} " + unit_str + "<extra></extra>"
    ))

    # 4. Future Forecast Line (Accent Yellow)
    fig.add_trace(go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["Forecast_Value"],
        mode="lines+markers",
        name=f"Future Forecast ({len(forecast_df)}M)",
        line=dict(color="#EAB308", width=3),
        marker=dict(size=7, symbol="diamond", color="#CA8A04"),
        hovertemplate="<b>Forecast %{x|%b %Y}:</b> %{y:.1f} " + unit_str + "<extra></extra>"
    ))

    # 5. Confidence Interval Band (if available for SARIMA)
    if "Lower_CI_95%" in forecast_df.columns and "Upper_CI_95%" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df["Month"]) + list(forecast_df["Month"])[::-1],
            y=list(forecast_df["Upper_CI_95%"]) + list(forecast_df["Lower_CI_95%"])[::-1],
            fill="toself",
            fillcolor="rgba(234, 179, 8, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

    # Reference limit lines
    if target_col == "PM2.5":
        fig.add_hline(y=60, line_dash="dot", line_color="#22C55E",
                      annotation_text="NAAQS Safe Limit (60 µg/m³)",
                      annotation_position="bottom right")
    elif target_col == "AQI":
        fig.add_hline(y=100, line_dash="dot", line_color="#84CC16",
                      annotation_text="Satisfactory Limit (100)",
                      annotation_position="bottom right")
        fig.add_hline(y=200, line_dash="dot", line_color="#EAB308",
                      annotation_text="Moderate Limit (200)",
                      annotation_position="top right")

    lt_title = f"Long-Term {target_col} Forecast for {city_name} — {model_res['model_name']} ({len(forecast_df)} Months Ahead)"
    apply_chart_theme(fig, title=lt_title, x_title="Month", y_title=f"Monthly Average {target_col} ({unit_str})")
    return fig

