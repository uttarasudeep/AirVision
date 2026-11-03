"""
Model Evaluation Module for Air Quality Prediction System
Computes performance metrics, generates diagnostic residual and actual-vs-predicted plots,
and provides explanations suitable for academic viva and project reporting.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.chart_theme import apply_chart_theme


def calculate_regression_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate regression evaluation metrics: MAE, MSE, RMSE, R².
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "Model": model_name,
        "MAE": np.round(mae, 3),
        "MSE": np.round(mse, 3),
        "RMSE": np.round(rmse, 3),
        "R2_Score": np.round(r2, 4),
        "R2_Pct": f"{np.round(r2 * 100, 2)}%"
    }


def calculate_forecast_metrics(y_true, y_pred, model_name="Forecasting Model"):
    """
    Calculate time-series forecasting evaluation metrics on held-out test horizon:
    MAE, RMSE, and MAPE (Mean Absolute Percentage Error).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"Model": model_name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Compute MAPE with epsilon guard to prevent division by near-zero pollution values
    epsilon = 1e-3
    nonzero_mask = np.abs(y_true) > epsilon
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    else:
        mape = np.nan

    return {
        "Model": model_name,
        "MAE": np.round(mae, 3),
        "RMSE": np.round(rmse, 3),
        "MAPE": f"{np.round(mape, 2)}%" if not np.isnan(mape) else "N/A"
    }


def plot_actual_vs_predicted(y_true, y_pred, dates=None, model_name="Model", title=None):
    """
    Generates an interactive Plotly visualization comparing actual vs predicted PM2.5 values.
    Includes both a time-series tracking plot and a parity 1:1 scatter plot.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Chronological Timeline: Actual vs Predicted",
            f"Parity Plot (Ideal 45° Line)"
        ),
        column_widths=[0.6, 0.4]
    )

    # Subplot 1: Chronological comparison
    x_axis = dates if dates is not None else np.arange(len(y_true))

    fig.add_trace(
        go.Scatter(
            x=x_axis, y=y_true,
            mode="lines", name="Actual PM2.5",
            line=dict(color="#166534", width=2),
            hovertemplate="Actual: %{y:.1f} µg/m³<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=y_pred,
            mode="lines", name=f"{model_name} Prediction",
            line=dict(color="#EAB308", width=2, dash="dash"),
            hovertemplate="Predicted: %{y:.1f} µg/m³<extra></extra>"
        ),
        row=1, col=1
    )

    # Subplot 2: Parity scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred,
            mode="markers", name="Observations",
            marker=dict(color="#22C55E", size=6, opacity=0.6),
            hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>"
        ),
        row=1, col=2
    )

    # 45-degree reference line
    min_val = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    max_val = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Ideal 1:1 Parity",
            line=dict(color="#14532D", dash="dot", width=1.5),
            hoverinfo="none"
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Date / Observation Index", row=1, col=1)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    fig.update_xaxes(title_text="Actual PM2.5 (µg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted PM2.5 (µg/m³)", row=1, col=2)

    plot_title = title or f"{model_name} — Evaluation on Held-Out Test Set"
    apply_chart_theme(fig, title=plot_title)
    return fig


def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    Residual diagnostic plots:
    1. Residuals vs Predicted (checks for heteroscedasticity and non-linear patterns)
    2. Residual Histogram (checks for normal distribution of errors)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Residuals vs Predicted Values",
            f"Residual Distribution (Errors)"
        ),
        column_widths=[0.55, 0.45]
    )

    # Subplot 1: Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode="markers", name="Residual",
            marker=dict(color="#166534", size=6, opacity=0.6),
            hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#EAB308", row=1, col=1)

    # Subplot 2: Residual histogram
    fig.add_trace(
        go.Histogram(
            x=residuals, nbinsx=40, name="Error Spread",
            marker_color="#22C55E", opacity=0.75
        ),
        row=1, col=2
    )

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    fig.add_vline(x=mean_res, line_dash="dash", line_color="#14532D",
                  annotation_text=f"Mean: {mean_res:.2f}", row=1, col=2)

    fig.update_xaxes(title_text="Predicted PM2.5 (µg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (Actual - Predicted)", row=1, col=1)
    fig.update_xaxes(title_text="Residual Error (µg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    apply_chart_theme(fig, title=f"Residual Diagnostic Analysis: {model_name}")
    fig.update_layout(showlegend=False)
    return fig, {"mean_residual": np.round(mean_res, 3), "std_residual": np.round(std_res, 3)}


def get_metric_explanations():
    """
    Returns concise academic definitions for project presentation and viva examination.
    """
    return {
        "MAE (Mean Absolute Error)": (
            "Measures the average absolute magnitude of the prediction errors. "
            "Expressed in the original unit (µg/m³). An MAE of 15.0 means the model "
            "misses the actual PM2.5 level by 15.0 µg/m³ on average."
        ),
        "MSE (Mean Squared Error)": (
            "The average of the squared prediction errors. By squaring the differences, "
            "it places heavier weight on large prediction errors."
        ),
        "RMSE (Root Mean Squared Error)": (
            "The square root of MSE. Like MAE, it is expressed in µg/m³, but it penalizes "
            "severe outliers much more heavily than MAE. When RMSE is significantly higher than MAE, "
            "it indicates substantial large-magnitude errors."
        ),
        "R² (Coefficient of Determination)": (
            "The proportion of the variance in PM2.5 that is explained by the independent variables. "
            "Ranges up to 1.0 (or 100%). A higher R² on the held-out test set indicates superior predictive capability."
        ),
        "MAPE (Mean Absolute Percentage Error)": (
            "The percentage difference between forecasts and actual values. "
            "Useful for time-series forecasting across different scales."
        )
    }

