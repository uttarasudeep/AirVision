"""
Machine Learning Prediction Module for Air Quality System
Implements Multiple Linear Regression (Scikit-Learn & Statsmodels OLS)
and Decision Tree Regression with depth control and feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm

from src.evaluation import calculate_regression_metrics


def train_linear_regression(X_train, y_train, X_test, y_test, feature_names):
    """
    Trains Multiple Linear Regression using Scikit-Learn and Statsmodels OLS.
    Returns:
    - sklearn_model
    - metrics_train, metrics_test
    - y_pred_train, y_pred_test
    - statsmodels_summary (coefficients, std err, t-stat, p-value, 95% CI)
    - multicollinearity_info (condition number)
    """
    # 1. Scikit-learn Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    metrics_train = calculate_regression_metrics(y_train, y_pred_train, model_name="Multiple Linear Regression (Train)")
    metrics_test = calculate_regression_metrics(y_test, y_pred_test, model_name="Multiple Linear Regression (Test)")

    # 2. Statsmodels OLS for rigorous statistical inference (p-values, CIs)
    X_train_sm = sm.add_constant(X_train)
    sm_feature_names = ["Intercept"] + list(feature_names)

    ols_model = sm.OLS(y_train, X_train_sm).fit()

    conf = ols_model.conf_int()
    coef_df = pd.DataFrame({
        "Feature": sm_feature_names,
        "Coefficient": np.round(ols_model.params, 4),
        "Std_Error": np.round(ols_model.bse, 4),
        "t_Statistic": np.round(ols_model.tvalues, 3),
        "p_Value": np.round(ols_model.pvalues, 4),
        "CI_Lower_95%": np.round(conf[:, 0], 4),
        "CI_Upper_95%": np.round(conf[:, 1], 4),
    })

    # Significance indicator
    coef_df["Significance"] = coef_df["p_Value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "Not Significant"))
    )

    statistical_diagnostics = {
        "aic": float(ols_model.aic),
        "bic": float(ols_model.bic),
        "f_pvalue": float(ols_model.f_pvalue),
        "condition_number": float(ols_model.condition_number),
        "is_multicollinear": bool(ols_model.condition_number > 30),
        "coefficients_table": coef_df,
        "summary_text": str(ols_model.summary())
    }

    return {
        "model": lr,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "stats": statistical_diagnostics,
        "feature_names": feature_names
    }


def train_decision_tree(X_train, y_train, X_test, y_test, feature_names, max_depth=6):
    """
    Trains Decision Tree Regressor with controlled depth to prevent overfitting.
    Returns:
    - tree_model
    - metrics_train, metrics_test
    - y_pred_train, y_pred_test
    - feature_importance_df
    """
    dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    metrics_train = calculate_regression_metrics(y_train, y_pred_train, model_name=f"Decision Tree (Depth={max_depth}, Train)")
    metrics_test = calculate_regression_metrics(y_test, y_pred_test, model_name=f"Decision Tree (Depth={max_depth}, Test)")

    # Extract Feature Importances
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.round(dt.feature_importances_, 4),
        "Importance_%": np.round(dt.feature_importances_ * 100, 2)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "model": dt,
        "max_depth": max_depth,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "feature_importance": importance_df,
        "feature_names": feature_names
    }


def predict_single_pm25(model, input_dict, feature_names):
    """
    Validates input features and generates a single PM2.5 prediction.
    Enforces non-negative values and realistic pollutant bounds.
    """
    row = []
    for f in feature_names:
        val = input_dict.get(f, 0.0)
        # Validation: clamp negative pollutants to 0
        if f not in ["Month", "Day", "DayOfWeek", "IsWeekend"]:
            val = max(0.0, float(val))
        row.append(val)

    X_input = np.array([row])
    prediction = float(model.predict(X_input)[0])

    # Air pollution concentration cannot physically be negative
    prediction = max(0.0, np.round(prediction, 2))

    # Determine qualitative AQI category for PM2.5 according to CPCB criteria
    category = categorize_pm25(prediction)

    return prediction, category


def categorize_pm25(pm25_val):
    """
    Categorizes PM2.5 concentration according to Central Pollution Control Board (CPCB) standards:
    - Good: 0 - 30 µg/m³
    - Satisfactory: 31 - 60 µg/m³
    - Moderate: 61 - 90 µg/m³
    - Poor: 91 - 120 µg/m³
    - Very Poor: 121 - 250 µg/m³
    - Severe: > 250 µg/m³
    """
    if pm25_val <= 30:
        return "Good", "#2ecc71", "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif pm25_val <= 60:
        return "Satisfactory", "#27ae60", "Air quality is acceptable; minor breathing discomfort to sensitive people."
    elif pm25_val <= 90:
        return "Moderate", "#f39c12", "Breathing discomfort to people with lungs, asthma, and heart diseases."
    elif pm25_val <= 120:
        return "Poor", "#e67e22", "Breathing discomfort to most people on prolonged exposure."
    elif pm25_val <= 250:
        return "Very Poor", "#e74c3c", "Respiratory illness to the people on prolonged exposure."
    else:
        return "Severe", "#8e44ad", "Affects healthy people and seriously impacts those with existing diseases."
