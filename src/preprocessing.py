"""
Data Preprocessing Module for Air Quality Prediction System
Handles dataset discovery, loading, cleaning, validation, missing value imputation,
outlier analysis, and data preparation for regression and time-series models.
"""

import os
import numpy as np
import pandas as pd


def get_dataset_path(filename="city_day.csv"):
    """
    Search for dataset in common locations: current directory, 'data/' subdirectory,
    or relative paths. Returns absolute path if found, else None.
    """
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join("d:\\AirQuality", filename),
        os.path.join("d:\\AirQuality", "data", filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def load_raw_data(filepath=None):
    """
    Load raw CSV data. If filepath is None, automatically searches for city_day.csv.
    """
    if filepath is None:
        filepath = get_dataset_path("city_day.csv")

    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(
            "Dataset 'city_day.csv' was not found in the workspace root or 'data/' folder. "
            "Please verify the file exists or upload it via the dashboard."
        )

    df = pd.read_csv(filepath)
    return df


def inspect_data(df):
    """
    Inspect raw dataset structure, types, missing values, duplicates, and unique cities.
    Returns a comprehensive audit dictionary.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    duplicates = int(df.duplicated().sum())

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [str(t) for t in df.dtypes],
        "Missing_Values": missing_counts.values,
        "Missing_Percentage": np.round(missing_pct.values, 2)
    })

    cities = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []

    date_range = ("N/A", "N/A")
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if not dates.empty:
            date_range = (dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"))

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicates": duplicates,
        "missing_summary": missing_df,
        "cities": cities,
        "num_cities": len(cities),
        "date_range": date_range,
    }


def analyze_outliers(df, columns=None):
    """
    Calculate Interquartile Range (IQR) outlier bounds for numerical columns.
    Important note: In air quality analytics, extreme values during winter smog or
    festival periods (e.g. Diwali) represent genuine severe pollution events rather
    than sensor malfunctions. Therefore, outliers are detected for reporting rather
    than blind deletion.
    """
    if columns is None:
        columns = [c for c in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"] if c in df.columns]

    outlier_records = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_records.append({
            "Pollutant": col,
            "Count": len(series),
            "Median": np.round(series.median(), 2),
            "Q1 (25%)": np.round(q1, 2),
            "Q3 (75%)": np.round(q3, 2),
            "IQR": np.round(iqr, 2),
            "Upper_Bound": np.round(upper_bound, 2),
            "Outlier_Count": outlier_count,
            "Outlier_Pct": np.round((outlier_count / len(series)) * 100, 2),
            "Max_Observed": np.round(series.max(), 2),
        })

    return pd.DataFrame(outlier_records)


def clean_and_preprocess(df, selected_city=None, target_col="PM2.5"):
    """
    Clean dataset:
    - Parses Date to datetime.
    - Sorts chronologically.
    - Removes duplicates.
    - Filters by city (if selected).
    - Drops rows where target_col (PM2.5) is missing (cannot train or evaluate without target).
    - Imputes predictor pollutants using forward fill (temporal continuity) + city/global median.
    - Returns cleaned DataFrame and quality dictionary.
    """
    initial_rows = len(df)
    data = df.copy()

    # 1. Parse Date and sort chronologically
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])

    sort_cols = ["City", "Date"] if "City" in data.columns else ["Date"]
    data = data.sort_values(by=sort_cols).reset_index(drop=True)

    # 2. Remove duplicate rows
    data = data.drop_duplicates()

    # 3. Filter by city if specified
    if selected_city and selected_city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == selected_city].copy().reset_index(drop=True)

    # 4. Standard pollutant columns
    numeric_pollutants = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
        "Benzene", "Toluene", "Xylene", "AQI"
    ]
    present_numeric = [c for c in numeric_pollutants if c in data.columns]
    for col in present_numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 5. Drop rows with missing target variable (PM2.5)
    if target_col in data.columns:
        data = data.dropna(subset=[target_col]).copy()

    # 6. Predictor imputation (Forward fill then median)
    predictor_cols = [c for c in present_numeric if c != target_col and c != "AQI"]
    for col in predictor_cols:
        if "City" in data.columns:
            data[col] = data.groupby("City")[col].transform(lambda s: s.ffill().bfill())
        else:
            data[col] = data[col].ffill().bfill()
        # Fallback to column median if entire series had nulls
        median_val = data[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        data[col] = data[col].fillna(median_val)

    cleaned_rows = len(data)
    rows_dropped = initial_rows - cleaned_rows

    quality_summary = {
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "rows_dropped": rows_dropped,
        "retention_rate": np.round((cleaned_rows / max(1, initial_rows)) * 100, 2),
        "selected_city": selected_city or "All Cities",
        "date_min": data["Date"].min().strftime("%Y-%m-%d") if "Date" in data.columns and not data.empty else "N/A",
        "date_max": data["Date"].max().strftime("%Y-%m-%d") if "Date" in data.columns and not data.empty else "N/A",
    }

    return data, quality_summary


def prepare_regression_dataset(df, target_col="PM2.5", train_ratio=0.8):
    """
    Prepares chronological features for PM2.5 regression estimation.
    Features: Available chemical pollutants (NO2, CO, SO2, O3, etc.) + calendar features.
    CRITICAL: Excludes AQI and AQI_Bucket to prevent data leakage (AQI is mathematically
    derived from PM2.5 sub-index).
    Uses chronological train/test split (no random shuffling) to respect temporal sequence.
    """
    df_feat = df.copy()

    # Create calendar temporal features
    if "Date" in df_feat.columns:
        df_feat["Month"] = df_feat["Date"].dt.month
        df_feat["Day"] = df_feat["Date"].dt.day
        df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
        df_feat["IsWeekend"] = df_feat["DayOfWeek"].apply(lambda d: 1 if d in [5, 6] else 0)

    # Candidate pollutant predictors (strictly excluding AQI and AQI_Bucket)
    potential_pollutants = ["PM10", "NO2", "CO", "SO2", "O3", "NOx", "NH3", "NO", "Benzene", "Toluene"]
    pollutant_features = [col for col in potential_pollutants if col in df_feat.columns]

    calendar_features = [c for c in ["Month", "Day", "DayOfWeek", "IsWeekend"] if c in df_feat.columns]
    feature_names = pollutant_features + calendar_features

    # Ensure chronological order
    if "Date" in df_feat.columns:
        df_feat = df_feat.sort_values(by="Date").reset_index(drop=True)

    # Drop any remaining NaN in target or selected features
    model_df = df_feat.dropna(subset=[target_col] + feature_names).copy()

    X = model_df[feature_names].values
    y = model_df[target_col].values
    dates = model_df["Date"].values if "Date" in model_df.columns else None

    # Chronological Split (Train: older records, Test: recent records)
    n = len(model_df)
    train_size = int(n * train_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    dates_train = dates[:train_size] if dates is not None else None
    dates_test = dates[train_size:] if dates is not None else None

    split_info = {
        "n_samples": n,
        "n_train": train_size,
        "n_test": n - train_size,
        "train_ratio": train_ratio,
        "train_dates": (str(dates_train[0])[:10], str(dates_train[-1])[:10]) if dates_train is not None and len(dates_train) > 0 else None,
        "test_dates": (str(dates_test[0])[:10], str(dates_test[-1])[:10]) if dates_test is not None and len(dates_test) > 0 else None,
    }

    return X_train, X_test, y_train, y_test, feature_names, split_info, dates_test


def prepare_time_series_dataset(df, city, target_col="PM2.5"):
    """
    Extracts a continuous, daily time-series of PM2.5 for a specific city.
    Ensures a continuous daily frequency (freq='D') by reindexing across the city's
    date range and applying linear interpolation + backfill/forward-fill to bridge
    occasional sensor downtime days.
    """
    city_data = df[df["City"] == city].copy() if "City" in df.columns else df.copy()

    if "Date" in city_data.columns:
        city_data["Date"] = pd.to_datetime(city_data["Date"], errors="coerce")
        city_data = city_data.dropna(subset=["Date"])
        city_data = city_data.sort_values(by="Date").drop_duplicates(subset=["Date"])
        city_data = city_data.set_index("Date")

    if target_col not in city_data.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset for city {city}.")

    # Resample / reindex to complete daily calendar
    ts = pd.to_numeric(city_data[target_col], errors="coerce")
    if ts.empty or len(ts.dropna()) < 30:
        return None

    full_date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="D")
    ts_daily = ts.reindex(full_date_range)

    # Impute missing dates smoothly with time-based linear interpolation
    ts_imputed = ts_daily.interpolate(method="time").ffill().bfill()
    ts_imputed.name = target_col

    return ts_imputed

