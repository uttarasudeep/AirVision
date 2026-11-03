"""
Exploratory Data Analysis (EDA) Module for Air Quality System
Generates interactive visualizations and statistical metrics using Plotly and Pandas.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.chart_theme import apply_chart_theme


def plot_aqi_trend(df, city=None, target_col="AQI"):
    """
    Time-series line chart of AQI or PM2.5 over time for a selected city or overall average.
    """
    data = df.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", target_col])

    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]
        title = f"Daily {target_col} Trend for {city}"
    else:
        # Aggregate across all cities
        data = data.groupby("Date")[target_col].mean().reset_index()
        title = f"Daily Average {target_col} Across All Cities"

    fig = go.Figure()

    # Add trend line
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data[target_col],
        mode="lines",
        name=f"Daily {target_col}",
        line=dict(color="#166534", width=1.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>" + target_col + ":</b> %{y:.1f} µg/m³<extra></extra>"
    ))

    # Add 30-day rolling average for smooth trend visibility
    if len(data) >= 30:
        data["Rolling_30"] = data[target_col].rolling(window=30, min_periods=7).mean()
        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["Rolling_30"],
            mode="lines",
            name="30-Day Moving Average",
            line=dict(color="#EAB308", width=2.5, dash="dash"),
            hovertemplate="<b>30-Day Avg:</b> %{y:.1f} µg/m³<extra></extra>"
        ))

    # Add standard CPCB threshold reference line for PM2.5 (60 µg/m³ standard)
    if target_col == "PM2.5":
        fig.add_hline(y=60, line_dash="dot", line_color="#22C55E",
                      annotation_text="NAAQS Safe Limit (60 µg/m³)",
                      annotation_position="top right")

    apply_chart_theme(fig, title=title, x_title="Date", y_title=f"{target_col} (µg/m³)")
    return fig


def plot_pollutant_distribution(df, pollutant="PM2.5", city=None):
    """
    Histogram and KDE-style distribution plot for a given pollutant.
    """
    data = df.copy()
    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]

    series = pd.to_numeric(data[pollutant], errors="coerce").dropna()
    if series.empty:
        return go.Figure()

    mean_val = series.mean()
    median_val = series.median()

    fig = px.histogram(
        series,
        x=pollutant,
        nbins=60,
        marginal="box",
        labels={pollutant: f"{pollutant} Concentration (µg/m³)"},
        color_discrete_sequence=["#166534"],
        opacity=0.85
    )

    fig.add_vline(x=mean_val, line_dash="dash", line_color="#EAB308",
                  annotation_text=f"Mean: {mean_val:.1f}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="#22C55E",
                  annotation_text=f"Median: {median_val:.1f}")

    dist_title = f"Statistical Distribution of {pollutant} {f'in {city}' if city else 'Nationwide'}"
    apply_chart_theme(fig, title=dist_title, x_title=f"{pollutant} (µg/m³)", y_title="Frequency (Days)")
    return fig


def plot_pm10_distribution(df, city=None):
    """
    Dedicated distribution plot for PM10 particulate matter.
    """
    if "PM10" in df.columns:
        return plot_pollutant_distribution(df, pollutant="PM10", city=city)
    return go.Figure()


def plot_correlation_heatmap(df, city=None):
    """
    Correlation heatmap across numerical pollutants.
    """
    data = df.copy()
    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]

    numeric_cols = [
        col for col in ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "AQI"]
        if col in data.columns
    ]

    corr = data[numeric_cols].apply(pd.to_numeric, errors="coerce").corr()
    corr_rounded = np.round(corr, 2)

    fig = px.imshow(
        corr_rounded,
        text_auto=True,
        color_continuous_scale="YlGn_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    h_title = f"Pollutant Correlation Matrix {f'({city})' if city else '(All Cities)'}"
    apply_chart_theme(fig, title=h_title, x_title="Pollutants", y_title="Pollutants")
    return fig


def plot_monthly_pollution(df, pollutant="PM2.5", city=None):
    """
    Bar chart of monthly average pollution showing seasonal patterns.
    Highlights winter peaks (Nov-Jan) vs monsoon dips (Jul-Sep) in green & yellow.
    """
    data = df.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Month_Num"] = data["Date"].dt.month
        data["Month_Name"] = data["Date"].dt.strftime("%b")

    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_stats = data.groupby("Month_Name")[pollutant].agg(["mean", "median", "std"]).reindex(month_order).reset_index()

    fig = go.Figure()

    # Bar for mean: winter peak in Accent Yellow (#EAB308), Monsoon dip in Fresh Green (#22C55E), other in Forest Green (#166534)
    bar_colors = [
        "#EAB308" if m in ["Nov", "Dec", "Jan"] else "#22C55E" if m in ["Jul", "Aug", "Sep"] else "#166534"
        for m in monthly_stats["Month_Name"]
    ]

    fig.add_trace(go.Bar(
        x=monthly_stats["Month_Name"],
        y=monthly_stats["mean"],
        name="Mean Concentration",
        marker_color=bar_colors,
        error_y=dict(type="data", array=monthly_stats["std"] / 2, visible=True, color="#64748B"),
        hovertemplate="<b>%{x}:</b> Mean %{y:.1f} µg/m³<extra></extra>"
    ))

    m_title = f"Monthly Average {pollutant} Levels (Seasonal Variation) {f'in {city}' if city else ''}"
    apply_chart_theme(fig, title=m_title, x_title="Month", y_title=f"{pollutant} (µg/m³)")
    return fig


def plot_city_comparison(df, cities=None, pollutant="PM2.5", top_n=15):
    """
    Compare average pollution levels across cities using a refined green-yellow palette.
    """
    data = df.copy()
    if "City" not in data.columns or pollutant not in data.columns:
        return go.Figure()

    if cities and len(cities) > 0:
        data = data[data["City"].isin(cities)]

    city_means = (
        data.groupby("City")[pollutant]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        city_means,
        x=pollutant,
        y="City",
        orientation="h",
        labels={pollutant: f"Average {pollutant} (µg/m³)", "City": "City"},
        color=pollutant,
        color_continuous_scale="YlGn_r"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))
    c_title = f"Top {len(city_means)} Cities by Average {pollutant} Concentration"
    apply_chart_theme(fig, title=c_title, x_title=f"Average {pollutant} (µg/m³)", y_title="City")
    return fig


def plot_pollutant_boxplots(df, pollutants=None, city=None):
    """
    Boxplot visualization across major pollutants using green and yellow palette.
    """
    data = df.copy()
    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]

    if pollutants is None:
        pollutants = [p for p in ["PM2.5", "PM10", "NO2", "SO2", "O3"] if p in data.columns]

    melted = pd.melt(data[pollutants], var_name="Pollutant", value_name="Concentration").dropna()

    fig = px.box(
        melted,
        x="Pollutant",
        y="Concentration",
        color="Pollutant",
        points="outliers",
        color_discrete_sequence=["#166534", "#22C55E", "#EAB308", "#14532D", "#84CC16"],
        labels={"Concentration": "Concentration (µg/m³)"}
    )

    fig.update_layout(showlegend=False)
    b_title = f"Pollutant Spread and Outlier Boxplots {f'({city})' if city else ''}"
    apply_chart_theme(fig, title=b_title, x_title="Pollutant", y_title="Concentration (µg/m³)")
    return fig


def get_summary_statistics(df, city=None):
    """
    Generates a clean statistical summary table of numerical pollutants.
    """
    data = df.copy()
    if city and city != "All Cities" and "City" in data.columns:
        data = data[data["City"] == city]

    numeric_cols = [
        col for col in ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "AQI"]
        if col in data.columns
    ]

    desc = data[numeric_cols].describe().T
    desc["median"] = data[numeric_cols].median()
    desc["missing"] = data[numeric_cols].isnull().sum()
    desc["missing_%"] = np.round((desc["missing"] / len(data)) * 100, 2)

    rename_map = {
        "count": "Valid Records",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Min",
        "25%": "25% (Q1)",
        "median": "Median",
        "75%": "75% (Q3)",
        "max": "Max",
        "missing": "Missing Count",
        "missing_%": "Missing (%)"
    }

    cols_order = ["Valid Records", "Mean", "Std Dev", "Min", "25% (Q1)", "Median", "75% (Q3)", "Max", "Missing Count", "Missing (%)"]
    out = desc.rename(columns=rename_map)[cols_order].round(2)
    return out

