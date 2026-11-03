"""
AirQuality — Environmental Analytics & Forecasting Platform
A modern, climate-tech inspired Air Quality Intelligence Platform.
Engineered with strict high-contrast typography, clear hierarchy,
robust green & yellow environmental styling, and zero white-on-white text issues.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import (
    load_raw_data,
    inspect_data,
    analyze_outliers,
    clean_and_preprocess,
    prepare_regression_dataset,
    prepare_time_series_dataset
)
from src.eda import (
    plot_aqi_trend,
    plot_pollutant_distribution,
    plot_pm10_distribution,
    plot_correlation_heatmap,
    plot_monthly_pollution,
    plot_city_comparison,
    plot_pollutant_boxplots,
    get_summary_statistics
)
from src.prediction import (
    train_linear_regression,
    train_decision_tree,
    predict_single_pm25,
    categorize_pm25
)
from src.forecasting import (
    run_all_forecasting_models,
    plot_forecast_results
)
from src.long_term_forecasting import (
    prepare_monthly_series,
    run_all_long_term_models,
    plot_long_term_forecast,
    categorize_target
)
from src.evaluation import (
    plot_actual_vs_predicted,
    plot_residuals
)
from src.chart_theme import apply_chart_theme

# ---------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AirQuality — Environmental Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Bulletproof High-Contrast CSS (Strict Green & Yellow Environmental Theme)
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAF5 !important;
        color: #1F2937 !important;
    }

    /* Target all markdown containers to prevent white text leakage */
    .stMarkdown,
    [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] {
        color: #1F2937 !important;
    }

    /* Main Headings */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #14532D !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    /* Paragraphs and general text */
    p, span, li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #1F2937 !important;
    }

    /* Form and Widget Labels */
    label,
    .stSelectbox label,
    .stRadio label,
    .stSlider label,
    .stNumberInput label {
        color: #14532D !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* =================================================================
       SIDEBAR STYLING
       ================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
        padding-top: 1rem !important;
    }

    /* Brand Header in Sidebar */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.5rem 1.25rem 0.5rem;
        border-bottom: 2px solid #F0FDF4;
        margin-bottom: 1.25rem;
    }
    .sidebar-brand-icon {
        font-size: 1.6rem;
        background: #DCFCE7;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        border: 1px solid #BBF7D0;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #14532D !important;
        line-height: 1.1;
    }
    .sidebar-brand-subtitle {
        font-size: 0.72rem;
        color: #166534 !important;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Sidebar Radio Navigation items */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: #F0FDF4 !important;
        border-color: #DCFCE7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #DCFCE7 !important;
        border-color: #86EFAC !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label div {
        color: #14532D !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Sidebar Footer Info */
    .sidebar-footer {
        font-size: 0.82rem;
        color: #64748B !important;
        padding: 0.75rem 0.5rem;
        line-height: 1.5;
        border-top: 1px solid #F3F4F6;
        margin-top: 1.5rem;
    }
    .sidebar-footer strong {
        color: #166534 !important;
    }

    /* =================================================================
       PAGE HEADER / HERO SECTION
       ================================================================= */
    .app-header {
        background-color: #DCFCE7;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 22px;
    }
    .app-header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #14532D !important;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .app-header-subtitle {
        font-size: 0.98rem;
        color: #166534 !important;
        margin: 0;
        font-weight: 500;
    }

    /* =================================================================
       KPI CARDS
       ================================================================= */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 22px;
    }
    @media (max-width: 900px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-top: 3px solid #166534;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(20, 83, 45, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(20, 83, 45, 0.08);
        border-top-color: #EAB308;
    }
    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #14532D !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-icon-pill {
        font-size: 0.95rem;
        background: #F0FDF4;
        color: #166534 !important;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #DCFCE7;
    }
    .kpi-number {
        font-size: 1.9rem;
        font-weight: 800;
        color: #14532D !important;
        line-height: 1.15;
    }
    .kpi-unit-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B !important;
        margin-left: 3px;
    }
    .kpi-footnote {
        font-size: 0.76rem;
        color: #166534 !important;
        font-weight: 500;
        margin-top: 5px;
    }

    /* =================================================================
       NOTICE & ADVISORY BOXES
       ================================================================= */
    .notice-card {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #EAB308;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #92400E !important;
        margin: 16px 0;
        line-height: 1.5;
    }
    .notice-card strong {
        color: #78350F !important;
    }

    .info-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #166534;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #14532D !important;
        margin: 16px 0;
        line-height: 1.5;
    }

    /* =================================================================
       PREDICTION RESULT CARD
       ================================================================= */
    .prediction-box {
        background: #FEFCE8;
        border: 2px solid #FDE047;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(234, 179, 8, 0.12);
        margin-bottom: 16px;
    }
    .prediction-heading {
        font-size: 0.84rem;
        font-weight: 700;
        color: #854D0E !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prediction-metric {
        font-size: 3rem;
        font-weight: 800;
        color: #713F12 !important;
        line-height: 1.1;
        margin: 8px 0;
    }
    .prediction-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* =================================================================
       INPUTS & BUTTONS
       ================================================================= */
    .stButton > button,
    button[kind="primary"] {
        background-color: #166534 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #14532D !important;
    }
    .stButton > button * {
        color: #FFFFFF !important;
    }

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #D1D5DB !important;
    }
    div[data-baseweb="select"] * {
        color: #1F2937 !important;
    }
    input {
        color: #1F2937 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 9px 16px;
        font-weight: 600;
        color: #4B5563 !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #4B5563 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #14532D !important;
        border-bottom: 3px solid #166534 !important;
        background-color: #DCFCE7 !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #14532D !important;
        font-weight: 700 !important;
    }

    /* Metric Cards within Streamlit */
    [data-testid="stMetricValue"] * {
        color: #14532D !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #166534 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Cached Data Loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads primary city_day.csv dataset."""
    return load_raw_data()


@st.cache_data(show_spinner=False)
def get_cleaned_city_data(df, city):
    """Filters and preprocesses city data."""
    return clean_and_preprocess(df, selected_city=city)


# ---------------------------------------------------------------------
# Load Data & Handle Missing Gracefully
# ---------------------------------------------------------------------
try:
    df_raw = load_dataset()
    data_audit = inspect_data(df_raw)
    city_list = data_audit["cities"]
    all_cities_option = ["All Cities"] + city_list
    global_avg_pm25 = round(float(df_raw["PM2.5"].mean()), 1)
    global_avg_aqi = round(float(df_raw["AQI"].mean()), 1)
    latest_dataset_date = data_audit["date_range"][1]
    earliest_dataset_date = data_audit["date_range"][0]
except Exception as e:
    st.error(f"❌ Error loading dataset `city_day.csv`: {e}")
    st.info("Please verify `city_day.csv` is present in the workspace root or `data/` folder.")
    st.stop()


# ---------------------------------------------------------------------
# Clean Sidebar Navigation (Exactly 5 Pages, Dark Green Text)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🌿</div>
        <div>
            <div class="sidebar-brand-title">AirQuality</div>
            <div class="sidebar-brand-subtitle">Environmental Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Home & Overview",
            "Exploratory Data Analysis",
            "PM2.5 Prediction",
            "Time-Series Forecasting",
            "Long-Term Forecast"
        ],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <div class="sidebar-footer">
        <strong>Data Source:</strong> CPCB Daily Records<br>
        <strong>Observed:</strong> {earliest_dataset_date} to {latest_dataset_date}<br>
        <strong>Coverage:</strong> 26 Indian Cities
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE 1: HOME & OVERVIEW
# =====================================================================
if page == "Home & Overview":
    # Header Section
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">Air Quality Intelligence</div>
        <div class="app-header-subtitle">Analyze pollution patterns. Predict PM2.5 levels. Explore future air quality trends.</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Cards (Responsive grid, clean white cards with green headers)
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Total Observations</span>
                <span class="kpi-icon-pill">📊 Records</span>
            </div>
            <div class="kpi-number">{data_audit['total_rows']:,}</div>
            <div class="kpi-footnote">Daily city observations</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Cities Covered</span>
                <span class="kpi-icon-pill">🏙️ Stations</span>
            </div>
            <div class="kpi-number">{data_audit['num_cities']}</div>
            <div class="kpi-footnote">Metropolitan centers across India</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Average PM2.5</span>
                <span class="kpi-icon-pill">💨 PM2.5</span>
            </div>
            <div class="kpi-number">{global_avg_pm25}<span class="kpi-unit-label">µg/m³</span></div>
            <div class="kpi-footnote">Safe limit: 60 µg/m³ (NAAQS)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Average AQI</span>
                <span class="kpi-icon-pill">📈 Index</span>
            </div>
            <div class="kpi-number">{global_avg_aqi}</div>
            <div class="kpi-footnote">CPCB Moderate tier (101–200)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MAIN CHARTS: Two-Column Layout (Historical PM2.5 Trend | PM2.5 Distribution)
    st.markdown("### 📈 Main Environmental Trends")
    col_chart1, col_chart2 = st.columns([1.5, 1.0], gap="medium")

    df_clean_all, _ = get_cleaned_city_data(df_raw, "All Cities")

    with col_chart1:
        fig_trend = plot_aqi_trend(df_clean_all, city="All Cities", target_col="PM2.5")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        fig_dist = plot_pollutant_distribution(df_clean_all, pollutant="PM2.5", city=None)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ADDITIONAL ANALYSIS: Monthly Pollution Trends & Pollution Summary
    st.markdown("### 🗓️ Seasonal Dynamics & Empirical Summary")
    col_add1, col_add2 = st.columns([1.3, 1.0], gap="medium")

    with col_add1:
        fig_month = plot_monthly_pollution(df_clean_all, pollutant="PM2.5", city=None)
        st.plotly_chart(fig_month, use_container_width=True)

    with col_add2:
        st.markdown(f"""
        <div class="notice-card">
            📅 <strong>Historical Observation Notice:</strong> The dataset active timeframe concludes on <strong>{latest_dataset_date}</strong>. Historical observations are preserved to enable rigorous chronological training without lookahead bias.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <strong>Key Environmental Insights:</strong>
            <ul style="margin-top: 6px; padding-left: 18px; margin-bottom: 0;">
                <li><strong>Winter Surge:</strong> PM2.5 values peak sharply during November to January due to thermal boundary inversions and post-monsoon biomass burning.</li>
                <li><strong>Monsoon Washing:</strong> July through September records the cleanest air due to precipitation scavenging.</li>
                <li><strong>Leading Stations:</strong> Delhi (2,007 days), Lucknow (1,907 days), and Chennai (1,892 days) maintain the most complete continuous daily records.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# =====================================================================
elif page == "Exploratory Data Analysis":
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">Exploratory Data Analysis</div>
        <div class="app-header-subtitle">Interactive inspection of pollutant distributions, temporal seasonality, and city-level rankings.</div>
    </div>
    """, unsafe_allow_html=True)

    # City Selector
    selected_city = st.selectbox(
        "Select City for Detailed Analysis:",
        all_cities_option,
        index=10 if len(all_cities_option) > 10 else 0
    )

    df_city_clean, q_summary = get_cleaned_city_data(df_raw, selected_city)

    # Compact KPI strip
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Clean Records", f"{q_summary['cleaned_rows']:,}")
    with s2:
        st.metric("Target Available", f"{q_summary['retention_rate']}%")
    with s3:
        st.metric("First Record", q_summary['date_min'])
    with s4:
        st.metric("Latest Record", q_summary['date_max'])

    eda_tabs = st.tabs([
        "📈 Time Trends",
        "📊 Distributions",
        "🔥 Pollutant Correlations",
        "🗓️ Seasonal Patterns",
        "🏙️ City Rankings & Dispersion",
        "📋 Data Quality & Summary"
    ])

    with eda_tabs[0]:
        st.markdown("#### Daily Pollutant Time Trend")
        param = st.radio("Select Pollutant Parameter:", ["PM2.5", "AQI", "PM10", "NO2", "CO", "SO2"], horizontal=True)
        if param in df_city_clean.columns:
            fig_t = plot_aqi_trend(df_city_clean, city=selected_city, target_col=param)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info(f"Parameter {param} not recorded for {selected_city}.")

    with eda_tabs[1]:
        st.markdown("#### Particulate Matter Distributions (PM2.5 vs PM10)")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_p1 = plot_pollutant_distribution(df_city_clean, pollutant="PM2.5", city=selected_city)
            st.plotly_chart(fig_p1, use_container_width=True)
        with c_p2:
            fig_p2 = plot_pm10_distribution(df_city_clean, city=selected_city)
            st.plotly_chart(fig_p2, use_container_width=True)

    with eda_tabs[2]:
        st.markdown("#### Inter-Pollutant Correlation Matrix")
        st.caption("Quantifies linear co-movement between chemical species. Notice strong associations between combustion byproducts.")
        fig_c = plot_correlation_heatmap(df_city_clean, city=selected_city)
        st.plotly_chart(fig_c, use_container_width=True)

    with eda_tabs[3]:
        st.markdown("#### Monthly Seasonality (Winter Inversion vs Monsoon Washout)")
        fig_m = plot_monthly_pollution(df_city_clean, pollutant="PM2.5", city=selected_city)
        st.plotly_chart(fig_m, use_container_width=True)

    with eda_tabs[4]:
        st.markdown("#### City Rankings & Dispersion")
        c_rk1, c_rk2 = st.columns(2)
        with c_rk1:
            fig_rk = plot_city_comparison(df_raw, pollutant="PM2.5", top_n=12)
            st.plotly_chart(fig_rk, use_container_width=True)
        with c_rk2:
            fig_bx = plot_pollutant_boxplots(df_city_clean, city=selected_city)
            st.plotly_chart(fig_bx, use_container_width=True)

    with eda_tabs[5]:
        st.markdown("#### Statistical Summary Table")
        st_summary = get_summary_statistics(df_city_clean, city=selected_city)
        st.dataframe(st_summary, use_container_width=True)

        st.markdown("#### Interquartile Range (IQR) Outlier Audit")
        outliers_df = analyze_outliers(df_city_clean)
        st.dataframe(outliers_df, use_container_width=True)
        st.caption("💡 Outliers represent genuine atmospheric winter episodes and are retained to avoid biasing predictive models.")


# =====================================================================
# PAGE 3: PM2.5 PREDICTION (MACHINE LEARNING)
# =====================================================================
elif page == "PM2.5 Prediction":
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">PM2.5 Prediction</div>
        <div class="app-header-subtitle">Estimate fine particulate matter concentration using machine learning regression models.</div>
    </div>
    """, unsafe_allow_html=True)

    pred_city = st.selectbox(
        "Select City for Training & Prediction:",
        [c for c in city_list if c != "All Cities"],
        index=9 if len(city_list) > 9 else 0
    )

    df_pred_clean, _ = get_cleaned_city_data(df_raw, pred_city)

    if len(df_pred_clean) < 100:
        st.warning(f"Selected city has {len(df_pred_clean)} records, which is insufficient for reliable training. Please choose a city with more observations (e.g. Delhi, Bengaluru, Lucknow).")
        st.stop()

    X_train, X_test, y_train, y_test, feature_names, split_info, dates_test = prepare_regression_dataset(df_pred_clean)

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        st.markdown("#### Model Selection & Features")
        model_choice = st.selectbox(
            "Select Prediction Model:",
            ["Multiple Linear Regression", "Decision Tree Regressor"]
        )

        tree_depth = 6
        if model_choice == "Decision Tree Regressor":
            tree_depth = st.slider("Tree Maximum Depth (Overfitting Control):", min_value=3, max_value=10, value=6)

        st.caption("🛡️ **Methodological Safeguard:** `AQI` and `AQI_Bucket` are strictly excluded from inputs to eliminate target leakage. Models use chronological (80/20) train/test partitions.")

        # Train model
        if model_choice == "Multiple Linear Regression":
            model_res = train_linear_regression(X_train, y_train, X_test, y_test, feature_names)
        else:
            model_res = train_decision_tree(X_train, y_train, X_test, y_test, feature_names, max_depth=tree_depth)

        st.markdown("#### Enter Input Pollutant Values:")
        with st.form("ml_prediction_form"):
            input_values = {}
            f_cols = st.columns(2)
            for i, feat in enumerate(feature_names):
                target_c = f_cols[i % 2]
                with target_c:
                    feat_idx = feature_names.index(feat)
                    default_v = float(np.median(X_train[:, feat_idx]))
                    if feat in ["Month", "Day", "DayOfWeek", "IsWeekend"]:
                        input_values[feat] = st.number_input(f"{feat}", value=int(default_v), step=1)
                    else:
                        input_values[feat] = st.number_input(f"{feat} (µg/m³)", value=round(default_v, 1), min_value=0.0, step=1.0)

            submitted = st.form_submit_button("Predict PM2.5 Concentration")

    with col_right:
        st.markdown("#### Prediction Result & Model Evaluation")

        # Initial prediction based on current inputs
        pred_val, (cat_name, cat_color, cat_desc) = predict_single_pm25(model_res["model"], input_values, feature_names)

        # Soft yellow prediction card with high contrast text
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-heading">Estimated PM2.5 Concentration</div>
            <div class="prediction-metric">{pred_val} <span style="font-size:1.4rem; font-weight:600;">µg/m³</span></div>
            <div class="prediction-badge" style="background:{cat_color};">
                CPCB Category: {cat_name}
            </div>
            <div style="font-size:0.85rem; color:#78350F; margin-top:10px; font-weight:500;">{cat_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Test set performance metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Test MAE", f"{model_res['metrics_test']['MAE']} µg/m³")
        with m2:
            st.metric("Test RMSE", f"{model_res['metrics_test']['RMSE']} µg/m³")
        with m3:
            st.metric("Test R² Score", f"{model_res['metrics_test']['R2_Score']}")

        # Actual vs Predicted chart on held-out test data
        fig_avp = plot_actual_vs_predicted(y_test, model_res["y_pred_test"], dates=dates_test, model_name=model_choice)
        st.plotly_chart(fig_avp, use_container_width=True)


# =====================================================================
# PAGE 4: TIME-SERIES FORECASTING (SHORT-TERM)
# =====================================================================
elif page == "Time-Series Forecasting":
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">Air Quality Forecast</div>
        <div class="app-header-subtitle">Explore future PM2.5 concentrations based on historical pollution patterns.</div>
    </div>
    """, unsafe_allow_html=True)

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    with fc_col1:
        st_city = st.selectbox(
            "Select City:",
            [c for c in city_list if c != "All Cities"],
            index=9 if len(city_list) > 9 else 0
        )
    with fc_col2:
        st_horizon = st.selectbox("Forecast Horizon:", [7, 14, 30], index=1, format_func=lambda d: f"{d} Days Ahead")
    with fc_col3:
        st_model = st.selectbox("Forecasting Model:", ["ARIMA", "Moving Average", "Exponential Smoothing"], index=0)

    ts_daily = prepare_time_series_dataset(df_raw, city=st_city, target_col="PM2.5")

    if ts_daily is None or len(ts_daily) < 60:
        st.warning(f"Insufficient historical observations for {st_city} to generate short-term forecasts. Please select another city.")
        st.stop()

    with st.spinner("Calculating time-series forecast..."):
        try:
            fc_dict = run_all_forecasting_models(ts_daily, test_days=30, horizon=st_horizon)
        except Exception as e:
            st.error(f"Error computing forecast: {e}")
            st.stop()

    key_map = {
        "ARIMA": "arima",
        "Moving Average": "moving_average",
        "Exponential Smoothing": "exponential_smoothing"
    }
    cur_res = fc_dict[key_map[st_model]]

    st.markdown(f"""
    <div class="notice-card">
        📌 <strong>Historical Observation Notice:</strong> Forecasts project from <strong>{ts_daily.index.max().strftime('%Y-%m-%d')}</strong> forward. These are statistical estimates based on historical sensor observations and do not represent live real-time conditions.
    </div>
    """, unsafe_allow_html=True)

    # Metrics on 30-day held-out test horizon
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Test MAE", f"{cur_res['metrics']['MAE']} µg/m³")
    with k2:
        st.metric("Test RMSE", f"{cur_res['metrics']['RMSE']} µg/m³")
    with k3:
        st.metric("Test MAPE", f"{cur_res['metrics']['MAPE']}")
    with k4:
        st.metric("Horizon", f"{st_horizon} Days")

    # Forecast Chart (Green historical, Yellow forecast)
    fig_st = plot_forecast_results(fc_dict, selected_model_key=key_map[st_model], city_name=st_city)
    st.plotly_chart(fig_st, use_container_width=True)

    # Forecast Table
    st.markdown(f"#### {st_horizon}-Day Forecast Projections ({st_city})")
    fc_table = cur_res["forecast_df"].copy()
    fc_table["Date"] = fc_table["Date"].dt.strftime("%Y-%m-%d")
    fc_table["Unit"] = "µg/m³"
    fc_table["CPCB_Category"] = fc_table["Forecast_PM2.5"].apply(lambda v: categorize_pm25(v)[0])

    cols_show = ["Date", "Forecast_PM2.5", "Unit"]
    if "Lower_CI_95%" in fc_table.columns:
        cols_show += ["Lower_CI_95%", "Upper_CI_95%"]
    cols_show.append("CPCB_Category")

    st.dataframe(fc_table[cols_show], use_container_width=True)


# =====================================================================
# PAGE 5: LONG-TERM FORECAST
# =====================================================================
elif page == "Long-Term Forecast":
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">Long-Term Air Quality Forecast</div>
        <div class="app-header-subtitle">Forecast multi-month pollution trends and Air Quality Index (AQI) over the next 3, 6, or 12 months.</div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    lt_c1, lt_c2, lt_c3, lt_c4 = st.columns(4)
    with lt_c1:
        lt_city = st.selectbox(
            "Select City:",
            [c for c in city_list if c != "All Cities"],
            index=9 if len(city_list) > 9 else 0
        )
    with lt_c2:
        lt_target = st.selectbox("Forecast Target:", ["PM2.5", "AQI"])
    with lt_c3:
        lt_horizon_months = st.selectbox("Forecast Horizon:", [3, 6, 12], index=1, format_func=lambda m: f"{m} Months Ahead")
    with lt_c4:
        lt_model_type = st.selectbox(
            "Forecasting Model:",
            ["Seasonal ARIMA (SARIMA)", "Holt-Winters Exponential Smoothing", "Seasonal Naive (Baseline)"]
        )

    # Prepare continuous monthly series
    monthly_series = prepare_monthly_series(df_raw, city=lt_city, target_col=lt_target)

    if monthly_series is None or len(monthly_series) < 24:
        st.warning(f"Insufficient historical monthly observations available for {lt_city} to support long-term seasonal forecasting. Please choose a city with longer records (e.g. Delhi, Bengaluru, Lucknow).")
        st.stop()

    with st.spinner("Computing monthly long-term seasonal forecast..."):
        try:
            lt_results = run_all_long_term_models(
                monthly_series,
                test_months=min(6, len(monthly_series) // 4),
                horizon_months=lt_horizon_months,
                target_col=lt_target
            )
        except Exception as e:
            st.error(f"Error training long-term models: {e}")
            st.stop()

    lt_key_map = {
        "Seasonal ARIMA (SARIMA)": "sarima",
        "Holt-Winters Exponential Smoothing": "holt_winters",
        "Seasonal Naive (Baseline)": "seasonal_naive"
    }
    cur_lt_key = lt_key_map[lt_model_type]
    cur_lt_res = lt_results[cur_lt_key]

    # Prominent Historical Notice
    st.markdown(f"""
    <div class="notice-card">
        ⚠️ <strong>Important Historical Notice:</strong> Forecasts are based on historical observations ending in <strong>{lt_results['last_observed_date']}</strong>. These estimates do not represent current air quality measurements. Long-term forecasts are statistical projections and become more uncertain further into the future.
    </div>
    """, unsafe_allow_html=True)

    # Metrics on backtested period
    unit_label = "µg/m³" if lt_target == "PM2.5" else "AQI"
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Backtest MAE", f"{cur_lt_res['metrics']['MAE']} {unit_label}")
    with m_c2:
        st.metric("Backtest RMSE", f"{cur_lt_res['metrics']['RMSE']} {unit_label}")
    with m_c3:
        st.metric("Backtest MAPE", f"{cur_lt_res['metrics']['MAPE']}")
    with m_c4:
        st.metric("Horizon", f"{lt_horizon_months} Months")

    # Long-Term Forecast Visualization (Green historical, Yellow forecast, shaded CI)
    st.markdown(f"#### Long-Term {lt_target} Monthly Projections ({lt_city})")
    fig_lt = plot_long_term_forecast(lt_results, selected_model_key=cur_lt_key, city_name=lt_city)
    st.plotly_chart(fig_lt, use_container_width=True)

    # Comparative Performance Table
    st.markdown("#### Model Evaluation Benchmark (Held-Out Test Months)")
    st.dataframe(lt_results["comparison_table"], use_container_width=True)

    # Forecast Data Table
    st.markdown(f"#### {lt_horizon_months}-Month Forecast Table ({lt_city})")
    lt_fc_df = cur_lt_res["forecast_df"].copy()
    lt_fc_df["Month"] = lt_fc_df["Month"].dt.strftime("%B %Y")
    lt_fc_df["Category"] = lt_fc_df["Forecast_Value"].apply(lambda v: categorize_target(v, target_col=lt_target)[0])

    cols_lt_show = ["Month", "Forecast_Value"]
    if "Lower_CI_95%" in lt_fc_df.columns:
        cols_lt_show += ["Lower_CI_95%", "Upper_CI_95%"]
    cols_lt_show.append("Category")

    st.dataframe(lt_fc_df[cols_lt_show], use_container_width=True)

    # Download Button
    csv_bytes = lt_fc_df[cols_lt_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {lt_horizon_months}-Month {lt_target} Forecast CSV",
        data=csv_bytes,
        file_name=f"{lt_city}_{lt_target}_longterm_forecast_{lt_horizon_months}m.csv",
        mime="text/csv"
    )
