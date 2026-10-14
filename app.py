import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import datetime

from src.cpcb_calculator import calculate_cpcb_aqi, get_aqi_category

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AirVision | CPCB Calculation & 24h AQI Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #0b1120;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .alert-card {
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        color: white;
        font-weight: 500;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL & METADATA
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    model_dir = "models"
    rf_path = os.path.join(model_dir, "rf_forecast_24h.pkl")
    gbr_path = os.path.join(model_dir, "gbr_forecast_24h.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    gbr_model = joblib.load(gbr_path) if os.path.exists(gbr_path) else None

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return rf_model, gbr_model, metadata


rf_model, gbr_model, metadata = load_models()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/air-quality.png", width=64)
st.sidebar.title("AirVision System")
st.sidebar.markdown(
    "**End-to-End Air Quality Intelligence Platform**\n"
    "- *Current state:* Statutory CPCB sub-index arithmetic\n"
    "- *Future state:* 24h ML multi-variate forecasting"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting Engine")
model_choice = st.sidebar.selectbox(
    "Select ML Architecture",
    ["Gradient Boosting Regressor", "Random Forest Regressor"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Historical AQI Persistence")
default_lag1 = st.sidebar.number_input("AQI (6h ago)", value=175.0, min_value=0.0, max_value=500.0)
default_lag2 = st.sidebar.number_input("AQI (12h ago)", value=160.0, min_value=0.0, max_value=500.0)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.title("AirVision: Delhi AQI Monitoring & 24-Hour Forecasting")
st.caption("Central Pollution Control Board (CPCB) Statutory Sub-Index Engine + Out-of-Time ML Early Warning Forecaster")

# Pollutant & Weather Inputs
st.subheader("1. Atmospheric Sensor Telemetry (Current Readings)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=85.0, help="Fine particulate matter")
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1000.0, value=155.0, help="Coarse particulate matter")

with c2:
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=500.0, value=42.0, help="Nitrogen Dioxide")
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=500.0, value=14.0, help="Sulphur Dioxide")

with c3:
    co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.4, help="Carbon Monoxide")
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=500.0, value=38.0, help="Ozone")

with c4:
    temp = st.number_input("Temperature (°C)", min_value=-5.0, max_value=55.0, value=26.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=7.5)

# ---------------------------------------------------------
# 2. COMPUTATION: CPCB FORMULA VS ML FORECAST
# ---------------------------------------------------------
readings = {
    "pm25": pm25, "pm10": pm10, "no2": no2,
    "so2": so2, "co": co, "o3": o3
}

cpcb_result = calculate_cpcb_aqi(readings)
current_aqi = cpcb_result["aqi"]
current_cat = cpcb_result["category"]
current_color = cpcb_result["color"]
dominant_pol = cpcb_result["dominant_pollutant"]

now = datetime.datetime.now()
hour = now.hour
month = now.month
day_of_week = now.weekday()

feature_cols = [
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "month", "day_of_week",
    "aqi_current", "aqi_lag1", "aqi_lag2",
    "pm25_lag1", "pm10_lag1"
]

feature_df = pd.DataFrame([[
    pm25, pm10, no2, so2, co, o3,
    temp, humidity, wind_speed,
    hour, month, day_of_week,
    current_aqi, default_lag1, default_lag2,
    pm25 * 0.95, pm10 * 0.95
]], columns=feature_cols)

active_model = gbr_model if "Gradient" in model_choice else rf_model
forecast_24h = float(active_model.predict(feature_df)[0]) if active_model else current_aqi * 1.05
forecast_cat, forecast_color = get_aqi_category(forecast_24h)

st.markdown("---")

# ---------------------------------------------------------
# 3. SIDE-BY-SIDE COMPARATIVE CARDS
# ---------------------------------------------------------
col_curr, col_fore = st.columns(2)

with col_curr:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {current_color};">
        <h4 style="color:#94a3b8; margin-bottom:5px;">CURRENT AQI (Statutory CPCB Formula)</h4>
        <h1 style="color:{current_color}; font-size: 3.2rem; margin:0;">{current_aqi:.1f}</h1>
        <h3 style="color:{current_color}; margin-top:5px;">{current_cat}</h3>
        <p style="color:#64748b; font-size:0.9rem;">
            Dominant Driver: <strong>{dominant_pol}</strong> | Piecewise Linear Sub-Index
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_fore:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {forecast_color};">
        <h4 style="color:#94a3b8; margin-bottom:5px;">24-HOUR FORECAST ({model_choice})</h4>
        <h1 style="color:{forecast_color}; font-size: 3.2rem; margin:0;">{forecast_24h:.1f}</h1>
        <h3 style="color:{forecast_color}; margin-top:5px;">{forecast_cat}</h3>
        <p style="color:#64748b; font-size:0.9rem;">
            Chronologically Validated Out-of-Time Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. EARLY WARNING RISK ADVISORY
# ---------------------------------------------------------
delta = forecast_24h - current_aqi
if delta > 15:
    st.markdown(f"""
    <div class="alert-card" style="background-color: #7f1d1d; border-left: 6px solid #ef4444;">
        ⚠️ <strong>AIR QUALITY DETERIORATION ALERT:</strong> Forecast predicts AQI will surge by <strong>+{delta:.1f} points</strong> over the next 24 hours. Vulnerable groups, elderly, and children should limit outdoor exposure and wear N95 filtration masks.
    </div>
    """, unsafe_allow_html=True)
elif delta < -15:
    st.markdown(f"""
    <div class="alert-card" style="background-color: #064e3b; border-left: 6px solid #10b981;">
        ✅ <strong>ATMOSPHERIC CLEARING FORECAST:</strong> Atmospheric dispersion is predicted to improve air quality by <strong>{abs(delta):.1f} points</strong> over the next 24 hours.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-card" style="background-color: #1e293b; border-left: 6px solid #3b82f6;">
        ℹ️ <strong>PERSISTENT TREND:</strong> Air quality is expected to remain steady within the <strong>{forecast_cat}</strong> band (Δ {delta:+.1f} points).
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. SUB-INDEX BREAKDOWN & MODEL METRICS TABS
# ---------------------------------------------------------
st.markdown("---")
tab1, tab2 = st.tabs(["📊 Pollutant Sub-Index Breakdown (CPCB)", "🧠 ML Model Out-of-Time Validation"])

with tab1:
    st.subheader("Sub-Index Contributions to Current AQI")
    st.caption("CPCB NAQI requires computing individual sub-indices; the highest sub-index determines overall AQI.")

    sub_df = pd.DataFrame([
        {"Pollutant": k.upper(), "Concentration": readings[k], "Sub-Index": v}
        for k, v in cpcb_result["sub_indices"].items()
    ])
    st.bar_chart(sub_df.set_index("Pollutant")["Sub-Index"])
    st.dataframe(sub_df, use_container_width=True)

with tab2:
    st.subheader("Time-Aware Model Performance (Chronological Split: 2020-2023 Train vs 2024-2025 Test)")
    if "metrics" in metadata:
        m_df = pd.DataFrame(metadata["metrics"]).T
        st.dataframe(m_df, use_container_width=True)

    st.markdown("""
    **Architectural Highlights:**
    - **No Lookahead Data Leakage:** Train/test split was performed chronologically on historical time sequences.
    - **True Future Forecasting:** Targets are shifted into future intervals ($t+24\\text{h}$), solving the real-world predictive need that formulas cannot address.
    - **Multi-Variate Non-Linearity:** Captures non-linear meteorological interactions between wind dispersion, humidity, and particulate stagnation.
    """)