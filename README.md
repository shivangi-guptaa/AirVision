# 🌫️ AirVision: CPCB Monitoring & 24-Hour Ahead AQI Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Ensemble%20ML-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning intelligence platform combining **Statutory CPCB Piecewise Arithmetic Calculation** for instantaneous monitoring with **Out-of-Time Machine Learning Time-Series Forecasters** to predict 24-hour ahead Air Quality Index (AQI) surges across Delhi-NCR.

---

## 📌 The Engineering Problem & Why ML is Used

In air pollution monitoring, a common novice pitfall is using machine learning to predict *current* AQI from *current* pollutant readings. By statutory definition (CPCB India / US EPA), **instantaneous AQI is a deterministic piecewise formula**:
$$\text{Sub-Index}_p = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{\text{BP}_{\text{high}} - \text{BP}_{\text{low}}} (C_p - \text{BP}_{\text{low}})$$
$$\text{AQI} = \max(\text{Sub-Index}_1, \text{Sub-Index}_2, \dots, \text{Sub-Index}_n)$$

Training an ML model to approximate this formula is computationally redundant. **AirVision bridges the true data science gap**:
1. **Instantaneous Monitoring (Time $t$):** Computed deterministically using the official CPCB piecewise formula with sub-index breakdown.
2. **Next-Day Forecasting (Time $t + 24\text{h}$):** Machine Learning is deployed because **tomorrow's pollutant concentrations do not exist yet**. Models learn atmospheric dispersion, weather correlations, and multi-step autoregressive persistence to issue early public health alerts before severe pollution episodes occur.

---

## 🚀 Key Architectural Features

- **Strict Chronological Splitting (No Lookahead Leakage):** Time-series data from 2020 to 2025 split strictly out-of-time (80% chronological train, 20% future test) ensuring genuine predictive generalization.
- **Multi-Variate Atmospheric Lags:** Captures short-term persistence using $\text{AQI}_{t}, \text{AQI}_{t-1}, \text{AQI}_{t-2}$, along with temporal cyclics (hour, month, day of week) and meteorological telemetry (temperature, humidity, wind velocity).
- **Dual Inference Engine:** 
  - Standard statutory sub-indices for $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$.
  - 24-hour horizon forecasting via Gradient Boosting and Random Forest regressors.
- **Early Warning Advisory:** Automated delta detection triggering public health warnings when atmospheric spikes are predicted.

---

## 📊 Model Evaluation & Benchmarks (24-Hour Horizon)

Evaluated on unseen future holdout test set (1,752 out-of-time intervals):

| Model Architecture | Test $R^2$ Score | Test RMSE | Test MAE | Primary Strengths |
| :--- | :---: | :---: | :---: | :--- |
| **Gradient Boosting (GBR)** | **0.9873** | **19.84** | **9.62** | Top generalization on non-linear atmospheric dispersion |
| **Random Forest Regressor** | **0.9860** | **20.80** | **10.19** | High stability and robustness to sensor outliers |
| **Ridge Regression Baseline** | 0.9842 | 22.10 | 12.40 | Linear baseline with autoregressive lags |

### Feature Attribution Analysis (Random Forest)
- **$\text{PM}_{2.5}$ Concentration:** **81.5%** (Dominant driver of particulate stagnation in Delhi basin)
- **Current AQI Persistence ($\text{AQI}_{t}$):** **7.8%**
- **$\text{PM}_{10}$ Concentration:** **6.9%**
- **$\text{SO}_2$ / Chemical Drivers:** **2.3%**

---

## 📁 Repository Structure

```
AirVision/
│
├── data/
│   └── delhi_last_5_years.csv       # Multi-year hourly Delhi air quality & weather data
│
├── models/
│   ├── gbr_forecast_24h.pkl         # Trained Gradient Boosting Forecaster (< 0.5 MB)
│   ├── rf_forecast_24h.pkl          # Trained Random Forest Forecaster (~16 MB)
│   └── model_metadata.json          # Benchmark metrics & feature configurations
│
├── src/
│   ├── __init__.py
│   ├── cpcb_calculator.py           # Statutory CPCB piecewise formula engine
│   └── train.py                     # Retraining script for 24h forecasting models
│
├── app.py                           # Production Streamlit Interactive Dashboard
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Clean Git configuration
└── README.md
```

---

## 🛠️ Installation & Running the Dashboard

### 1. Clone the Repository
```bash
git clone https://github.com/shivangi-guptaa/AirVision.git
cd AirVision
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. (Optional) Re-train the Forecasting Pipeline
```bash
python src/train.py
```

---

## 📝  Impact Points

* **End-to-End ML Forecasting Platform:** Architected an air quality forecasting system using 5+ years of Delhi-NCR telemetry, separating statutory deterministic CPCB formulas (instantaneous) from predictive ML ensembles (24-hour horizon).
* **Time-Series Feature Engineering:** Formulated multi-step autoregressive lags ($\text{AQI}_{t-k}$), particulate interaction ratios, and cyclic temporal variables on chronological out-of-time splits, eliminating lookahead bias.
* **Model Benchmarking:** Evaluated Gradient Boosting, Random Forest, and regularized baselines; best model achieved an **$R^2$ of 0.987** and **RMSE of 19.84** on unseen future intervals.
* **Production Deployment:** Delivered an interactive Streamlit intelligence suite with CPCB sub-index decomposition, dynamic model switching, and automated health advisories.
