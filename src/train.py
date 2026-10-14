"""
AirVision Model Training Pipeline
Trains time-aware forecasting models (Random Forest & Gradient Boosting)
to forecast next-period (6h) and next-day (24h) Air Quality Index (AQI).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. Autoregressive Lags (strictly at or before time t)
    df["aqi_current"] = df["aqi"]
    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_lag2"] = df["aqi"].shift(2)
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm10_lag1"] = df["pm10"].shift(1)

    # 2. Predictive Targets (strictly in the FUTURE)
    # Each row is ~6 hours interval in this dataset
    df["target_6h"] = df["aqi"].shift(-1)   # Next interval (~6 hours ahead)
    df["target_24h"] = df["aqi"].shift(-4)  # 4 intervals ahead (~24 hours / next day)

    # Clean rows with shift NaNs
    df_clean = df.dropna().copy()
    return df_clean


def train_forecasting_models(csv_path: str, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_prepare_data(csv_path)

    # Features known at prediction time t
    features = [
        "pm25", "pm10", "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed",
        "hour", "month", "day_of_week",
        "aqi_current", "aqi_lag1", "aqi_lag2",
        "pm25_lag1", "pm10_lag1"
    ]

    # Chronological Train-Test Split (80% historical train, 20% future test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features]
    X_test = test_df[features]

    # We train models for 24-hour ahead forecasting (the primary public health objective)
    y_train = train_df["target_24h"]
    y_test = test_df["target_24h"]

    print("=" * 60)
    print("AirVision: 24-Hour Ahead AQI Forecasting Model Benchmark")
    print("=" * 60)
    print(f"Total Samples: {len(df)}")
    print(f"Train samples (pre-cutoff): {len(train_df)}")
    print(f"Test samples (out-of-time): {len(test_df)}")
    print(f"Features: {features}")
    print("-" * 60)

    # Model 1: Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)

    print(f"Random Forest  -> R2: {rf_r2:.4f} | RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f}")

    # Model 2: Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)

    gbr_r2 = r2_score(y_test, gbr_pred)
    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
    gbr_mae = mean_absolute_error(y_test, gbr_pred)

    print(f"Gradient Boost -> R2: {gbr_r2:.4f} | RMSE: {gbr_rmse:.2f} | MAE: {gbr_mae:.2f}")

    # Model 3: Baseline Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_mae = mean_absolute_error(y_test, ridge_pred)

    print(f"Ridge Baseline -> R2: {ridge_r2:.4f} | RMSE: {ridge_rmse:.2f} | MAE: {ridge_mae:.2f}")
    print("-" * 60)

    # Feature Importance from Random Forest
    importances = dict(zip(features, rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("Top Feature Attributions (RF):")
    for feat, imp in sorted_imp[:6]:
        print(f"  - {feat:12s}: {imp * 100:.2f}%")

    # Serialize Best Model and Metadata
    model_path = os.path.join(output_dir, "rf_forecast_24h.pkl")
    gbr_path = os.path.join(output_dir, "gbr_forecast_24h.pkl")
    joblib.dump(rf, model_path)
    joblib.dump(gbr, gbr_path)

    metadata = {
        "features": features,
        "target": "target_24h",
        "metrics": {
            "random_forest": {"r2": round(rf_r2, 4), "rmse": round(rf_rmse, 2), "mae": round(rf_mae, 2)},
            "gradient_boosting": {"r2": round(gbr_r2, 4), "rmse": round(gbr_rmse, 2), "mae": round(gbr_mae, 2)},
            "ridge_baseline": {"r2": round(ridge_r2, 4), "rmse": round(ridge_rmse, 2), "mae": round(ridge_mae, 2)},
        },
        "top_features": sorted_imp
    }

    with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels successfully saved to: {output_dir}/")
    return metadata


if __name__ == "__main__":
    data_path = "delhi-aqi-prediction-using-ml/data/delhi_last_5_years.csv"
    if not os.path.exists(data_path):
        data_path = "data/delhi_last_5_years.csv"
    train_forecasting_models(data_path)
