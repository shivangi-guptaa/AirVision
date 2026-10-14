"""
CPCB (Central Pollution Control Board, India) National Air Quality Index (NAQI)
Official Statutory Sub-Index and AQI Calculator.
"""

BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 600.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 800.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 2000.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1000.0, 401, 500),
    ],
}


def calculate_sub_index(pollutant: str, conc: float) -> float:
    """Calculate the sub-index for a single pollutant using linear interpolation."""
    if conc is None or conc < 0:
        return 0.0

    pollutant = pollutant.lower()
    if pollutant not in BREAKPOINTS:
        return 0.0

    table = BREAKPOINTS[pollutant]
    for b_lo, b_hi, i_lo, i_hi in table:
        if b_lo <= conc <= b_hi:
            return round(i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (conc - b_lo), 1)

    # Beyond highest breakpoint -> capped at extreme limit
    last_b_lo, last_b_hi, last_i_lo, last_i_hi = table[-1]
    if conc > last_b_hi:
        slope = (last_i_hi - last_i_lo) / (last_b_hi - last_b_lo)
        return min(round(last_i_hi + slope * (conc - last_b_hi), 1), 500.0)

    return 0.0


def calculate_cpcb_aqi(readings: dict) -> dict:
    """
    Calculate statutory overall AQI from a dict of pollutant concentrations.
    Returns:
        {
            "aqi": float,
            "category": str,
            "dominant_pollutant": str,
            "sub_indices": dict,
            "color": str
        }
    """
    sub_indices = {}
    for pol in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        val = readings.get(pol, 0.0)
        sub_indices[pol] = calculate_sub_index(pol, float(val))

    # Overall AQI is the maximum of the sub-indices
    max_pol = max(sub_indices, key=sub_indices.get)
    overall_aqi = sub_indices[max_pol]

    category, color = get_aqi_category(overall_aqi)

    return {
        "aqi": overall_aqi,
        "category": category,
        "dominant_pollutant": max_pol.upper(),
        "sub_indices": sub_indices,
        "color": color,
    }


def get_aqi_category(aqi: float) -> tuple[str, str]:
    """Returns official CPCB NAQI category and associated hex color."""
    if aqi <= 50:
        return "Good", "#10B981"          # Emerald green
    elif aqi <= 100:
        return "Satisfactory", "#84CC16"  # Light green / Lime
    elif aqi <= 200:
        return "Moderate", "#F59E0B"      # Amber
    elif aqi <= 300:
        return "Poor", "#F97316"          # Orange
    elif aqi <= 400:
        return "Very Poor", "#EF4444"     # Red
    else:
        return "Severe", "#7F1D1D"        # Maroon
