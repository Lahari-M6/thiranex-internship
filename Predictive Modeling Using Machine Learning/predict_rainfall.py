"""
predict_rainfall.py
--------------------
Use the trained Linear Regression model to predict rainfall for a
NEW day's weather conditions (not from the training data).

Run 03_model_training.py first -- this script loads the model it saves
at outputs/rainfall_model.joblib.

HOW TO USE:
Edit the values inside `example_day` below to whatever weather
conditions you want to test, then run:
    python predict_rainfall.py
"""

import joblib
import pandas as pd

MODEL_PATH = "outputs/rainfall_model.joblib"


def load_model(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["features"]


def predict_rainfall(weather: dict, model, scaler, features) -> float:
    """
    weather must be a dict with these keys:
      'Temperature_Avg (°C)', 'Humidity (%)', 'Wind_Speed (km/h)',
      'Pressure (hPa)', 'Cloud_Cover (%)', 'AQI', 'City', 'Season'

    'City' must be one of the city names from your dataset.
    'Season' must be one of: Winter, Summer, Monsoon, Post-Monsoon.
    """
    row = pd.DataFrame([weather])
    row_encoded = pd.get_dummies(row, columns=["City", "Season"])

    # Make sure the new row has exactly the same columns (in the same
    # order) as the data the model was trained on. Any City/Season
    # dummy column not present in this single row is filled with 0.
    row_encoded = row_encoded.reindex(columns=features, fill_value=0)

    row_scaled = pd.DataFrame(
        scaler.transform(row_encoded), columns=row_encoded.columns
    )
    prediction = model.predict(row_scaled)[0]

    return max(0.0, prediction)  # rainfall can't be negative


if __name__ == "__main__":
    model, scaler, features = load_model()

    # ----- EDIT THESE VALUES to the conditions you want to predict for -----
    example_day = {
        "Temperature_Avg (°C)": 28.5,
        "Humidity (%)": 78,
        "Wind_Speed (km/h)": 12,
        "Pressure (hPa)": 1004,
        "Cloud_Cover (%)": 85,
        "AQI": 95,
        "City": "Mumbai",       # e.g. Delhi, Mumbai, Chennai, Kolkata...
        "Season": "Monsoon",    # Winter / Summer / Monsoon / Post-Monsoon
    }
    # -------------------------------------------------------------------

    result = predict_rainfall(example_day, model, scaler, features)
    print(f"Predicted rainfall: {result:.2f} mm")