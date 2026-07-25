"""
03_model_training.py
--------------------
STAGE 3 of the project: Model Training & Prediction

Goal   : Predict Rainfall (mm) using Linear Regression, trained on
         same-day weather features (Temperature, Humidity, Wind Speed,
         Pressure, Cloud Cover, AQI, City, Season).

Input  : cleaned_climate_data.csv   (output of 01_data_cleaning.py)
Output : outputs/actual_vs_predicted.png
         outputs/residual_plot.png
         outputs/feature_importance.png
         outputs/rainfall_model.joblib   (trained model + scaler, reusable)

Run 01_data_cleaning.py first to generate the cleaned CSV.

NOTE ON EXPECTATIONS:
Rainfall is naturally "spiky" -- most days are 0mm and a few days are
very wet. A plain Linear Regression will usually produce a modest R^2
on this kind of target. That's expected and not a bug. The residual
and actual-vs-predicted plots below are there specifically to help you
see *where* the model struggles (usually on the heavy-rain days).
If you want better accuracy later, a Random Forest / Gradient Boosting
Regressor or a log-transform of Rainfall are the natural next steps.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN_PATH = "cleaned_climate_data.csv"
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "rainfall_model.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

TARGET = "Rainfall (mm)"

# Same-day weather features used to predict rainfall.
# Temperature_Max / Temperature_Min are left out in favor of
# Temperature_Avg to avoid redundant, highly-collinear inputs.
NUMERIC_FEATURES = [
    "Temperature_Avg (°C)",
    "Humidity (%)",
    "Wind_Speed (km/h)",
    "Pressure (hPa)",
    "Cloud_Cover (%)",
    "AQI",
]
CATEGORICAL_FEATURES = ["City", "Season"]


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def prepare_features(df: pd.DataFrame):
    """Select model columns, drop rows with missing values, and
    one-hot encode the categorical columns."""
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    data = df[cols].dropna()

    X = pd.get_dummies(
        data[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
    )
    y = data[TARGET]
    return X, y


def train_model(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n===== MODEL PERFORMANCE (Linear Regression) =====")
    print(f"MAE  : {mae:.2f} mm  (avg. absolute error)")
    print(f"RMSE : {rmse:.2f} mm  (penalizes big misses more)")
    print(f"R^2  : {r2:.3f}  (share of variance explained)")

    return preds, mae, rmse, r2


# ---------- Plots ----------

def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.4, edgecolor="k", linewidth=0.3)
    max_val = max(float(y_test.max()), float(preds.max()))
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title("Actual vs Predicted Rainfall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "actual_vs_predicted.png"))
    plt.close()


def plot_residuals(y_test, preds):
    residuals = y_test.reset_index(drop=True) - pd.Series(preds)

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="steelblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Residual Distribution (Actual - Predicted)")
    plt.xlabel("Residual (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residual_plot.png"))
    plt.close()


def plot_feature_importance(model, feature_names):
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()

    plt.figure(figsize=(9, max(5, len(coefs) * 0.3)))
    colors = ["indianred" if c < 0 else "seagreen" for c in coefs.values]
    plt.barh(coefs.index, coefs.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Linear Regression Coefficients (Effect on Predicted Rainfall)")
    plt.xlabel("Coefficient Value (on standardized features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"))
    plt.close()


def main():
    df = load_clean_data(CLEAN_PATH)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize numeric+dummy features. This makes the coefficients
    # in the feature-importance chart comparable to each other, and
    # keeps the pipeline consistent if you swap in Ridge/Lasso later.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    model = train_model(X_train_scaled, y_train)
    preds, mae, rmse, r2 = evaluate_model(model, X_test_scaled, y_test)

    plot_actual_vs_predicted(y_test, preds)
    plot_residuals(y_test, preds)
    plot_feature_importance(model, X_train_scaled.columns)

    joblib.dump(
        {"model": model, "scaler": scaler, "features": list(X.columns)},
        MODEL_PATH,
    )
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Charts saved to '{OUT_DIR}/' folder.")


if __name__ == "__main__":
    main()