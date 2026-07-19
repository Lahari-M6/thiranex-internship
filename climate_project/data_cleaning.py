"""
01_data_cleaning.py
--------------------
STAGE 1 of the project: Data Cleaning & Preprocessing

Input  : data/Indian_Climate_Dataset_2024_2025.csv   (raw data)
Output : data/cleaned_climate_data.csv                (cleaned data used by Stage 2)

What this script does:
  1. Loads the raw dataset and inspects its shape/structure
  2. Checks for missing values
  3. Checks for duplicate rows
  4. Fixes data types (Date -> datetime)
  5. Detects outliers using the IQR method on key numeric columns
  6. Adds helpful derived columns (Month, Year, Season)
  7. Saves a clean, analysis-ready CSV
"""

import pandas as pd

RAW_PATH = "Indian_Climate_Dataset_2024_2025.csv"
CLEAN_PATH = "cleaned_climate_data.csv"

NUMERIC_COLS = [
    "Temperature_Max (°C)",
    "Temperature_Min (°C)",
    "Temperature_Avg (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind_Speed (km/h)",
    "AQI",
    "Pressure (hPa)",
    "Cloud_Cover (%)",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values per column ---")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("\n--- Duplicate rows ---")
    dup_count = df.duplicated().sum()
    print(f"{dup_count} duplicate rows found.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicate rows (if any)
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {before - len(df)} duplicate rows.")

    # 2. Handle missing values
    #    Numeric columns -> fill with column median (robust to outliers)
    #    Categorical columns -> fill with mode
    for col in NUMERIC_COLS:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median ({median_val:.2f})")

    for col in ["City", "State", "AQI_Category"]:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled missing values in '{col}' with mode ('{mode_val}')")

    # 3. Fix data types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 4. Derived columns for easier analysis later
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    def to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Summer"
        if month in (6, 7, 8, 9):
            return "Monsoon"
        return "Post-Monsoon"

    df["Season"] = df["Month"].apply(to_season)

    return df


def detect_outliers(df: pd.DataFrame) -> None:
    print("\n--- Outlier check (IQR method, 1.5x rule) ---")
    for col in NUMERIC_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} potential outliers "
                  f"(valid range ~[{lower:.1f}, {upper:.1f}])")
        else:
            print(f"{col}: no outliers detected")


def main():
    df = load_data(RAW_PATH)
    inspect_data(df)
    detect_outliers(df)
    df_clean = clean_data(df)

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset saved to: {CLEAN_PATH}")
    print(f"Final shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")


if __name__ == "__main__":
    main()