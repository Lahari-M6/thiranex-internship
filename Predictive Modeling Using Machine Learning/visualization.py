"""
02_visualization.py
--------------------
STAGE 2 of the project: Visualization & Insights

Input  : data/cleaned_climate_data.csv   (output of 01_data_cleaning.py)
Output : outputs/*.png                    (individual charts)
         outputs/dashboard.png            (combined summary dashboard)

Run 01_data_cleaning.py first to generate the cleaned CSV.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_PATH = "cleaned_climate_data.csv"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


# ---------- Individual charts ----------

def plot_temperature_trend(df: pd.DataFrame):
    monthly = df.groupby(["Year", "Month", "City"], as_index=False)["Temperature_Avg (°C)"].mean()
    monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

    plt.figure(figsize=(12, 6))
    for city in df["City"].unique():
        city_data = monthly[monthly["City"] == city].sort_values(["Year", "Month"])
        plt.plot(city_data["Period"], city_data["Temperature_Avg (°C)"], marker="o", markersize=3, label=city)
    plt.xticks(rotation=90)
    plt.title("Average Monthly Temperature by City (2024-2025)")
    plt.xlabel("Month")
    plt.ylabel("Avg Temperature (°C)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temperature_trend.png"))
    plt.close()


def plot_avg_aqi_by_city(df: pd.DataFrame):
    aqi_by_city = df.groupby("City")["AQI"].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=aqi_by_city.values, y=aqi_by_city.index, hue=aqi_by_city.index,
                palette="Reds_r", legend=False)
    plt.title("Average AQI by City")
    plt.xlabel("Average AQI")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "avg_aqi_by_city.png"))
    plt.close()


def plot_rainfall_by_season(df: pd.DataFrame):
    rain = df.groupby(["City", "Season"])["Rainfall (mm)"].sum().reset_index()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=rain, x="City", y="Rainfall (mm)", hue="Season")
    plt.title("Total Rainfall by City and Season")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rainfall_by_season.png"))
    plt.close()


def plot_temperature_boxplot(df: pd.DataFrame):
    plt.figure(figsize=(11, 6))
    order = df.groupby("City")["Temperature_Avg (°C)"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="City", y="Temperature_Avg (°C)", order=order, hue="City",
                palette="coolwarm", legend=False)
    plt.title("Temperature Distribution by City (Outlier View)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temperature_boxplot.png"))
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)",
        "Wind_Speed (km/h)", "AQI", "Pressure (hPa)", "Cloud_Cover (%)",
    ]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Between Climate Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
    plt.close()


def plot_aqi_category_distribution(df: pd.DataFrame):
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor"]
    counts = df["AQI_Category"].value_counts().reindex(order)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette="YlOrRd", legend=False)
    plt.title("Distribution of Air Quality Categories (All Cities)")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_category_distribution.png"))
    plt.close()


# ---------- Combined dashboard ----------

def build_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Indian Climate Dashboard (2024-2025)", fontsize=16, fontweight="bold")

    # Panel 1: Avg AQI by city
    aqi_by_city = df.groupby("City")["AQI"].mean().sort_values(ascending=False)
    sns.barplot(x=aqi_by_city.values, y=aqi_by_city.index, hue=aqi_by_city.index,
                palette="Reds_r", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Average AQI by City")
    axes[0, 0].set_xlabel("AQI")

    # Panel 2: Temperature boxplot
    order = df.groupby("City")["Temperature_Avg (°C)"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="City", y="Temperature_Avg (°C)", order=order, hue="City",
                palette="coolwarm", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Temperature Distribution by City")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Panel 3: Rainfall by season (total, all cities combined)
    rain_season = df.groupby("Season")["Rainfall (mm)"].sum().reindex(
        ["Winter", "Summer", "Monsoon", "Post-Monsoon"])
    sns.barplot(x=rain_season.index, y=rain_season.values, hue=rain_season.index,
                palette="Blues", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Total Rainfall by Season (All Cities)")
    axes[1, 0].set_ylabel("Rainfall (mm)")

    # Panel 4: Correlation heatmap
    numeric_cols = ["Temperature_Avg (°C)", "Humidity (%)", "Rainfall (mm)", "AQI"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, "dashboard.png"))
    plt.close()


# ---------- Text insights ----------

def print_insights(df: pd.DataFrame):
    hottest_city = df.groupby("City")["Temperature_Avg (°C)"].mean().idxmax()
    coolest_city = df.groupby("City")["Temperature_Avg (°C)"].mean().idxmin()
    worst_aqi_city = df.groupby("City")["AQI"].mean().idxmax()
    best_aqi_city = df.groupby("City")["AQI"].mean().idxmin()
    wettest_city = df.groupby("City")["Rainfall (mm)"].sum().idxmax()

    print("\n===== KEY INSIGHTS =====")
    print(f"Hottest city (avg temp):    {hottest_city}")
    print(f"Coolest city (avg temp):    {coolest_city}")
    print(f"Worst air quality (AQI):    {worst_aqi_city}")
    print(f"Best air quality (AQI):     {best_aqi_city}")
    print(f"Highest total rainfall:     {wettest_city}")


def main():
    df = load_clean_data(CLEAN_PATH)

    plot_temperature_trend(df)
    plot_avg_aqi_by_city(df)
    plot_rainfall_by_season(df)
    plot_temperature_boxplot(df)
    plot_correlation_heatmap(df)
    plot_aqi_category_distribution(df)
    build_dashboard(df)
    print_insights(df)

    print(f"\nAll charts saved to '{OUT_DIR}/' folder.")


if __name__ == "__main__":
    main()