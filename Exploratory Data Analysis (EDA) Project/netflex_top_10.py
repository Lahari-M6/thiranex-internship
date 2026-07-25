"""
=====================================================================
 Exploratory Data Analysis (EDA) Project
 Dataset : Netflix "Top 10" TV Shows and Films (Weekly, by Country)
 Author  : (your name here)
 Purpose : Internship EDA project - statistical summaries,
           visualizations, correlations, and a written report.
=====================================================================

HOW TO USE THIS SCRIPT
-----------------------
1. Download the CSV from Kaggle and place it in the SAME FOLDER
   as this script.
2. Update the DATA_FILE variable below with the exact file name.
3. Run this script:  python netflix_top10_eda.py
4. Check the "eda_outputs" folder that gets created - it will have
   all the charts (.png) and a written report (EDA_Report.md).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# SETTINGS - change these two lines if needed
# ---------------------------------------------------------------
DATA_FILE = "netflix_top10.csv"     # <-- put your CSV file name here
OUTPUT_DIR = "eda_outputs"          # folder where charts/report are saved

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ---------------------------------------------------------------
def load_data(path):
    """Read the CSV and stop with a clear message if it's missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Make sure the CSV file is in the "
            f"same folder as this script, and DATA_FILE matches its name."
        )
    df = pd.read_csv(path)
    print(f"Loaded '{path}' successfully.")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


# ---------------------------------------------------------------
# STEP 2: BASIC INSPECTION
# ---------------------------------------------------------------
def inspect_data(df):
    print("=" * 60)
    print("STEP 2: BASIC INSPECTION")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nStatistical summary (numeric columns):")
    print(df.describe())
    print()


# ---------------------------------------------------------------
# STEP 3: CLEAN THE DATA
# ---------------------------------------------------------------
def clean_data(df):
    print("=" * 60)
    print("STEP 3: CLEANING DATA")
    print("=" * 60)

    df = df.copy()

    # Convert 'week' to a proper date so we can sort/plot it over time
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

    # season_title is naturally blank for Films (they don't have seasons)
    # so we fill it with a label instead of leaving it as missing data
    df["season_title"] = df["season_title"].fillna("Not Applicable (Film)")

    # Remove exact duplicate rows (same show, same rank, same week, same country)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    # Drop rows only if a truly essential field is missing
    essential_cols = ["week", "show_title", "category", "weekly_rank", "country_name"]
    df = df.dropna(subset=essential_cols)

    # Make sure rank and cumulative weeks are numeric
    df["weekly_rank"] = pd.to_numeric(df["weekly_rank"], errors="coerce")
    df["cumulative_weeks_in_top_10"] = pd.to_numeric(
        df["cumulative_weeks_in_top_10"], errors="coerce"
    )
    df = df.dropna(subset=["weekly_rank", "cumulative_weeks_in_top_10"])

    print(f"Final cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


# ---------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS (looking at one column at a time)
# ---------------------------------------------------------------
def univariate_analysis(df):
    print("=" * 60)
    print("STEP 4: UNIVARIATE ANALYSIS")
    print("=" * 60)

    results = {}

    # --- Category distribution ---
    category_counts = df["category"].value_counts()
    print("\nEntries per category:")
    print(category_counts)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette="viridis", legend=False)
    plt.title("Number of Top 10 Entries by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Entries")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_category_distribution.png")
    plt.close()

    # --- Countries with the most Top 10 entries ---
    top_countries = df["country_name"].value_counts().head(15)
    print("\nTop 15 countries by number of Top 10 entries:")
    print(top_countries)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, palette="mako", legend=False)
    plt.title("Top 15 Countries by Number of Top 10 Entries")
    plt.xlabel("Number of Entries")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_top_countries.png")
    plt.close()

    # --- Distribution of weekly rank ---
    plt.figure(figsize=(7, 5))
    sns.countplot(x="weekly_rank", data=df, hue="weekly_rank", palette="crest", legend=False)
    plt.title("Distribution of Weekly Rank (1 = Most Popular)")
    plt.xlabel("Weekly Rank")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_weekly_rank_distribution.png")
    plt.close()

    # --- Distribution of cumulative weeks in Top 10 ---
    plt.figure(figsize=(7, 5))
    sns.histplot(df["cumulative_weeks_in_top_10"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Cumulative Weeks in Top 10")
    plt.xlabel("Cumulative Weeks in Top 10")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_cumulative_weeks_distribution.png")
    plt.close()

    results["category_counts"] = category_counts
    results["top_countries"] = top_countries
    print()
    return results


# ---------------------------------------------------------------
# STEP 5: TOP SHOWS
# ---------------------------------------------------------------
def top_shows_analysis(df):
    print("=" * 60)
    print("STEP 5: TOP SHOWS")
    print("=" * 60)

    # Most frequent shows by number of times they appeared in the Top 10
    # (counted across all countries and weeks)
    top_by_appearances = df["show_title"].value_counts().head(10)
    print("\nTop 10 shows by number of Top 10 appearances:")
    print(top_by_appearances)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_by_appearances.values, y=top_by_appearances.index, hue=top_by_appearances.index, palette="flare", legend=False)
    plt.title("Top 10 Shows by Number of Top 10 Appearances")
    plt.xlabel("Number of Appearances")
    plt.ylabel("Show Title")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_top_shows_by_appearances.png")
    plt.close()

    # Shows with the longest streak in the Top 10 (highest cumulative weeks)
    top_by_weeks = (
        df.groupby("show_title")["cumulative_weeks_in_top_10"]
        .max()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 shows by longest streak (max cumulative weeks in Top 10):")
    print(top_by_weeks)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_by_weeks.values, y=top_by_weeks.index, hue=top_by_weeks.index, palette="rocket", legend=False)
    plt.title("Top 10 Shows by Longest Streak in Top 10")
    plt.xlabel("Max Cumulative Weeks in Top 10")
    plt.ylabel("Show Title")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_top_shows_by_streak.png")
    plt.close()

    print()
    return top_by_appearances, top_by_weeks


# ---------------------------------------------------------------
# STEP 6: TRENDS OVER TIME
# ---------------------------------------------------------------
def trend_analysis(df):
    print("=" * 60)
    print("STEP 6: TRENDS OVER TIME")
    print("=" * 60)

    # How many different shows appeared in the Top 10 each week (globally)
    weekly_unique_shows = df.groupby("week")["show_title"].nunique().sort_index()
    print("\nUnique shows in Top 10 per week (first 5 weeks):")
    print(weekly_unique_shows.head())

    plt.figure(figsize=(9, 5))
    plt.plot(weekly_unique_shows.index, weekly_unique_shows.values, marker="o", color="darkorange")
    plt.title("Number of Unique Shows in the Top 10 Over Time")
    plt.xlabel("Week")
    plt.ylabel("Unique Shows")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_unique_shows_over_time.png")
    plt.close()
    print()
    return weekly_unique_shows


# ---------------------------------------------------------------
# STEP 7: RELATIONSHIPS BETWEEN VARIABLES ("influencing factors")
# ---------------------------------------------------------------
def relationship_analysis(df):
    print("=" * 60)
    print("STEP 7: RELATIONSHIPS BETWEEN VARIABLES")
    print("=" * 60)

    # Does category (Film vs TV) affect how high something ranks on average?
    avg_rank_by_category = df.groupby("category")["weekly_rank"].mean().sort_values()
    print("\nAverage weekly rank by category (lower = ranks higher on average):")
    print(avg_rank_by_category)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=avg_rank_by_category.index, y=avg_rank_by_category.values, hue=avg_rank_by_category.index, palette="viridis", legend=False)
    plt.title("Average Weekly Rank by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Weekly Rank (lower = more popular)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_avg_rank_by_category.png")
    plt.close()

    # Heatmap: which categories are popular in which of the top countries
    top_10_country_list = df["country_name"].value_counts().head(10).index
    subset = df[df["country_name"].isin(top_10_country_list)]
    pivot = pd.crosstab(subset["country_name"], subset["category"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Top 10 Entries by Country and Category")
    plt.xlabel("Category")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_country_category_heatmap.png")
    plt.close()

    # Correlation between numeric columns (rank vs cumulative weeks)
    numeric_cols = df[["weekly_rank", "cumulative_weeks_in_top_10"]]
    corr = numeric_cols.corr()
    print("\nCorrelation matrix (numeric columns):")
    print(corr)

    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation: Weekly Rank vs Cumulative Weeks")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_correlation_heatmap.png")
    plt.close()

    print()
    return avg_rank_by_category, corr


# ---------------------------------------------------------------
# STEP 8: WRITE THE FINAL REPORT
# ---------------------------------------------------------------
def write_report(df, category_counts, top_countries, top_by_appearances,
                  top_by_weeks, avg_rank_by_category, corr):

    report_path = f"{OUTPUT_DIR}/EDA_Report.md"

    top_category = category_counts.idxmax()
    top_country = top_countries.idxmax()
    most_frequent_show = top_by_appearances.idxmax()
    longest_streak_show = top_by_weeks.idxmax()
    longest_streak_value = int(top_by_weeks.max())
    rank_corr_value = round(corr.loc["weekly_rank", "cumulative_weeks_in_top_10"], 3)

    report_text = f"""# Netflix Top 10 - Exploratory Data Analysis Report

## 1. Introduction
This report explores the Netflix "Top 10" dataset, which tracks the weekly
Top 10 ranked shows and films across different countries. The goal is to
identify patterns in content popularity, compare categories, and highlight
shows with the strongest performance.

## 2. Dataset Overview
- Total records analyzed (after cleaning): **{len(df)}**
- Number of countries covered: **{df['country_name'].nunique()}**
- Number of unique shows/films: **{df['show_title'].nunique()}**
- Date range: **{df['week'].min().date()}** to **{df['week'].max().date()}**

## 3. Key Statistical Summary
- The most common category in the dataset is **{top_category}**, with
  **{category_counts.max()}** entries.
- **{top_country}** has the highest number of Top 10 entries overall
  (**{top_countries.max()}** entries).
- Average weekly rank by category:
{avg_rank_by_category.to_string()}

## 4. Top Performing Shows
- The show that appeared most often across all Top 10 charts is
  **"{most_frequent_show}"** ({top_by_appearances.max()} appearances).
- The longest continuous streak in the Top 10 belongs to
  **"{longest_streak_show}"**, with **{longest_streak_value} cumulative weeks**.

## 5. Correlation and Influencing Factors
- Correlation between weekly rank and cumulative weeks in Top 10:
  **{rank_corr_value}**.
  {"A negative correlation suggests shows that stay longer in the Top 10 tend to hold higher (better) ranks." if rank_corr_value < 0 else "A positive value suggests limited or no strong linear relationship between rank and how long a show stays in the Top 10."}
- Category appears to influence average ranking position (see chart 08),
  and content popularity is unevenly distributed across countries
  (see the country-category heatmap, chart 09).

## 6. Visualizations Generated
All charts are saved in the `{OUTPUT_DIR}` folder:
1. Category distribution
2. Top countries by entries
3. Weekly rank distribution
4. Cumulative weeks distribution
5. Top shows by appearances
6. Top shows by streak length
7. Unique shows over time
8. Average rank by category
9. Country vs category heatmap
10. Correlation heatmap

## 7. Conclusion
The analysis shows clear differences in how content performs across
categories and countries. A small number of shows account for a large
share of long-running Top 10 appearances, and category (Film vs TV)
has a measurable effect on average ranking. Further analysis could
explore seasonal trends or compare English vs non-English content if
that breakdown is available in the raw data.
"""

    with open(report_path, "w") as f:
        f.write(report_text)

    print("=" * 60)
    print(f"Report written to: {report_path}")
    print("=" * 60)


# ---------------------------------------------------------------
# MAIN - runs everything in order
# ---------------------------------------------------------------
def main():
    df = load_data(DATA_FILE)
    inspect_data(df)
    df = clean_data(df)

    uni_results = univariate_analysis(df)
    top_by_appearances, top_by_weeks = top_shows_analysis(df)
    trend_analysis(df)
    avg_rank_by_category, corr = relationship_analysis(df)

    write_report(
        df,
        uni_results["category_counts"],
        uni_results["top_countries"],
        top_by_appearances,
        top_by_weeks,
        avg_rank_by_category,
        corr,
    )

    print("\nAll done! Check the 'eda_outputs' folder for charts and the report.")


if __name__ == "__main__":
    main()