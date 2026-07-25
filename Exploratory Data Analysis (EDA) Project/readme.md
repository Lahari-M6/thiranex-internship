# Netflix Top 10 - Exploratory Data Analysis (EDA) Project

## Overview
This project performs an Exploratory Data Analysis (EDA) on the Netflix
"Top 10" dataset, which tracks weekly Top 10 ranked Films and TV shows
across different countries. The goal is to uncover patterns in content
popularity, compare categories, identify top-performing shows, and present
the findings in a structured report.

This project was completed as part of a data analytics internship task.

## Dataset
**Source:** Kaggle - Netflix "Top 10" TV Shows and Films (Weekly, by Country)


**Size:** ~112,300 rows, 8 columns

## Project Structure
```
Exploratory Data Analysis (EDA) Project/
│
├── netflix_top10_eda.py     # Main analysis script
├── netflix_top10.csv        # Raw dataset (downloaded from Kaggle)
├── README.md                 # This file
└── eda_outputs/               # Generated automatically when script runs
    ├── 01_category_distribution.png
    ├── 02_top_countries.png
    ├── 03_weekly_rank_distribution.png
    ├── 04_cumulative_weeks_distribution.png
    ├── 05_top_shows_by_appearances.png
    ├── 06_top_shows_by_streak.png
    ├── 07_unique_shows_over_time.png
    ├── 08_avg_rank_by_category.png
    ├── 09_country_category_heatmap.png
    ├── 10_correlation_heatmap.png
    └── EDA_Report.md          # Auto-generated written report


## Requirements
- Python 3.8+
- pandas
- matplotlib
- seaborn

Install dependencies:
```
pip install pandas matplotlib seaborn
```

## How to Run
1. Download the dataset CSV from Kaggle and place it in the project folder.
2. Open `netflix_top10_eda.py` and confirm the `DATA_FILE` variable matches
   your CSV's exact filename.
3. Run the script:
```
python netflix_top10_eda.py
```
4. Once it finishes, open the `eda_outputs` folder to view all charts and
   the written report (`EDA_Report.md`).

## Analysis Steps
The script performs the following steps in order:
1. **Load Data** - reads the CSV and confirms it loaded correctly.
2. **Basic Inspection** - checks shape, data types, missing values, and
   summary statistics.
3. **Data Cleaning** - converts dates, fills missing `season_title` values
   for films, removes duplicates, and drops rows with missing essential
   fields.
4. **Univariate Analysis** - distribution of categories, countries, weekly
   ranks, and cumulative weeks in the Top 10.
5. **Top Shows Analysis** - most frequently appearing shows and shows with
   the longest streaks in the Top 10.
6. **Trend Analysis** - how the number of unique shows in the Top 10
   changes week over week.
7. **Relationship Analysis** - average rank by category, a country vs.
   category heatmap, and a correlation check between rank and cumulative
   weeks.
8. **Report Generation** - writes all key numbers and findings into
   `EDA_Report.md`.

## Key Findings
- There is virtually no correlation (~0.05) between a show's weekly rank
  and how long it stays in the Top 10 - ranking high in a given week does
  not predict a longer overall run.
- A small number of shows (e.g. *Stranger Things*, *Money Heist*) account
  for a disproportionate number of Top 10 appearances across countries.
- Some shows achieve very long single streaks (up to 60 cumulative weeks)
  without necessarily being the most frequently appearing titles overall -
  frequency and longevity are different measures of popularity.
- Category counts and average rank being identical between Films and TV
  is a structural feature of the dataset (each category is ranked 1-10
  separately every week), not a real pattern in viewer behavior.

## Notes / Limitations
- The dataset does not include viewership numbers, only rank position,
  so "popularity" here is inferred from rank and streak length rather
  than actual view counts.
- Findings are descriptive (EDA), not predictive - no modeling was
  performed as part of this project.

## Author
(your name here)
Data Analytics Intern