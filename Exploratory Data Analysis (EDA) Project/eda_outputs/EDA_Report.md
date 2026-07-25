# Netflix Top 10 - Exploratory Data Analysis Report

## 1. Introduction
This report explores the Netflix "Top 10" dataset, which tracks the weekly
Top 10 ranked shows and films across different countries. The goal is to
identify patterns in content popularity, compare categories, and highlight
shows with the strongest performance.

## 2. Dataset Overview
- Total records analyzed (after cleaning): **112300**
- Number of countries covered: **94**
- Number of unique shows/films: **3459**
- Date range: **2021-07-04** to **2022-08-21**

## 3. Key Statistical Summary
- The most common category in the dataset is **Films**, with
  **56150** entries.
- **Argentina** has the highest number of Top 10 entries overall
  (**1200** entries).
- Average weekly rank by category:
category
Films    5.5
TV       5.5

## 4. Top Performing Shows
- The show that appeared most often across all Top 10 charts is
  **"Stranger Things"** (3119 appearances).
- The longest continuous streak in the Top 10 belongs to
  **"Pasión de Gavilanes"**, with **60 cumulative weeks**.

## 5. Correlation and Influencing Factors
- Correlation between weekly rank and cumulative weeks in Top 10:
  **0.051**.
  A positive value suggests limited or no strong linear relationship between rank and how long a show stays in the Top 10.
- Category appears to influence average ranking position (see chart 08),
  and content popularity is unevenly distributed across countries
  (see the country-category heatmap, chart 09).

## 6. Visualizations Generated
All charts are saved in the `eda_outputs` folder:
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
