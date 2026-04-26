# Population Shifts in Finland

This notebook explores how Finland's population has been changing over the past few decades, and tries to predict what might happen by 2040.

The main question we're trying to answer is:

> **How is Finland's population likely to change in the next 10-20 years, and what factors explain the trend**

---

## Data Sources

All data comes from **Statistics Finland** (via Avoindata.fi), which is Finland's official statistics provider.

We used three datasets:

| Dataset | What it contains | Years |
|---|---|---|
| 11re | Population by age group, per municipality | 1972-2025 |
| 12dy | Births, deaths, and natural increase | 1990-2024 |
| 11a7 | Immigration, emigration, net migration | 1990-2024 |

These three datasets were cleaned and merged into one combined dataset covering **308 Finnish municipalities** from **1990 to 2024**.

---

## Inside the Notebook

### 1. Data Loading & Cleaning
Each dataset was loaded, cleaned, and reshaped from wide format to long format. All three were then merged into a single working dataset with 17 columns and zero missing values.

### 2. Exploratory Data Analysis (EDA)
Four charts that give a big-picture view of what's happening:

- **Population growth rate** over time, with key turning points detected from data
- **Natural increase vs net migration** - which one is driving population change?
- **Aging vs growth scatter** - the relationship between getting older and losing people
- **Fastest aging municipalities** - top 15 with both speed and current level

### 3. Interpretation
Direct answers to the four questions from the task:

- Which regions are aging fastest? Rural municipalities, led by Rääkkylä (+16.5% aging in 10 years)
- Which municipalities grow or decline the most? Helsinki suburbs grow; remote rural areas shrink up to -23%
- Birth rate vs immigration as drivers? Migration has been the only growth driver since 2016
- Which areas need more services? 75 municipalities classified as High Pressure (old + shrinking)

### 4. Forecasting Model
Three regression models predict national population from 2025 to 2040:

- **Model A** Linear regression, full data (1990-2024)
- **Model B** Linear regression, pre-spike data (1990-2021)
- **Model C** Polynomial regression, full data (1990-2024)

All three models agree: Finland's population will reach around **5.91–5.93 million by 2040**.

### 5. Clustering
K-Means clustering (k=3) groups all 308 municipalities into three types based on aging rate, aging speed, population growth, and migration:

| Cluster | Count | Profile |
|---|---|---|
| Aging & Shrinking | 101 | Old, aging fast, losing population (-15%) |
| Stable Declining | 131 | Middle ground, slowly shrinking (-8%) |
| Growing Urban | 76 | Young, growing (+5.5%), high migration |

### 6. Anomaly Detection
Two types of unusual patterns were detected:

- **Temporal** — 2023 had the highest net migration in 35 years (z=3.76), while 2022–2024 saw the deepest natural decline on record
- **Spatial** — Sottunga is the most unusual municipality (furthest from its cluster centroid), with very high migration but zero population growth

---

## Key point

1. Finland's population is still growing, but only because of immigration
2. Since 2016, deaths outnumber births every year
3. Only 1 in 4 municipalities is actually growing — the rest are shrinking
4. Rural areas are aging fast and losing people; cities are young and attracting migrants
5. Without continued immigration, Finland's population would begin to decline

---

## How to Run

1. Make sure you have the three CSV files in the same folder: `11re.csv`, `12dy.csv`, `11a7.csv`
2. Open the notebook in Jupyter
3. Run all cells from top to bottom (`Kernel, Restart & Run All`)

**Required libraries:** `pandas`, `numpy`, `matplotlib`, `sklearn`, `scipy`