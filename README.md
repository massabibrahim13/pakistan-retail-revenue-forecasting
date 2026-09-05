# Pakistan Retail Revenue Forecasting System

An end-to-end data science and machine learning project that forecasts daily realized revenue from historical Pakistani e-commerce sales data.

## Project Overview

The project starts from raw transaction-level e-commerce data and builds a daily revenue forecasting pipeline.

The workflow includes:
- data cleaning
- missing-value handling
- order-status filtering
- prevention of revenue double-counting
- daily sales aggregation
- time-series exploratory analysis
- lag and rolling feature engineering
- chronological train/test splitting
- baseline comparison
- Random Forest modeling
- log transformation for skewed revenue
- model evaluation
- Streamlit dashboard

## Dataset

The project uses the **Pakistan Largest Ecommerce Dataset**.

The raw dataset contains transaction-level records including:
- order status
- order date
- SKU
- price
- quantity ordered
- order grand total
- product category
- discount amount
- payment method
- customer information

The raw CSV is kept unchanged. Cleaned and transformed data is generated separately.

## Data Cleaning

Key cleaning steps included:
- removing completely empty rows
- removing empty export columns
- standardizing column names
- dropping redundant fields
- parsing order dates
- handling missing values
- filtering to successful order statuses
- keeping one row per order before summing `grand_total`
- excluding zero and negative realized-revenue orders
- aggregating successful orders into daily revenue and order counts

Successful statuses used for realized revenue:
- `complete`
- `received`
- `paid`
- `closed`

## Forecasting Features

The final model uses:
- day of week
- month
- day of month
- week of year
- weekend flag
- revenue lags: 1, 2, 3, 7, 14, and 30 days
- rolling revenue averages: 3, 7, 14, and 30 days
- 7-day revenue volatility

## Model Development

Models and baselines tested included:
- previous-day revenue baseline
- 7-day seasonal baseline
- Random Forest Regressor
- Histogram Gradient Boosting
- log-transformed Random Forest

The final model is a **Random Forest Regressor trained on log-transformed daily revenue**.

## Final Model Performance

On the held-out chronological test period:
- **MAE:** PKR 920,095
- **RMSE:** PKR 2,456,754
- **SMAPE:** 54.3%

The model improved MAE by roughly 23% compared with the previous-day baseline.

## Key Findings

The model performs reasonably well during normal revenue periods but struggles with very large promotional or event-driven spikes.

Recent revenue lags and rolling statistics are the most important predictors.

Simple holiday/event flags produced negligible improvement, suggesting that major spikes are likely driven by factors such as promotions, campaigns, or marketing activity that are not available in the dataset.

## Streamlit App

The Streamlit dashboard includes:
- next-day revenue forecast
- latest daily revenue
- 7-day and 30-day averages
- historical revenue chart
- recent revenue trend
- model performance metrics
- actual vs predicted revenue
- recent sales data

Run locally with:

```bash
streamlit run app.py
```

## Project Structure

```text
e-commerce_project/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── processed/
│   │   └── daily_sales.csv
│   └── raw/
│       └── Pakistan Largest Ecommerce Dataset.csv
├── models/
│   └── revenue_forecast_model.pkl
├── notebooks/
│   ├── data_cleaning.ipynb
│   └── model_evaluation.ipynb
└── screenshots/
```

## Limitations

The model does not have access to:
- promotional campaign data
- advertising spend
- inventory constraints
- external economic indicators
- detailed holiday marketing activity

These missing variables likely explain many of the largest forecasting errors.

## Future Improvements

Possible next steps:
- add promotion and campaign features
- add category-level forecasting
- compare XGBoost or LightGBM
- perform time-series cross-validation
- add prediction intervals
- deploy the Streamlit app publicly
