import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pakistan Retail Revenue Forecast",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/daily_sales.csv",
        parse_dates=["date"]
    )

@st.cache_resource
def load_model():
    return joblib.load(
        "models/revenue_forecast_model.pkl"
    )

daily_sales = load_data()
model = load_model()

features = [
    "day_of_week",
    "month",
    "day",
    "week_of_year",
    "is_weekend",
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "revenue_lag_7",
    "revenue_lag_14",
    "revenue_lag_30",
    "revenue_roll_3",
    "revenue_roll_7",
    "revenue_roll_14",
    "revenue_roll_30",
    "revenue_std_7"
]

latest_date = daily_sales["date"].max()
next_date = latest_date + pd.Timedelta(days=1)

next_row = {
    "day_of_week": next_date.dayofweek,
    "month": next_date.month,
    "day": next_date.day,
    "week_of_year": int(next_date.isocalendar().week),
    "is_weekend": int(next_date.dayofweek >= 5),

    "revenue_lag_1": daily_sales["revenue"].iloc[-1],
    "revenue_lag_2": daily_sales["revenue"].iloc[-2],
    "revenue_lag_3": daily_sales["revenue"].iloc[-3],
    "revenue_lag_7": daily_sales["revenue"].iloc[-7],
    "revenue_lag_14": daily_sales["revenue"].iloc[-14],
    "revenue_lag_30": daily_sales["revenue"].iloc[-30],

    "revenue_roll_3": daily_sales["revenue"].iloc[-3:].mean(),
    "revenue_roll_7": daily_sales["revenue"].iloc[-7:].mean(),
    "revenue_roll_14": daily_sales["revenue"].iloc[-14:].mean(),
    "revenue_roll_30": daily_sales["revenue"].iloc[-30:].mean(),

    "revenue_std_7": daily_sales["revenue"].iloc[-7:].std()
}

next_X = pd.DataFrame([next_row])[features]

pred_log = model.predict(next_X)
next_day_forecast = np.expm1(pred_log[0])

latest_revenue = daily_sales["revenue"].iloc[-1]
avg_7d = daily_sales["revenue"].tail(7).mean()
avg_30d = daily_sales["revenue"].tail(30).mean()

st.title("Pakistan Retail Revenue Forecasting System")

st.caption(
    "Machine learning dashboard for forecasting daily e-commerce revenue "
    "using historical Pakistani retail sales data."
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Next-Day Forecast",
    f"PKR {next_day_forecast:,.0f}"
)

col2.metric(
    "Latest Daily Revenue",
    f"PKR {latest_revenue:,.0f}"
)

col3.metric(
    "7-Day Avg Revenue",
    f"PKR {avg_7d:,.0f}"
)

col4.metric(
    "30-Day Avg Revenue",
    f"PKR {avg_30d:,.0f}"
)

st.caption(
    f"Forecast date: {next_date.strftime('%d %B %Y')}"
)

st.divider()

st.subheader("Historical Daily Revenue")

historical_chart = (
    daily_sales[
        ["date", "revenue"]
    ]
    .set_index("date")
)

st.line_chart(historical_chart)

st.divider()

st.subheader("Recent Revenue Trend")

recent_days = st.slider(
    "Select number of recent days",
    min_value=7,
    max_value=90,
    value=30
)

recent_data = daily_sales.tail(recent_days)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    recent_data["date"],
    recent_data["revenue"],
    label="Daily Revenue"
)

ax.plot(
    recent_data["date"],
    recent_data["revenue"].rolling(7).mean(),
    label="7-Day Average"
)

ax.set_xlabel("Date")
ax.set_ylabel("Revenue (PKR)")
ax.set_title(f"Revenue Trend - Last {recent_days} Days")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.divider()

st.subheader("Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "MAE",
    "PKR 920,095"
)

metric2.metric(
    "RMSE",
    "PKR 2,456,754"
)

metric3.metric(
    "SMAPE",
    "54.3%"
)

st.caption(
    "The final model is a Random Forest trained on log-transformed revenue "
    "using lag, rolling-window, volatility, and calendar-based features."
)

st.divider()

st.subheader("Actual vs Predicted Revenue")

model_df = daily_sales.dropna().copy()

split_index = int(len(model_df) * 0.8)

test_df = model_df.iloc[split_index:].copy()

X_test = test_df[features]
y_test = test_df["revenue"]

pred_log = model.predict(X_test)
test_predictions = np.expm1(pred_log)

prediction_df = pd.DataFrame({
    "date": test_df["date"],
    "Actual Revenue": y_test.values,
    "Predicted Revenue": test_predictions
}).set_index("date")

st.line_chart(prediction_df)

st.caption(
    "The model tracks normal revenue periods reasonably well, "
    "while sudden promotional or event-driven spikes remain harder to predict."
)

st.divider()

st.subheader("Recent Sales Data")

recent_table = (
    daily_sales[
        ["date", "revenue", "orders"]
    ]
    .tail(10)
    .copy()
)

recent_table["date"] = recent_table["date"].dt.strftime("%Y-%m-%d")
recent_table["revenue"] = recent_table["revenue"].round(0)
recent_table["orders"] = recent_table["orders"].astype(int)

recent_table = recent_table.rename(
    columns={
        "date": "Date",
        "revenue": "Revenue (PKR)",
        "orders": "Orders"
    }
)

st.dataframe(
    recent_table,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Project Summary")

st.write(
    """
    This project uses historical Pakistani e-commerce transaction data
    to forecast daily realized sales revenue.

    The original transaction-level dataset was cleaned and filtered to
    identify successful sales. Since the order grand total was repeated
    across multiple product rows, the data was converted to one row per
    order before revenue was calculated.

    Successful orders were then aggregated into daily revenue and order
    counts. Time-series features including lagged revenue, rolling
    averages, recent volatility, and calendar features were created for
    machine learning.

    Several forecasting approaches were evaluated against simple
    baselines. The final model uses a Random Forest trained on
    log-transformed daily revenue.
    """
)

st.info(
    "Model limitation: major promotional or event-driven revenue spikes "
    "are difficult to predict because detailed advertising and campaign "
    "information is not available in the dataset."
)