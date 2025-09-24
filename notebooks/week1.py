# Databricks notebook source
# MAGIC %md
# MAGIC # Week 1 – Data Preprocessor

# COMMAND ----------

# Importing libraries
import pandas as pd
import numpy as np

# COMMAND ----------

# Reading data
sales = pd.read_csv("../data/raw/sales.csv")
calendar = pd.read_csv("../data/raw/calendar.csv")
sell_prices = pd.read_csv("../data/raw/sell_prices.csv")

# COMMAND ----------

# Merge Files
id_cols = [c for c in sales.columns if not c.startswith("d_")]
d_cols = [c for c in sales.columns if c.startswith("d_")]
n_days = len(d_cols)

# Sales Long
sales_long = sales.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")

# Calendar
calendar = calendar.copy()
calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
calendar["date"] = pd.to_datetime(calendar["date"])
calendar = calendar.sort_values("date").reset_index(drop=True)
calendar = calendar.iloc[:n_days].copy()

# Merge Sales Long and Calendar
df = sales_long.merge(calendar, on="d", how="left")

# Merge with Sell Prices
df = df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

# COMMAND ----------

# Robust revenue (treat missing prices as 0 so revenue doesn't become NaN)
df["revenue"] = df["sales"] * df["sell_price"].fillna(0)

# Single SNAP flag per row (use the state the row belongs to)
df["snap_active"] = (
    (df["state_id"].eq("CA") * df["snap_CA"].fillna(0))
    + (df["state_id"].eq("TX") * df["snap_TX"].fillna(0))
    + (df["state_id"].eq("WI") * df["snap_WI"].fillna(0))
).astype(float)

# Event day flags (count how many days in a month had events)
df["event1_flag"] = df["event_name_1"].notna().astype(int)
df["event2_flag"] = df["event_name_2"].notna().astype(int)

# --- MONTHLY at (state_id, item_id) (keeping cat/dept for reference) ---
df_month = df.groupby(
    ["state_id", "cat_id", "dept_id", "item_id", pd.Grouper(key="date", freq="MS")], as_index=False
).agg(
    sales=("sales", "sum"),
    revenue=("revenue", "sum"),
    snap_active=("snap_active", "mean"),  # share of days active in the month
    event1_days=("event1_flag", "sum"),
    event2_days=("event2_flag", "sum"),
    stores_active=("store_id", pd.Series.nunique),  # how many stores sold this item this month
)

# Sales-weighted monthly price across stores within the state
df_month["sell_price"] = df_month["revenue"] / df_month["sales"]
df_month.loc[df_month["sales"] == 0, "sell_price"] = pd.NA

# (Optional) tidy column order
df_month = df_month[
    [
        "state_id",
        "cat_id",
        "dept_id",
        "item_id",
        "date",
        "sales",
        "sell_price",
        "revenue",
        "stores_active",
        "snap_active",
        "event1_days",
        "event2_days",
    ]
].sort_values(["state_id", "cat_id", "dept_id", "item_id", "date"])


# COMMAND ----------
# Create my forecasting dataset
df_forecast = df_month.groupby(["state_id", "dept_id", "date"]).agg({"sales": "sum"}).reset_index()

# Create unique_id
df_forecast["unique_id"] = df_forecast["state_id"] + "_" + df_forecast["dept_id"]

# Reorder columns for time series forecasting
df_forecast = df_forecast[["unique_id", "state_id", "dept_id", "date", "sales"]]
