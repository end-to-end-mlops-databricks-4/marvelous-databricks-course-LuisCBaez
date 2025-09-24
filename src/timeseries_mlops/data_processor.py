from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd

from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

from .config import CFG


def get_spark(app_name: str = "week1-preprocess") -> SparkSession:
    """
    Get a SparkSession via Databricks Connect using databricks.yml bundle config.
    """
    spark = DatabricksSession.builder.profile("DEFAULT").clusterId("0911-131432-9ef6b08w").getOrCreate()
    print(f"✅ Connected to Databricks cluster with app name: {app_name}")
    return spark


@dataclass
class DataProcessor:
    """Read raw CSVs from UC Volumes with Spark, transform in pandas, write Delta tables with Spark."""

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        spark = get_spark("week1-preprocess-load")
        reader = spark.read.format("csv").option("header", True).option("inferSchema", True)

        sales = reader.load(CFG.volume_file(CFG.sales_file)).toPandas()
        calendar = reader.load(CFG.volume_file(CFG.calendar_file)).toPandas()
        sell_prices = reader.load(CFG.volume_file(CFG.prices_file)).toPandas()
        return sales, calendar, sell_prices

    @staticmethod
    def make_monthly_forecast_df(
        sales: pd.DataFrame, calendar: pd.DataFrame, sell_prices: pd.DataFrame
    ) -> pd.DataFrame:
        id_cols = [c for c in sales.columns if not c.startswith("d_")]
        d_cols = [c for c in sales.columns if c.startswith("d_")]
        n_days = len(d_cols)

        sales_long = sales.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")

        cal = calendar.copy()
        cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]
        cal["date"] = pd.to_datetime(cal["date"])
        cal = cal.sort_values("date").reset_index(drop=True).iloc[:n_days]

        df = sales_long.merge(cal, on="d", how="left")
        df = df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
        df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")

        if "wm_yr_wk" in df:
            df["wm_yr_wk"] = pd.to_numeric(df["wm_yr_wk"], errors="coerce")

        df["revenue"] = df["sales"] * df["sell_price"].fillna(0)

        for col in ("snap_CA", "snap_TX", "snap_WI"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["snap_active"] = (
            (df["state_id"].eq("CA") * df.get("snap_CA", 0))
            + (df["state_id"].eq("TX") * df.get("snap_TX", 0))
            + (df["state_id"].eq("WI") * df.get("snap_WI", 0))
        ).astype(float)
        df["event1_flag"] = df["event_name_1"].notna().astype(int)
        df["event2_flag"] = df["event_name_2"].notna().astype(int)

        df_month = (
            df.groupby(["state_id", "dept_id", pd.Grouper(key="date", freq="MS")], as_index=False)
            .agg(sales=("sales", "sum"))
            .sort_values(["state_id", "dept_id", "date"])
        )
        df_month = df_month.rename(columns={"date": "ds", "sales": "y"})
        df_month["unique_id"] = df_month["state_id"] + "_" + df_month["dept_id"]
        df_month["ds"] = pd.to_datetime(df_month["ds"])
        return df_month[["unique_id", "state_id", "dept_id", "ds", "y"]]

    @staticmethod
    def split_holdout(df_month: pd.DataFrame, holdout_months: int):
        ds = pd.to_datetime(df_month["ds"])
        max_ds = ds.max()
        cutoff = (max_ds.to_period("M") - (holdout_months - 1)).to_timestamp()
        train = df_month[df_month["ds"] < cutoff].copy()
        test = df_month[df_month["ds"] >= cutoff].copy()
        return train, test

    @staticmethod
    def write_delta(spark: SparkSession, pdf: pd.DataFrame, table_name: str) -> str:
        full = f"{CFG.catalog}.{CFG.schema}.{table_name}"
        sdf = spark.createDataFrame(pdf)
        (sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full))
        return full
