from timeseries_mlops.data_processor import DataProcessor, get_spark
from timeseries_mlops.config import CFG


def main():
    proc = DataProcessor()

    # 1) Load raw
    sales, calendar, sell_prices = proc.load_raw()

    # 2) Prepare monthly dataframe
    df_month = proc.make_monthly_forecast_df(sales, calendar, sell_prices)

    # 3) Split holdout
    train, test = proc.split_holdout(df_month, CFG.holdout_months)

    # 4) Write Delta
    spark = get_spark()
    t_prep = proc.write_delta(spark, df_month, CFG.tbl_prepared)
    t_train = proc.write_delta(spark, train, CFG.tbl_train)
    t_test = proc.write_delta(spark, test, CFG.tbl_test)

    # 5) Print summary
    print(
        "✅ Wrote Delta tables:\n"
        f"  {t_prep}  (rows={len(df_month)})\n"
        f"  {t_train} (rows={len(train)})\n"
        f"  {t_test}  (rows={len(test)})"
    )


if __name__ == "__main__":
    main()
