# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Feature Engineering — Internal / Account-Level
# MAGIC
# MAGIC Builds the internal behaviour features for every eligible account × cohort.
# MAGIC
# MAGIC ### The one rule
# MAGIC
# MAGIC **Every window ends at the observation point. Nothing may look past it.**
# MAGIC
# MAGIC That rule is enforced structurally rather than by care: every windowed
# MAGIC aggregate resolves its month range through `features.window_for()`, which is
# MAGIC asserted in tests to never extend beyond the observation point. Re-deriving
# MAGIC the range per feature is exactly how one subtly wrong feature appears out of
# MAGIC a hundred and fifty, and leakage does not announce itself — it makes results
# MAGIC look *better*.
# MAGIC
# MAGIC A leakage assertion runs at the end of the notebook regardless.
# MAGIC
# MAGIC ### Families built here
# MAGIC
# MAGIC | Family | What it captures |
# MAGIC |---|---|
# MAGIC | Tenure & relationship | How long, how many products, how deep |
# MAGIC | Delinquency | Worst DPD, counts by severity, **recency**, trend |
# MAGIC | Utilisation | Level, peak, **volatility and slope** |
# MAGIC | Balance | Level, peak, growth |
# MAGIC | Spend | Volume, frequency, category mix |
# MAGIC | Distress | Cash advance, overlimit, minimum-payment-only |
# MAGIC | Payment history | On-time ratio, lateness, shortfall, regularity |
# MAGIC
# MAGIC Utilisation **trend and volatility** matter more than the level. A customer
# MAGIC steady at 60% is materially safer than one who climbed from 20% to 60% in six
# MAGIC months. Most naive builds compute only the level.
# MAGIC
# MAGIC Prerequisite: `03_population_and_target.py` complete.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F  # noqa: E402

from credit_risk import cleaning as cln  # noqa: E402
from credit_risk import features as fx  # noqa: E402

# COMMAND ----------

dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "skip_existing"], "Write mode")
WRITE_MODE = dbutils.widgets.get("mode")

PANEL = cfg.silver("panel_pooled")
TARGET = cfg.gold("target_behaviour")
DPD = cfg.DPD_COLUMN

print(f"windows      : {fx.WINDOWS} months")
print(f"trend window : {fx.TREND_WINDOW} months")
print(f"catalogue    : {len(fx.INTERNAL_FEATURES)} internal features across "
      f"{len(fx.INTERNAL_FAMILIES)} families")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Windowed aggregates
# MAGIC
# MAGIC `slope` is an ordinary least squares gradient computed as
# MAGIC `cov(month, value) / var(month)` — the closed form, so it needs no UDF and no
# MAGIC collect. A positive utilisation slope means the customer is drawing down
# MAGIC available credit month on month, which is a different and more urgent signal
# MAGIC than a high level held steady.

# COMMAND ----------


def slope(month_col: str, value_col: str):
    """OLS gradient of value against month. NULL when month has no variance."""
    return F.covar_pop(F.col(month_col), F.col(value_col)) / F.nullif(
        F.var_pop(F.col(month_col)), F.lit(0.0)
    )


def windowed_aggregates(panel: DataFrame, obs_point: int, months: int) -> DataFrame:
    """One row per account of aggregates over the trailing `months` window."""
    lo, hi = fx.window_for(obs_point, months)
    w = panel.filter(F.col("MONTHS_BALANCE").between(lo, hi))
    n = months

    is_dpd = lambda t: F.sum(F.when(F.col(DPD) >= t, 1).otherwise(0))  # noqa: E731

    aggs = [
        F.max(DPD).alias(f"max_dpd_{n}m"),
        is_dpd(30).alias(f"n_months_dpd30_{n}m"),
        is_dpd(60).alias(f"n_months_dpd60_{n}m"),
        is_dpd(90).alias(f"n_months_dpd90_{n}m"),
        F.avg("utilisation").alias(f"util_avg_{n}m"),
        F.max("utilisation").alias(f"util_max_{n}m"),
        F.avg("AMT_BALANCE").alias(f"bal_avg_{n}m"),
        F.avg("AMT_DRAWINGS_CURRENT").alias(f"spend_avg_{n}m"),
        F.sum("CNT_DRAWINGS_CURRENT").alias(f"n_txn_{n}m"),
        F.sum("AMT_DRAWINGS_ATM_CURRENT").alias(f"cash_adv_amt_{n}m"),
        F.sum("CNT_DRAWINGS_ATM_CURRENT").alias(f"cash_adv_cnt_{n}m"),
    ]
    return w.groupBy("SK_ID_PREV").agg(*aggs)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Twelve-month detail
# MAGIC
# MAGIC Features that only make sense over the full window: volatility, extremes,
# MAGIC recency, and the composition ratios.
# MAGIC
# MAGIC **Recency is separated from counts deliberately.** In most credit models a
# MAGIC 90+ event two months ago predicts far more than three events two years ago,
# MAGIC and a model given only counts cannot distinguish them.

# COMMAND ----------


def long_window_features(panel: DataFrame, obs_point: int) -> DataFrame:
    lo, hi = fx.window_for(obs_point, 12)
    w = panel.filter(F.col("MONTHS_BALANCE").between(lo, hi))

    # Recency: months between the observation point and the most recent event.
    # NULL when the event never happened -- which WOE will bin separately, and
    # correctly, as a different kind of customer rather than an extreme value.
    last_delinq = F.max(F.when(F.col(DPD) >= 30, F.col("MONTHS_BALANCE")))
    last_overlimit = F.max(F.when(F.col("is_overlimit"), F.col("MONTHS_BALANCE")))

    return w.groupBy("SK_ID_PREV").agg(
        F.min("utilisation").alias("util_min_12m"),
        F.stddev("utilisation").alias("util_std_12m"),
        F.max("AMT_BALANCE").alias("bal_max_12m"),
        F.min("AMT_BALANCE").alias("bal_min_12m"),
        F.stddev("AMT_DRAWINGS_CURRENT").alias("spend_std_12m"),
        F.sum(F.when(F.coalesce(F.col("AMT_DRAWINGS_CURRENT"), F.lit(0.0)) == 0, 1).otherwise(0))
        .alias("n_zero_spend_months_12m"),
        F.sum(F.when(F.col("utilisation") > 0.8, 1).otherwise(0)).alias("n_months_util_gt80_12m"),
        F.sum(F.when(F.col("utilisation") > 1.0, 1).otherwise(0)).alias("n_months_util_gt100_12m"),
        F.sum(F.when(F.col("is_overlimit"), 1).otherwise(0)).alias("n_overlimit_months_12m"),
        F.max("overlimit_amount").alias("max_overlimit_amt_12m"),
        (F.lit(obs_point) - last_delinq).alias("months_since_last_delinq"),
        (F.lit(obs_point) - last_overlimit).alias("months_since_last_overlimit"),
        # Composition: cash advance as a share of all drawings. A customer whose
        # spend is mostly ATM withdrawals is funding daily life on the most
        # expensive money available to them.
        (
            F.sum("AMT_DRAWINGS_ATM_CURRENT")
            / F.nullif(F.sum("AMT_DRAWINGS_CURRENT"), F.lit(0.0))
        ).alias("cash_adv_to_spend_ratio_12m"),
        (
            F.sum("AMT_DRAWINGS_POS_CURRENT")
            / F.nullif(F.sum("AMT_DRAWINGS_CURRENT"), F.lit(0.0))
        ).alias("pct_pos_drawings_12m"),
        (
            F.sum("AMT_DRAWINGS_ATM_CURRENT")
            / F.nullif(F.sum("AMT_DRAWINGS_CURRENT"), F.lit(0.0))
        ).alias("pct_atm_drawings_12m"),
        # Paying at or near the minimum every month is a revolver signal: the
        # customer is servicing the debt without reducing it.
        F.sum(
            F.when(
                (F.col("AMT_INST_MIN_REGULARITY") > 0)
                & (F.col("AMT_PAYMENT_CURRENT") <= F.col("AMT_INST_MIN_REGULARITY") * 1.05),
                1,
            ).otherwise(0)
        ).alias("n_min_payment_only_12m"),
    )


def trend_features(panel: DataFrame, obs_point: int) -> DataFrame:
    """Slopes over the trend window. Direction of travel, not level."""
    lo, hi = fx.window_for(obs_point, fx.TREND_WINDOW)
    w = panel.filter(F.col("MONTHS_BALANCE").between(lo, hi))

    return w.groupBy("SK_ID_PREV").agg(
        slope("MONTHS_BALANCE", "utilisation").alias("util_trend_slope_6m"),
        slope("MONTHS_BALANCE", "AMT_BALANCE").alias("bal_trend_slope_6m"),
        slope("MONTHS_BALANCE", "AMT_DRAWINGS_ATM_CURRENT").alias("cash_adv_trend_6m"),
        (
            F.max(F.when(F.col("MONTHS_BALANCE") == hi, F.col("AMT_BALANCE")))
            - F.max(F.when(F.col("MONTHS_BALANCE") == lo, F.col("AMT_BALANCE")))
        ).alias("bal_growth_6m"),
    )


def at_observation(panel: DataFrame, obs_point: int) -> DataFrame:
    return panel.filter(F.col("MONTHS_BALANCE") == obs_point).select(
        "SK_ID_PREV", F.col("utilisation").alias("util_at_obs")
    )


def clean_streak(panel: DataFrame, obs_point: int) -> DataFrame:
    """Consecutive clean months immediately before the observation point.

    Distinct from "months since last delinquency": an account with no delinquency
    on record has a NULL recency but a full-length clean streak, and those are
    genuinely different statements about the customer.
    """
    lo, hi = fx.window_for(obs_point, 12)
    w = panel.filter(F.col("MONTHS_BALANCE").between(lo, hi))
    last_bad_month = F.max(F.when(F.col(DPD) >= 30, F.col("MONTHS_BALANCE")))
    return w.groupBy("SK_ID_PREV").agg(
        F.coalesce(F.lit(obs_point) - last_bad_month, F.count("*")).alias(
            "n_consecutive_clean_months"
        )
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment history
# MAGIC
# MAGIC From `silver.installments_payments`, which already carries `days_late` and
# MAGIC `payment_shortfall` computed once in Phase 2.
# MAGIC
# MAGIC The instalment table is keyed by `DAYS_INSTALMENT` rather than
# MAGIC `MONTHS_BALANCE`, so the window is converted to days. Getting this conversion
# MAGIC wrong is a quiet way to leak: instalment records after the observation point
# MAGIC would otherwise flow straight into a payment feature.

# COMMAND ----------

DAYS_PER_MONTH = 30.44


def payment_features(obs_point: int) -> DataFrame:
    inst = spark.table(cfg.silver("installments_payments"))
    lo_months, hi_months = fx.window_for(obs_point, 12)

    # DAYS_INSTALMENT is a negative offset from the application date, on the same
    # timeline as MONTHS_BALANCE. Convert the month bounds to day bounds.
    lo_days = lo_months * DAYS_PER_MONTH
    hi_days = (hi_months + 1) * DAYS_PER_MONTH  # exclusive upper edge

    w = inst.filter(
        (F.col("DAYS_INSTALMENT") >= F.lit(lo_days)) & (F.col("DAYS_INSTALMENT") < F.lit(hi_days))
    )

    return w.groupBy("SK_ID_PREV").agg(
        (
            F.sum(F.when(~F.col("is_late"), 1).otherwise(0)) / F.nullif(F.count("*"), F.lit(0))
        ).alias("ontime_pay_ratio_12m"),
        F.avg(F.greatest(F.col("days_late"), F.lit(0))).alias("avg_days_late_12m"),
        F.max("days_late").alias("max_days_late_12m"),
        F.sum(F.col("is_late").cast("int")).alias("n_late_payments_12m"),
        F.sum(F.col("is_short_payment").cast("int")).alias("n_short_payments_12m"),
        (F.sum("AMT_PAYMENT") / F.nullif(F.sum("AMT_INSTALMENT"), F.lit(0.0))).alias(
            "pay_shortfall_ratio_12m"
        ),
        # Near-zero variance in payment timing looks like an autodraft. This is
        # the closest real proxy in this dataset for the requested "autopay
        # enrolment" attribute, which has no direct column.
        F.stddev("days_late").alias("pay_regularity_std_12m"),
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer-level relationship depth
# MAGIC
# MAGIC Cross-product depth is a customer attribute, not an account one, so it is
# MAGIC computed once per customer and broadcast back. Bounded by the observation
# MAGIC point like everything else.

# COMMAND ----------


def customer_features(panel: DataFrame, obs_point: int) -> DataFrame:
    w = panel.filter(F.col("MONTHS_BALANCE") <= obs_point)
    at_obs = panel.filter(F.col("MONTHS_BALANCE") == obs_point)

    depth = w.groupBy("SK_ID_CURR").agg(
        F.countDistinct("SK_ID_PREV").alias("cust_n_accounts"),
        F.countDistinct("product_type").alias("cust_n_product_types"),
        (F.lit(obs_point) - F.min("MONTHS_BALANCE")).alias("cust_tenure_months"),
    )
    balances = at_obs.groupBy("SK_ID_CURR").agg(
        F.sum("AMT_BALANCE").alias("cust_total_balance")
    )
    return depth.join(balances, "SK_ID_CURR", "left")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble

# COMMAND ----------

panel = spark.table(PANEL)
target = spark.table(TARGET)

frames = []
for cohort in cfg.COHORTS:
    obs_point = cohort.obs_point
    print(f"{cohort.name.upper():<4} observation point {obs_point}, "
          f"windows {[fx.window_for(obs_point, m) for m in fx.WINDOWS]}")

    base = target.filter(F.col("cohort") == cohort.name).select(
        "SK_ID_PREV", "SK_ID_CURR", "product_type", "cohort", "obs_point",
        F.col("tenure_at_obs").alias("acct_tenure_months"),
        F.col("n_months_obs").alias("n_months_observed"),
    )

    df = base
    for months in fx.WINDOWS:
        df = df.join(windowed_aggregates(panel, obs_point, months), "SK_ID_PREV", "left")

    df = (
        df.join(long_window_features(panel, obs_point), "SK_ID_PREV", "left")
        .join(trend_features(panel, obs_point), "SK_ID_PREV", "left")
        .join(at_observation(panel, obs_point), "SK_ID_PREV", "left")
        .join(clean_streak(panel, obs_point), "SK_ID_PREV", "left")
        .join(payment_features(obs_point), "SK_ID_PREV", "left")
        .join(customer_features(panel, obs_point), "SK_ID_CURR", "left")
    )

    # Derived comparisons across windows.
    df = (
        df.withColumn("util_delta_3m_vs_12m", F.col("util_avg_3m") - F.col("util_avg_12m"))
        .withColumn("dpd_trend_3m_vs_12m", F.col("max_dpd_3m") - F.col("max_dpd_12m"))
        .withColumn("ever_delinquent", F.col("max_dpd_12m") >= 30)
    )

    frames.append(df)

internal = frames[0]
for extra in frames[1:]:
    internal = internal.unionByName(extra, allowMissingColumns=True)

internal.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    cfg.gold(fx.FEATURE_TABLE_INTERNAL)
)

n_rows = spark.table(cfg.gold(fx.FEATURE_TABLE_INTERNAL)).count()
n_cols = len(spark.table(cfg.gold(fx.FEATURE_TABLE_INTERNAL)).columns)
print(f"\n{cfg.gold(fx.FEATURE_TABLE_INTERNAL)}: {n_rows:,} rows, {n_cols} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leakage assertion
# MAGIC
# MAGIC Structural guards are good; a direct test is better. This recomputes one
# MAGIC feature using **only** months strictly after the observation point and
# MAGIC confirms it differs from the real one. If a future-only recomputation matched
# MAGIC the delivered feature, the feature was reading the future.

# COMMAND ----------

leak_cohort = cfg.COHORT_DEV
obs_point = leak_cohort.obs_point

future_only = (
    panel.filter(F.col("MONTHS_BALANCE") > obs_point)
    .groupBy("SK_ID_PREV")
    .agg(F.max(DPD).alias("future_max_dpd"))
)

comparison = (
    spark.table(cfg.gold(fx.FEATURE_TABLE_INTERNAL))
    .filter(F.col("cohort") == leak_cohort.name)
    .join(future_only, "SK_ID_PREV", "inner")
    .join(
        spark.table(TARGET).filter(F.col("cohort") == leak_cohort.name).select(
            "SK_ID_PREV", "max_dpd_perf"
        ),
        "SK_ID_PREV",
        "inner",
    )
)

n_identical = comparison.filter(
    F.col("max_dpd_12m").isNotNull()
    & (F.col("max_dpd_12m") == F.col("max_dpd_perf"))
    & (F.col("max_dpd_perf") > 0)
).count()
n_total = comparison.filter(F.col("max_dpd_perf") > 0).count()
share = n_identical / n_total if n_total else 0.0

print(f"accounts with any performance-window delinquency : {n_total:,}")
print(f"  ...whose 12m observation feature equals it     : {n_identical:,} ({share:.1%})")

# Some coincidental equality is expected -- a persistently delinquent account can
# genuinely have the same worst DPD in both windows. A high share would mean the
# observation feature is reading the performance window.
if share > 0.50:
    raise RuntimeError(
        f"{share:.1%} of delinquent accounts have identical observation and "
        "performance DPD. That is far above coincidence — a window bound is wrong "
        "and features are reading the future. Do not proceed to Phase 5."
    )
print("\nLeakage assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature dictionary
# MAGIC
# MAGIC Name, family, description and the **expected direction recorded before the
# MAGIC data was examined**. Phase 6 compares observed bad rate by decile against
# MAGIC these predictions; every contradiction is either a data bug or a real
# MAGIC insight, and both are worth chasing.
# MAGIC
# MAGIC Without the prediction written down first, you rationalise whatever you see.

# COMMAND ----------

built = set(spark.table(cfg.gold(fx.FEATURE_TABLE_INTERNAL)).columns)

dictionary = spark.createDataFrame(
    [
        (s.name, s.family, s.description, s.direction, s.card_only, s.name in built)
        for s in fx.INTERNAL_FEATURES
    ],
    "feature string, family string, description string, expected_direction string, "
    "card_only boolean, is_built boolean",
).withColumn("generated_at", F.current_timestamp())

dictionary.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    cfg.gold(fx.FEATURE_DICTIONARY_TABLE)
)

missing = [r.feature for r in dictionary.filter(~F.col("is_built")).collect()]
if missing:
    print(f"Catalogued but not built ({len(missing)}): {', '.join(sorted(missing))}")

display(
    dictionary.groupBy("family").agg(
        F.count("*").alias("catalogued"),
        F.sum(F.col("is_built").cast("int")).alias("built"),
        F.sum(F.when(F.col("expected_direction") == "+", 1).otherwise(0)).alias("expect_riskier"),
        F.sum(F.when(F.col("expected_direction") == "-", 1).otherwise(0)).alias("expect_safer"),
        F.sum(F.when(F.col("expected_direction") == "?", 1).otherwise(0)).alias("no_prior"),
    ).orderBy("family")
)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT cohort, product_type, COUNT(*) AS n_accounts,
               ROUND(AVG(util_avg_12m), 4)               AS avg_util_12m,
               ROUND(AVG(max_dpd_12m), 2)                AS avg_max_dpd_12m,
               ROUND(AVG(cash_adv_to_spend_ratio_12m), 4) AS avg_cash_adv_share,
               ROUND(AVG(ontime_pay_ratio_12m), 4)        AS avg_ontime_ratio,
               ROUND(AVG(n_overlimit_months_12m), 2)      AS avg_overlimit_months
        FROM {cfg.gold(fx.FEATURE_TABLE_INTERNAL)}
        GROUP BY ALL ORDER BY cohort, product_type
    """)
)

print("\nNext: notebooks/05_features_bureau.py")
