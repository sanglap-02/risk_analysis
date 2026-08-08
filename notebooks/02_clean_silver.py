# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Cleaning & QA → Silver
# MAGIC
# MAGIC Bronze is a faithful copy of the source. Silver is where we start making
# MAGIC **decisions** — and every decision here is one a model validator can ask
# MAGIC about, so each is named, counted and recorded.
# MAGIC
# MAGIC ### The worklist
# MAGIC
# MAGIC Not invented for this notebook — it comes straight out of
# MAGIC `scripts/verify_raw.py`, whose findings are committed in
# MAGIC `data/manifests/raw_verification.json`:
# MAGIC
# MAGIC | # | Finding | What we do |
# MAGIC |---|---|---|
# MAGIC | 1 | **1,570,735** `365243` sentinels in `DAYS_*` | Null every one |
# MAGIC | 2 | `DAYS_FIRST_DRAWING` 56% sentinel, `DAYS_EMPLOYED` 18% | Add a named indicator — the missingness *is* the signal |
# MAGIC | 3 | `bureau_balance` spans `[-96, 0]`, internal panels `[-96, -1]` | Align the bureau month onto the internal convention |
# MAGIC | 4 | `AMT_INCOME_TOTAL` has a 117,000,000 outlier | Cap at the 99.5th percentile, flag the row |
# MAGIC | 5 | Card alone gives 870 bads, below the ~1,000 minimum | Pool card + POS into one panel with `product_type` |
# MAGIC | 6 | Bronze widened `DAYS_*` to double for safe reading | Narrow to int now that nulls are handled |
# MAGIC | 7 | `bureau_balance.STATUS` is a character code | Decode to a bucket number, `X` → NULL not 0 |
# MAGIC
# MAGIC ### What this notebook does **not** do
# MAGIC
# MAGIC No rows are dropped. The exclusion waterfall — dormant accounts, already-bad
# MAGIC at observation, insufficient history — belongs to Phase 3, where it is
# MAGIC counted step by step. Mixing exclusions into cleaning is how population
# MAGIC definitions become impossible to audit.
# MAGIC
# MAGIC Prerequisite: `01_ingest_bronze.py` complete.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json  # noqa: E402
import time  # noqa: E402

from pyspark.sql import DataFrame, functions as F  # noqa: E402
from pyspark.sql.types import BooleanType, DoubleType, IntegerType  # noqa: E402

from credit_risk import cleaning as cln  # noqa: E402

# COMMAND ----------

dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "skip_existing"], "Write mode")
dbutils.widgets.dropdown("run_profile", "true", ["true", "false"], "Build data quality report")

WRITE_MODE = dbutils.widgets.get("mode")
RUN_PROFILE = dbutils.widgets.get("run_profile") == "true"

print(f"mode        : {WRITE_MODE}")
print(f"dq profile  : {RUN_PROFILE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the measured findings
# MAGIC
# MAGIC The sentinel rules are driven by the committed verification report rather
# MAGIC than a hand-maintained list. A hand-maintained list goes stale silently the
# MAGIC moment the source data changes; this one cannot.

# COMMAND ----------

VERIFICATION_PATH = REPO_ROOT / "data" / "manifests" / "raw_verification.json"
PROFILE_PATH = REPO_ROOT / "data" / "manifests" / "raw_profile.json"

if not VERIFICATION_PATH.exists():
    raise FileNotFoundError(
        f"{VERIFICATION_PATH} not found. Run `python scripts/verify_raw.py` locally "
        "and commit data/manifests/, then pull in this Git folder."
    )

verification = json.loads(VERIFICATION_PATH.read_text())
raw_profile = json.loads(PROFILE_PATH.read_text())["tables"]

SENTINEL_COLUMNS = cln.sentinel_columns_from_report(verification)

print("Sentinel columns to scrub (from the committed verification report)")
print("-" * 78)
for table, columns in sorted(SENTINEL_COLUMNS.items()):
    n_rows = raw_profile[table]["n_rows"]
    for column, count in sorted(columns.items(), key=lambda kv: -kv[1]):
        flag = "  <- gets an indicator" if cln.needs_indicator(count, n_rows) else ""
        print(f"  {table}.{column:<28} {count:>9,}  ({count / n_rows:6.1%}){flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared transforms

# COMMAND ----------


def scrub_sentinels(df: DataFrame, table: str) -> tuple[DataFrame, list[str]]:
    """Null every 365243 sentinel; add named indicators where it carries meaning.

    Left in place, "months since first drawing" evaluates to roughly 12,000 for
    over half of previous_application, and every feature derived from it is
    silently wrong rather than obviously wrong.
    """
    columns = SENTINEL_COLUMNS.get(table, {})
    if not columns:
        return df, []

    n_rows = raw_profile[table]["n_rows"]
    added: list[str] = []

    for column, count in columns.items():
        if column not in df.columns:
            continue
        if cln.needs_indicator(count, n_rows):
            name = cln.indicator_name(column)
            df = df.withColumn(name, (F.col(column) == cfg.DAYS_SENTINEL).cast(BooleanType()))
            added.append(name)
        df = df.withColumn(
            column,
            F.when(F.col(column) == cfg.DAYS_SENTINEL, None).otherwise(F.col(column)),
        )

    return df, added


def narrow_days_to_int(df: DataFrame) -> DataFrame:
    """Bronze widened DAYS_* to double so reads could not fail. Narrow them now.

    Safe only after the sentinel has been nulled -- casting 365243.0 to int would
    preserve the poison in a tidier type.
    """
    for field in df.schema.fields:
        if cln.should_narrow_to_int(field.name) and isinstance(field.dataType, DoubleType):
            df = df.withColumn(field.name, F.col(field.name).cast(IntegerType()))
    return df


def cap_outliers(df: DataFrame, table: str) -> tuple[DataFrame, dict[str, float]]:
    """Cap at a percentile and flag, rather than dropping the row.

    Dropping loses a customer; capping keeps them and keeps the fact of capping
    visible to anyone reading the feature later.
    """
    applied: dict[str, float] = {}
    for column, percentile in cln.CAP_COLUMNS.items():
        if column not in df.columns:
            continue
        threshold = df.approxQuantile(column, [percentile], 0.001)[0]
        df = (
            df.withColumn(cln.cap_flag_name(column), F.col(column) > F.lit(threshold))
            .withColumn(column, F.least(F.col(column), F.lit(threshold)))
        )
        applied[column] = threshold
    return df, applied


def write_silver(df: DataFrame, name: str, comment: str) -> dict:
    """Write one silver table and return an audit record."""
    target = cfg.silver(name)
    started = time.time()

    if WRITE_MODE == "skip_existing" and spark.catalog.tableExists(target):
        n_rows = spark.table(target).count()
        print(f"  {name:<24} exists with {n_rows:,} rows — skipped")
        return {"table": name, "status": "skipped", "n_rows": n_rows, "duration_sec": 0.0}

    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
    n_rows = spark.table(target).count()
    spark.sql(f"COMMENT ON TABLE {target} IS '{comment.replace(chr(39), chr(39) * 2)}'")

    print(f"  {name:<24} {n_rows:>12,} rows, {len(df.columns):>3} cols  ({time.time() - started:,.0f}s)")
    return {
        "table": name,
        "status": "written",
        "n_rows": n_rows,
        "duration_sec": round(time.time() - started, 1),
    }


audit: list[dict] = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application tables
# MAGIC
# MAGIC `DAYS_EMPLOYED` is 18% sentinel. That is not a data collection failure — it
# MAGIC means the applicant is not employed (pensioners, unemployed). It becomes
# MAGIC `is_not_employed`, which is a genuine characteristic, and the underlying
# MAGIC column is nulled so no tenure feature is built from a 1,000-year value.

# COMMAND ----------

print("Application tables")
print("-" * 78)

for source, target_name, comment in [
    ("application_train", "application", "Cleaned application data with TARGET. Sentinels nulled, income capped."),
    ("application_holdout", "application_holdout", "Kaggle scoring set. No TARGET. Never used for training."),
]:
    bronze_name = "application_test" if source == "application_holdout" else source
    df = spark.table(cfg.bronze(bronze_name)).drop(sch.CORRUPT_RECORD_COL)

    df, indicators = scrub_sentinels(df, bronze_name)
    df, caps = cap_outliers(df, bronze_name)
    df = narrow_days_to_int(df)

    if indicators:
        print(f"    indicators added : {', '.join(indicators)}")
    for column, threshold in caps.items():
        print(f"    {column} capped at {threshold:,.0f} (p{cln.CAP_COLUMNS[column]:.1%})")

    audit.append(write_silver(df, target_name, comment))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bureau tradelines and the bureau panel
# MAGIC
# MAGIC Two decisions here:
# MAGIC
# MAGIC **`STATUS` decodes to a bucket number, not a day count.** Turning `'3'` into
# MAGIC "61 days" or "90 days" would fabricate precision the source does not contain,
# MAGIC and every feature built on it would inherit the invention.
# MAGIC
# MAGIC **`'X'` becomes NULL, not 0.** Unknown is not the same as current. Collapsing
# MAGIC them understates risk on exactly the accounts where the bureau has least
# MAGIC visibility.

# COMMAND ----------

print("Bureau")
print("-" * 78)

bureau = spark.table(cfg.bronze("bureau")).drop(sch.CORRUPT_RECORD_COL)
bureau, _ = scrub_sentinels(bureau, "bureau")
bureau = narrow_days_to_int(bureau)
audit.append(write_silver(bureau, "bureau", "Cleaned bureau tradelines from other institutions."))

# COMMAND ----------

status_decode = F.create_map(
    *[
        x
        for k, v in cln.BUREAU_STATUS_MAP.items()
        for x in (F.lit(k), F.lit(v).cast(IntegerType()))
    ]
)

bureau_balance = (
    spark.table(cfg.bronze("bureau_balance"))
    .drop(sch.CORRUPT_RECORD_COL)
    .withColumn("status_bucket", status_decode[F.upper(F.trim(F.col("STATUS")))])
    # Align onto the internal panel convention. bureau_balance's month 0 is
    # contemporaneous with the internal panels' month -1, so every bureau month
    # shifts back by one. Without this, bureau windows are off by one against
    # internal ones and the two halves of every feature set disagree on "now".
    .withColumn("months_balance_aligned", F.col("MONTHS_BALANCE") - F.lit(cln.BUREAU_MONTH_OFFSET))
    .withColumn("is_closed", F.col("STATUS") == F.lit("C"))
    .withColumn("is_status_unknown", F.col("STATUS") == F.lit("X"))
)

bureau_balance = (
    bureau_balance
    .withColumn("is_delinquent", F.col("status_bucket") >= F.lit(cln.BUREAU_DELINQUENT_FROM_BUCKET))
    .withColumn("is_90_plus", F.col("status_bucket") >= F.lit(cln.BUREAU_BUCKET_90_PLUS))
)

audit.append(
    write_silver(
        bureau_balance,
        "bureau_balance",
        "Bureau monthly panel. STATUS decoded to bucket 0-5 (X -> NULL). "
        "months_balance_aligned shifts onto the internal panel convention.",
    )
)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT STATUS, status_bucket, is_closed, is_status_unknown, is_delinquent,
               COUNT(*) AS n_rows
        FROM {cfg.silver('bureau_balance')}
        GROUP BY ALL ORDER BY STATUS
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Previous applications and instalment payments
# MAGIC
# MAGIC `previous_application` carries the bulk of the sentinel problem — 1.5M of the
# MAGIC 1.57M occurrences. `DAYS_FIRST_DRAWING` is 56% sentinel, which means it is not
# MAGIC really a duration at all; it becomes `never_drawn` plus a mostly-null column.

# COMMAND ----------

print("Previous applications and instalments")
print("-" * 78)

prev = spark.table(cfg.bronze("previous_application")).drop(sch.CORRUPT_RECORD_COL)
prev, indicators = scrub_sentinels(prev, "previous_application")
prev = narrow_days_to_int(prev)
print(f"    indicators added : {', '.join(indicators)}")
audit.append(write_silver(prev, "previous_application", "Cleaned prior applications. DAYS_ sentinels nulled."))

# COMMAND ----------

instalments = spark.table(cfg.bronze("installments_payments")).drop(sch.CORRUPT_RECORD_COL)
instalments, _ = scrub_sentinels(instalments, "installments_payments")
instalments = narrow_days_to_int(instalments)

# Payment timing and shortfall are the raw material for on-time payment ratio in
# Phase 4. Computing them once here means every downstream notebook agrees on
# what "late" and "short" mean.
instalments = (
    instalments
    .withColumn("days_late", F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"))
    .withColumn("is_late", F.col("DAYS_ENTRY_PAYMENT") > F.col("DAYS_INSTALMENT"))
    .withColumn("payment_shortfall", F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"))
    .withColumn("is_short_payment", F.col("AMT_PAYMENT") < F.col("AMT_INSTALMENT") * F.lit(0.99))
)

audit.append(
    write_silver(
        instalments,
        "installments_payments",
        "Payment-level record with days_late, payment_shortfall and their flags.",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The pooled behaviour panel
# MAGIC
# MAGIC **The key table of this notebook.** Card and POS accounts unioned into one
# MAGIC panel with a `product_type` characteristic.
# MAGIC
# MAGIC The two products do not share a column set. Rather than force them into a
# MAGIC lowest-common-denominator schema — which would discard the card behaviour
# MAGIC that most of the requested attributes depend on — both column sets are
# MAGIC carried and the absent ones are NULL.
# MAGIC
# MAGIC That is not a compromise. WOE gives "missing" its own bin with its own
# MAGIC weight, and `product_type` enters the model as a characteristic in its own
# MAGIC right. The technique chosen in Phase 7 resolves the problem created here.
# MAGIC
# MAGIC Row-level derivations (`utilisation`, `is_overlimit`) live here rather than in
# MAGIC Phase 4 because each encodes a decision about invalid source values —
# MAGIC principally that a zero or missing credit limit makes utilisation undefined,
# MAGIC not infinite. Every downstream notebook would otherwise re-implement that
# MAGIC guard, and they would not all re-implement it the same way.

# COMMAND ----------

card = spark.table(cfg.bronze("credit_card_balance")).drop(sch.CORRUPT_RECORD_COL)
pos = spark.table(cfg.bronze("pos_cash_balance")).drop(sch.CORRUPT_RECORD_COL)


def align_panel(df: DataFrame, product: str) -> DataFrame:
    """Give both products the same column list so they can be unioned."""
    df = df.withColumn("product_type", F.lit(product))

    for column in (*cln.PANEL_CARD_ONLY_COLUMNS, *cln.PANEL_POS_ONLY_COLUMNS):
        if column not in df.columns:
            df = df.withColumn(column, F.lit(None).cast(DoubleType()))

    if product == cln.PRODUCT_CARD:
        # A limit of zero or NULL makes utilisation undefined, not infinite.
        valid_limit = F.col("AMT_CREDIT_LIMIT_ACTUAL") > 0
        df = (
            df.withColumn("has_valid_limit", valid_limit)
            .withColumn(
                "utilisation",
                F.when(valid_limit, F.col("AMT_BALANCE") / F.col("AMT_CREDIT_LIMIT_ACTUAL")),
            )
            .withColumn(
                "is_overlimit",
                F.when(valid_limit, F.col("AMT_BALANCE") > F.col("AMT_CREDIT_LIMIT_ACTUAL")),
            )
            .withColumn(
                "overlimit_amount",
                F.when(
                    valid_limit & (F.col("AMT_BALANCE") > F.col("AMT_CREDIT_LIMIT_ACTUAL")),
                    F.col("AMT_BALANCE") - F.col("AMT_CREDIT_LIMIT_ACTUAL"),
                ).otherwise(F.lit(0.0)),
            )
        )
    else:
        # POS accounts have no revolving limit. NULL, not zero -- zero would read
        # as "fully utilised" once binned.
        df = (
            df.withColumn("has_valid_limit", F.lit(None).cast(BooleanType()))
            .withColumn("utilisation", F.lit(None).cast(DoubleType()))
            .withColumn("is_overlimit", F.lit(None).cast(BooleanType()))
            .withColumn("overlimit_amount", F.lit(None).cast(DoubleType()))
        )

    return df.select(*cln.panel_output_columns())


panel = align_panel(card, cln.PRODUCT_CARD).unionByName(align_panel(pos, cln.PRODUCT_POS))

print("Pooled behaviour panel")
print("-" * 78)
audit.append(
    write_silver(
        panel,
        "panel_pooled",
        "Card + POS monthly panel unioned with product_type. Card-only columns are "
        "NULL for POS by design -- WOE bins missing separately. utilisation is NULL "
        "where the credit limit is zero or absent.",
    )
)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT product_type,
               COUNT(*)                              AS n_rows,
               COUNT(DISTINCT SK_ID_PREV)            AS n_accounts,
               COUNT(DISTINCT SK_ID_CURR)            AS n_customers,
               MIN(MONTHS_BALANCE)                   AS min_month,
               MAX(MONTHS_BALANCE)                   AS max_month,
               ROUND(AVG(utilisation), 4)            AS avg_utilisation,
               SUM(CAST(is_overlimit AS INT))        AS n_overlimit_months,
               SUM(CASE WHEN SK_DPD >= 90 THEN 1 END) AS n_months_90plus
        FROM {cfg.silver('panel_pooled')}
        GROUP BY product_type ORDER BY product_type
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data quality report
# MAGIC
# MAGIC One row per silver column: null rate, approximate cardinality, range. This is
# MAGIC the artefact a validator asks for, and it is also the baseline that Phase 10's
# MAGIC PSI monitoring compares against.
# MAGIC
# MAGIC `approx_count_distinct` rather than exact — on a 37M-row pooled panel the
# MAGIC exact count costs a full shuffle per column and buys nothing at this stage.

# COMMAND ----------

if RUN_PROFILE:
    rows = []
    for name in cln.SILVER_TABLES:
        target = cfg.silver(name)
        if not spark.catalog.tableExists(target):
            continue
        df = spark.table(target)
        n_rows = df.count()

        aggs = []
        for field in df.schema.fields:
            col = F.col(field.name)
            aggs.append(F.count(col).alias(f"{field.name}__nonnull"))
            aggs.append(F.approx_count_distinct(col).alias(f"{field.name}__distinct"))

        stats = df.agg(*aggs).collect()[0].asDict()

        for field in df.schema.fields:
            non_null = stats[f"{field.name}__nonnull"]
            rows.append(
                (
                    name,
                    field.name,
                    field.dataType.simpleString(),
                    n_rows,
                    n_rows - non_null,
                    round((n_rows - non_null) / n_rows, 6) if n_rows else 0.0,
                    stats[f"{field.name}__distinct"],
                )
            )

    dq = spark.createDataFrame(
        rows,
        "table string, column string, data_type string, n_rows long, "
        "n_null long, null_rate double, approx_distinct long",
    ).withColumn("generated_at", F.current_timestamp())

    dq.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        cfg.reporting("silver_data_quality")
    )
    print(f"Data quality report: {dq.count():,} column profiles")
else:
    print("Data quality profile skipped (run_profile=false)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate
# MAGIC
# MAGIC Silver must not lose rows. Nothing in this notebook filters, so any row-count
# MAGIC change against bronze is a bug in a join or a union, not a cleaning decision.

# COMMAND ----------

EXPECTED_ROW_COUNTS = {
    "application": "application_train",
    "application_holdout": "application_test",
    "bureau": "bureau",
    "bureau_balance": "bureau_balance",
    "previous_application": "previous_application",
    "installments_payments": "installments_payments",
}

problems = []
for silver_name, bronze_name in EXPECTED_ROW_COUNTS.items():
    n_silver = spark.table(cfg.silver(silver_name)).count()
    n_bronze = spark.table(cfg.bronze(bronze_name)).count()
    if n_silver != n_bronze:
        problems.append(f"{silver_name}: {n_silver:,} rows vs bronze {n_bronze:,}")

n_panel = spark.table(cfg.silver("panel_pooled")).count()
n_expected = spark.table(cfg.bronze("credit_card_balance")).count() + spark.table(
    cfg.bronze("pos_cash_balance")
).count()
if n_panel != n_expected:
    problems.append(f"panel_pooled: {n_panel:,} rows vs card+POS {n_expected:,}")

if problems:
    raise RuntimeError(
        "Silver gate failed — cleaning must not change row counts:\n  - "
        + "\n  - ".join(problems)
    )

print(f"Silver complete: {len(audit)} tables, panel_pooled has {n_panel:,} rows.")
print("Row counts preserved against bronze throughout.")
print("\nNext: notebooks/03_population_and_target.py")

# COMMAND ----------

audit_df = spark.createDataFrame(
    [(r["table"], r["status"], r["n_rows"], r["duration_sec"]) for r in audit],
    "table string, status string, n_rows long, duration_sec double",
).withColumn("run_at", F.current_timestamp())

audit_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
    cfg.reporting("silver_audit")
)
display(audit_df)
