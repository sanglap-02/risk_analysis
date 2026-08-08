# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Population & Target Definition
# MAGIC
# MAGIC **The most important notebook in the project.** Everything downstream inherits
# MAGIC the decisions made here, and none of them can be fixed later without redoing
# MAGIC the feature engineering.
# MAGIC
# MAGIC ### Why this comes before feature engineering
# MAGIC
# MAGIC A feature like "maximum days past due over the trailing 12 months" is
# MAGIC meaningless until you say *twelve months before what*. Without an observation
# MAGIC point, that feature silently includes months from after the moment of
# MAGIC decision — which is exactly the delinquency we are trying to predict.
# MAGIC
# MAGIC That is **leakage**, and it is insidious because it makes results look
# MAGIC *better*, not worse. Nothing errors. KS comes out at 80. You find out in
# MAGIC production.
# MAGIC
# MAGIC ```
# MAGIC   MONTHS_BALANCE   -24 ─────────────── -13 │ -12 ─────────────── -1
# MAGIC                     └── FEATURES ───────┘  ↑  └──── LABEL ────────┘
# MAGIC                                    observation point
# MAGIC                     everything we may know  │  what we predict
# MAGIC ```
# MAGIC
# MAGIC ### What this notebook produces
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |---|---|
# MAGIC | `gold.target_behaviour` | One row per eligible account × cohort, with the label |
# MAGIC | `gold.target_application` | One row per customer for the secondary model |
# MAGIC | `reporting.exclusion_waterfall` | Every exclusion step with the count it dropped |
# MAGIC
# MAGIC The waterfall is not a diagnostic. It is a **deliverable** — the first thing a
# MAGIC model validator asks to see, because it is the only way to know whether a
# MAGIC population was defined or merely arrived at.
# MAGIC
# MAGIC Prerequisite: `02_clean_silver.py` complete.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F  # noqa: E402

from credit_risk import cleaning as cln  # noqa: E402
from credit_risk import population as pop  # noqa: E402

# COMMAND ----------

dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "skip_existing"], "Write mode")
WRITE_MODE = dbutils.widgets.get("mode")

PANEL = cfg.silver("panel_pooled")
DPD = cfg.DPD_COLUMN

print(f"panel      : {PANEL}")
print(f"dpd column : {DPD}")
print(f"bad        : max({DPD}) >= {cfg.BAD_DPD_THRESHOLD} in the performance window")
print(f"indet      : {cfg.INDETERMINATE_DPD_LOW} <= max({DPD}) < {cfg.BAD_DPD_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-cohort aggregates
# MAGIC
# MAGIC Three summaries per account, each strictly bounded by its window. The
# MAGIC observation aggregates may never touch a month after the observation point;
# MAGIC the performance aggregates may never touch a month at or before it. Those two
# MAGIC filters are the entire anti-leakage guarantee, so they are written once here
# MAGIC rather than repeated per feature.

# COMMAND ----------


def cohort_frame(cohort) -> DataFrame:
    """Assemble one row per account with everything the waterfall needs to judge it."""
    obs_lo, obs_hi = cohort.obs_window
    perf_lo, perf_hi = cohort.perf_window
    panel = spark.table(PANEL)

    # Lifetime facts. MIN over all months is the account's first appearance, so
    # tenure is measured from the panel itself rather than assumed.
    lifetime = panel.groupBy("SK_ID_PREV", "SK_ID_CURR", "product_type").agg(
        F.min("MONTHS_BALANCE").alias("first_month"),
        F.max("MONTHS_BALANCE").alias("last_month"),
    )

    # Observation window: features may draw on this, and nothing later.
    obs = (
        panel.filter(F.col("MONTHS_BALANCE").between(obs_lo, obs_hi))
        .groupBy("SK_ID_PREV")
        .agg(
            F.count("*").alias("n_months_obs"),
            F.max(DPD).alias("max_dpd_obs"),
            F.max("AMT_BALANCE").alias("max_balance_obs"),
            F.sum("AMT_DRAWINGS_CURRENT").alias("sum_drawings_obs"),
            F.max("utilisation").alias("max_utilisation_obs"),
        )
    )

    # State at the observation point itself: "is this account already bad?"
    at_obs = (
        panel.filter(F.col("MONTHS_BALANCE") == cohort.obs_point)
        .select(
            "SK_ID_PREV",
            F.col(DPD).alias("dpd_at_obs"),
            F.col("NAME_CONTRACT_STATUS").alias("status_at_obs"),
        )
    )

    # Performance window: the label comes from here and nowhere else.
    perf = (
        panel.filter(F.col("MONTHS_BALANCE").between(perf_lo, perf_hi))
        .groupBy("SK_ID_PREV")
        .agg(
            F.count("*").alias("n_months_perf"),
            F.max(DPD).alias("max_dpd_perf"),
        )
    )

    return (
        lifetime.join(obs, "SK_ID_PREV", "left")
        .join(at_obs, "SK_ID_PREV", "left")
        .join(perf, "SK_ID_PREV", "left")
        .withColumn("cohort", F.lit(cohort.name))
        .withColumn("obs_point", F.lit(cohort.obs_point))
        .withColumn("tenure_at_obs", F.lit(cohort.obs_point) - F.col("first_month"))
        .fillna({"n_months_obs": 0, "n_months_perf": 0})
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## The exclusion waterfall
# MAGIC
# MAGIC Applied in order, counting what each step drops. Two kinds of exclusion, and
# MAGIC the distinction matters:
# MAGIC
# MAGIC **Data availability** — we cannot build features or observe an outcome. These
# MAGIC are limits of the data, not statements about risk.
# MAGIC
# MAGIC **Risk policy** — the account is not a candidate for the decision this model
# MAGIC drives. An account already 90+ DPD at the observation point is excluded
# MAGIC because there is nothing left to predict: it has already happened. Leaving
# MAGIC those in inflates every performance metric and the model looks superb at
# MAGIC identifying customers who already defaulted.

# COMMAND ----------

# The step definitions -- names, kinds and rationale -- live in
# credit_risk.population so they can be tested without a cluster. Only the Spark
# expression that implements each one is written here, keyed by the same step id.

STEP_CONDITIONS = {
    "has_history": lambda c: F.col("first_month").isNotNull(),
    "min_tenure": lambda c: F.col("tenure_at_obs") >= cfg.MIN_ACCOUNT_TENURE_MONTHS,
    "full_obs_window": lambda c: F.col("n_months_obs") >= cfg.MIN_HISTORY_MONTHS_BEFORE_OBS,
    "min_perf_coverage": lambda c: F.col("n_months_perf") >= cfg.MIN_COVERAGE_MONTHS_AFTER_OBS,
    "not_closed": lambda c: (
        F.col("status_at_obs").isNull() | ~F.col("status_at_obs").isin(*pop.CLOSED_STATUSES)
    ),
    "not_already_bad": lambda c: (
        F.coalesce(F.col("dpd_at_obs"), F.lit(0)) < cfg.BAD_DPD_THRESHOLD
    ),
    # Dormancy is only measurable on card accounts, which have a balance and
    # drawings. POS accounts are amortising loans -- they cannot go dormant while
    # open, so they pass by construction rather than by a test we cannot run.
    "not_dormant": lambda c: (
        (F.col("product_type") != F.lit(cln.PRODUCT_CARD))
        | (F.coalesce(F.col("max_balance_obs"), F.lit(0.0)) > 0)
        | (F.coalesce(F.col("sum_drawings_obs"), F.lit(0.0)) > 0)
    ),
}

missing = set(pop.EXCLUSION_STEP_KEYS) - set(STEP_CONDITIONS)
if missing:
    raise RuntimeError(f"No Spark condition implemented for exclusion step(s): {sorted(missing)}")


def waterfall_steps(cohort):
    """(name, kind, condition) in the order the policy defines."""
    return [
        (step.name, step.kind, STEP_CONDITIONS[step.key](cohort))
        for step in pop.EXCLUSION_STEPS
    ]


def apply_waterfall(df: DataFrame, cohort) -> tuple[DataFrame, list[dict]]:
    """Filter step by step, recording the count dropped at each one.

    Implemented as a single cumulative pass rather than an iterative
    filter-and-count. Two reasons:

    1. Serverless compute rejects `.cache()` (PERSIST TABLE is not supported),
       and without caching an iterative version would re-read and re-aggregate
       the 13.8M-row panel once per step.
    2. It is simply better. One Spark action produces every count, instead of
       one action per step per cohort.

    Each step gets a cumulative boolean: "passed this step and every step before
    it". Summing those gives the survivor count at each stage, and consecutive
    differences give what each step dropped.
    """
    steps = waterfall_steps(cohort)

    cumulative = F.lit(True)
    flag_names = []
    flagged = df
    for i, (_, _, condition) in enumerate(steps, start=1):
        cumulative = cumulative & condition
        name = f"_passed_{i}"
        flagged = flagged.withColumn(name, cumulative)
        flag_names.append(name)

    counts = flagged.agg(
        F.count("*").alias("n_start"),
        *[F.sum(F.col(n).cast("long")).alias(n) for n in flag_names],
    ).collect()[0]

    n_start = counts["n_start"]
    rows = [
        {
            "cohort": cohort.name,
            "step": 0,
            "name": "starting population (all accounts in the pooled panel)",
            "kind": "start",
            "n_in": n_start,
            "n_dropped": 0,
            "n_out": n_start,
            "pct_dropped": 0.0,
        }
    ]

    previous = n_start
    for i, (name, kind, _) in enumerate(steps, start=1):
        # A cumulative flag is NULL only if every row failed an earlier step.
        surviving = counts[f"_passed_{i}"] or 0
        dropped = previous - surviving
        rows.append(
            {
                "cohort": cohort.name,
                "step": i,
                "name": name,
                "kind": kind,
                "n_in": previous,
                "n_dropped": dropped,
                "n_out": surviving,
                "pct_dropped": round(dropped / previous, 6) if previous else 0.0,
            }
        )
        previous = surviving

    # The eligible set is the final cumulative condition, applied once.
    eligible = flagged.filter(F.col(flag_names[-1])).drop(*flag_names)
    return eligible, rows


# COMMAND ----------

# MAGIC %md
# MAGIC ## Labelling
# MAGIC
# MAGIC **Indeterminates are excluded from training, not relabelled as good.**
# MAGIC
# MAGIC A model learns the boundary between two classes. Pushing ambiguous cases into
# MAGIC the good pile blurs that boundary and makes the model worse at separating the
# MAGIC clear ones. They are kept in the output table with `is_indeterminate = true`
# MAGIC so Phase 10 can score them: their mean score must land **between** goods and
# MAGIC bads. If it does not, the bad definition is wrong.

# COMMAND ----------


def label(df: DataFrame) -> DataFrame:
    max_dpd = F.coalesce(F.col("max_dpd_perf"), F.lit(0))
    return (
        df.withColumn("max_dpd_perf", max_dpd)
        .withColumn("target", (max_dpd >= cfg.BAD_DPD_THRESHOLD).cast("int"))
        .withColumn(
            "is_indeterminate",
            (max_dpd >= cfg.INDETERMINATE_DPD_LOW) & (max_dpd < cfg.BAD_DPD_THRESHOLD),
        )
    )


# COMMAND ----------

print("Building cohorts\n")
cohort_tables = []
waterfall_rows: list[dict] = []

for cohort in cfg.COHORTS:
    obs_lo, obs_hi = cohort.obs_window
    perf_lo, perf_hi = cohort.perf_window
    print(f"{cohort.name.upper():<4} obs [{obs_lo}, {obs_hi}]  perf [{perf_lo}, {perf_hi}]")
    print("-" * 78)

    eligible, rows = apply_waterfall(cohort_frame(cohort), cohort)
    waterfall_rows.extend(rows)

    for r in rows:
        marker = "" if r["kind"] != "policy" else "  [policy]"
        print(f"  {r['step']}. {r['name']:<52} {r['n_out']:>9,}  (-{r['n_dropped']:,}){marker}")

    labelled = label(eligible)
    n_total = labelled.count()
    n_bad = labelled.filter(F.col("target") == 1).count()
    n_indet = labelled.filter(F.col("is_indeterminate")).count()
    print(
        f"\n  eligible {n_total:,} | bads {n_bad:,} ({n_bad / n_total:.2%}) "
        f"| indeterminate {n_indet:,} ({n_indet / n_total:.2%})\n"
    )

    cohort_tables.append(labelled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the behaviour target table

# COMMAND ----------

target_behaviour = cohort_tables[0]
for extra in cohort_tables[1:]:
    target_behaviour = target_behaviour.unionByName(extra)

target_behaviour = target_behaviour.select(
    "SK_ID_PREV",
    "SK_ID_CURR",
    "product_type",
    "cohort",
    "obs_point",
    "target",
    "is_indeterminate",
    "max_dpd_perf",
    "max_dpd_obs",
    "dpd_at_obs",
    "tenure_at_obs",
    "n_months_obs",
    "n_months_perf",
    "first_month",
    "last_month",
)

target_behaviour.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cfg.gold("target_behaviour"))

spark.sql(
    f"""COMMENT ON TABLE {cfg.gold('target_behaviour')} IS
    'Behaviour scorecard target. One row per eligible account x cohort.
     bad = max({DPD}) >= {cfg.BAD_DPD_THRESHOLD} in the performance window.
     Indeterminates ({cfg.INDETERMINATE_DPD_LOW}-{cfg.BAD_DPD_THRESHOLD - 1} DPD) are
     flagged and must be filtered out before training.'"""
)

display(
    spark.sql(f"""
        SELECT cohort, product_type,
               COUNT(*)                                       AS n_accounts,
               SUM(target)                                    AS n_bads,
               ROUND(AVG(target), 4)                          AS bad_rate,
               SUM(CAST(is_indeterminate AS INT))             AS n_indeterminate,
               ROUND(AVG(tenure_at_obs), 1)                   AS avg_tenure_months
        FROM {cfg.gold('target_behaviour')}
        GROUP BY ALL ORDER BY cohort, product_type
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The secondary model's population
# MAGIC
# MAGIC Known-customer application scoring: predict performance on the *new* loan
# MAGIC using the customer's prior relationship behaviour plus bureau.
# MAGIC
# MAGIC No windows are needed. Every `MONTHS_BALANCE` is negative — that is, strictly
# MAGIC before the application — so all panel history precedes the label by
# MAGIC construction. This is a real industry model type, not a workaround.
# MAGIC
# MAGIC Its stated limitation stands: Home Credit ships no application date, so there
# MAGIC is no genuine out-of-time split. Phase 10 must present the holdout as a random
# MAGIC stratified holdout and say so plainly.

# COMMAND ----------

panel_customers = spark.table(PANEL).groupBy("SK_ID_CURR").agg(
    F.count("*").alias("n_panel_months"),
    F.countDistinct("SK_ID_PREV").alias("n_accounts"),
    F.countDistinct("product_type").alias("n_product_types"),
    F.min("MONTHS_BALANCE").alias("first_month"),
)

target_application = (
    spark.table(cfg.silver("application"))
    .select("SK_ID_CURR", F.col("TARGET").alias("target"))
    .join(panel_customers, "SK_ID_CURR", "inner")
    .withColumn("is_indeterminate", F.lit(False))
)

target_application.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cfg.gold("target_application"))

n_app = target_application.count()
n_app_bad = target_application.filter(F.col("target") == 1).count()
n_app_total = spark.table(cfg.silver("application")).count()

print(f"application_train customers          : {n_app_total:,}")
print(f"  ...with panel history (modellable) : {n_app:,} ({n_app / n_app_total:.1%})")
print(f"  bads                               : {n_app_bad:,} ({n_app_bad / n_app:.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Waterfall report

# COMMAND ----------

waterfall_df = spark.createDataFrame(
    [
        (r["cohort"], r["step"], r["name"], r["kind"], r["n_in"], r["n_dropped"], r["n_out"], r["pct_dropped"])
        for r in waterfall_rows
    ],
    "cohort string, step int, name string, kind string, n_in long, "
    "n_dropped long, n_out long, pct_dropped double",
).withColumn("generated_at", F.current_timestamp())

waterfall_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cfg.reporting("exclusion_waterfall"))

display(waterfall_df.orderBy("cohort", "step"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate
# MAGIC
# MAGIC Three things must hold before Phase 4 spends any effort on features.

# COMMAND ----------

dev = spark.table(cfg.gold("target_behaviour")).filter(F.col("cohort") == cfg.COHORT_DEV.name)
oot = spark.table(cfg.gold("target_behaviour")).filter(F.col("cohort") == cfg.COHORT_OOT.name)

n_dev, n_dev_bad = dev.count(), dev.filter(F.col("target") == 1).count()
n_oot, n_oot_bad = oot.count(), oot.filter(F.col("target") == 1).count()
dev_rate = n_dev_bad / n_dev
oot_rate = n_oot_bad / n_oot

problems, warnings = [], []

# 1. Enough bads to build a scorecard at all. This is the check that would have
#    caught the SK_DPD_DEF mistake in hour three rather than week five.
if not pop.has_enough_bads(n_dev_bad):
    problems.append(
        f"dev cohort has {n_dev_bad:,} bads, below the "
        f"{pop.MIN_BADS_FOR_SCORECARD:,} needed for a stable scorecard. Revisit "
        "the bad definition before engineering a single feature."
    )

# 2. Every bin in Phase 7 needs a minimum number of bads. That puts a hard
#    ceiling on scorecard length, whatever the candidate feature count.
if (warning := pop.scorecard_length_warning(n_dev_bad)) is not None:
    warnings.append(warning)

# 3. Dev and OOT must be comparable, or the out-of-time test stops being a test
#    of the model and becomes a test of whether the book moved underneath it.
drift = pop.bad_rate_drift(dev_rate, oot_rate)
if pop.drift_is_material(dev_rate, oot_rate):
    warnings.append(
        f"dev bad rate {dev_rate:.2%} vs OOT {oot_rate:.2%} — {drift:.0%} relative "
        "difference, above the {:.0%} tolerance. Investigate before treating OOT "
        "results as a clean read on model stability.".format(pop.MAX_BAD_RATE_DRIFT)
    )

print(f"DEV  {n_dev:>8,} accounts   {n_dev_bad:>6,} bads   {dev_rate:.2%}")
print(f"OOT  {n_oot:>8,} accounts   {n_oot_bad:>6,} bads   {oot_rate:.2%}")
print(f"APP  {n_app:>8,} customers  {n_app_bad:>6,} bads   {n_app_bad / n_app:.2%}")
print(f"\nrelative bad-rate drift dev->oot: {drift:.1%}")

for w in warnings:
    print(f"\n[warn] {w}")

if problems:
    raise RuntimeError("Population gate failed:\n  - " + "\n  - ".join(problems))

print("\nPopulation gate passed.")
print("Next: notebooks/04_features_internal.py")
