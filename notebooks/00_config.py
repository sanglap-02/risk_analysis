# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configuration & Environment Bootstrap
# MAGIC
# MAGIC Run this first in any session. It puts `src/` on the path, creates the Unity
# MAGIC Catalog objects, and re-exports every constant the later notebooks need so
# MAGIC they never define their own.
# MAGIC
# MAGIC The single source of truth is `src/credit_risk/config.py`. If a number
# MAGIC governs the model — the observation point, the bad definition, the PDO —
# MAGIC it lives there and nowhere else. Notebooks that redefine constants locally
# MAGIC are how a development scorecard and a production scorecard silently diverge.
# MAGIC
# MAGIC **Prerequisite:** this repo must be cloned as a Databricks Git folder so
# MAGIC `src/` and `schemas/` are on the workspace filesystem.

# COMMAND ----------

import os
import sys
from pathlib import Path

# Notebooks live in <repo>/notebooks, so the repo root is one level up.
REPO_ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
SRC_PATH = REPO_ROOT / "src"

if not SRC_PATH.exists():
    raise RuntimeError(
        f"Could not find {SRC_PATH}. This notebook expects to run from a Databricks "
        "Git folder containing the full repo (src/, schemas/, notebooks/). "
        f"Current working directory: {os.getcwd()}"
    )

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

print(f"repo root : {REPO_ROOT}")
print(f"src path  : {SRC_PATH}")

# COMMAND ----------

from credit_risk import config as cfg  # noqa: E402
from credit_risk import schemas as sch  # noqa: E402

CATALOG = cfg.CATALOG
RAW_VOLUME = cfg.RAW_VOLUME
SCHEMA_DIR = REPO_ROOT / "schemas"

print(f"catalog   : {CATALOG}")
print(f"volume    : {RAW_VOLUME}")
print(f"schemas   : {SCHEMA_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create catalog, schemas and the raw volume
# MAGIC
# MAGIC Four schemas, matching the medallion layout in `plans.md` §3.3. Bronze is
# MAGIC immutable and complete; silver is cleaned; gold is modelling-ready;
# MAGIC reporting holds the artefacts a reviewer reads.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")

for schema_name, comment in [
    (cfg.BRONZE_SCHEMA, "Raw ingested source data. Immutable, complete, never edited in place."),
    (cfg.SILVER_SCHEMA, "Cleaned and typed. Sentinels nulled, duplicates resolved, exclusions applied."),
    (cfg.GOLD_SCHEMA, "Modelling-ready: feature tables, target table, analytical base table, scored population."),
    (cfg.REPORTING_SCHEMA, "Validation metrics, PSI monitoring, portfolio simulation, swap-set analysis."),
]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema_name} COMMENT '{comment}'")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{cfg.BRONZE_SCHEMA}.raw_files")

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model constants
# MAGIC
# MAGIC Printed rather than hidden so every notebook run leaves an audit trail of
# MAGIC exactly which time framework and bad definition produced its output.

# COMMAND ----------

print("Time framework (plans.md 1.3, 2.2)")
print("-" * 72)
for cohort in cfg.COHORTS:
    obs_lo, obs_hi = cohort.obs_window
    perf_lo, perf_hi = cohort.perf_window
    print(
        f"  {cohort.name:<4} obs point MONTHS_BALANCE={cohort.obs_point:>4}  "
        f"features [{obs_lo:>4}, {obs_hi:>4}]  label [{perf_lo:>4}, {perf_hi:>4}]"
    )

print("\nTarget definition (plans.md 1.4)")
print("-" * 72)
print(f"  bad            : max({cfg.DPD_COLUMN}) >= {cfg.BAD_DPD_THRESHOLD} in the performance window")
print(f"  indeterminate  : {cfg.INDETERMINATE_DPD_LOW} <= max({cfg.DPD_COLUMN}) < {cfg.BAD_DPD_THRESHOLD}  (excluded from training)")
print(f"  good           : max({cfg.DPD_COLUMN}) < {cfg.INDETERMINATE_DPD_LOW}")

print("\nScore scaling (plans.md 8.3)")
print("-" * 72)
print(f"  PDO {cfg.PDO}, base score {cfg.BASE_SCORE} at {cfg.BASE_ODDS}:1 good:bad odds")

print("\nValidation acceptance (plans.md Phase 10)")
print("-" * 72)
print(f"  KS >= {cfg.MIN_KS}   Gini >= {cfg.MIN_GINI}   train-OOT Gini gap <= {cfg.MAX_TRAIN_OOT_GINI_GAP}")
print(f"  PSI < {cfg.PSI_STABLE} stable, >= {cfg.PSI_UNUSABLE} unusable")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source table registry

# COMMAND ----------

display(
    spark.createDataFrame(
        [
            (s.name, s.filename, s.grain, s.role, s.expected_rows, ", ".join(s.natural_key) or "-")
            for s in cfg.TABLE_SPECS
        ],
        "table string, source_file string, grain string, role string, expected_rows long, natural_key string",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload the raw CSVs
# MAGIC
# MAGIC Run locally first:
# MAGIC
# MAGIC ```bash
# MAGIC python scripts/download_data.py
# MAGIC python scripts/generate_schemas.py
# MAGIC python scripts/verify_raw.py
# MAGIC ```
# MAGIC
# MAGIC Then push the extract to the volume. The CLI is the only sane option at
# MAGIC this size — the workspace file uploader caps out well below the 375 MB
# MAGIC `bureau_balance.csv`:
# MAGIC
# MAGIC ```bash
# MAGIC databricks fs cp --recursive data/raw dbfs:/Volumes/credit_risk/bronze/raw_files/
# MAGIC ```
# MAGIC
# MAGIC Verify what landed before moving on.

# COMMAND ----------

try:
    files = dbutils.fs.ls(RAW_VOLUME)
    if not files:
        print(f"{RAW_VOLUME} is empty — upload the extract before running 01_ingest_bronze.")
    else:
        total = sum(f.size for f in files)
        for f in sorted(files, key=lambda x: -x.size):
            print(f"  {f.name:<46}{f.size / 1024**2:>10,.1f} MB")
        print(f"  {'total':<46}{total / 1024**3:>10,.2f} GB")
except Exception as exc:  # noqa: BLE001
    print(f"Could not list {RAW_VOLUME}: {exc}")
