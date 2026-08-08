# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingestion Layer → Bronze
# MAGIC
# MAGIC Lands all eight Home Credit source files as Delta tables in
# MAGIC `credit_risk.bronze`, **unmodified**.
# MAGIC
# MAGIC Bronze has exactly one job: be a complete, faithful, reproducible copy of
# MAGIC the source. No cleaning, no filtering, no type narrowing, no dropping of
# MAGIC columns we think we won't need. Every one of those is a decision, and
# MAGIC decisions belong in silver where they can be reviewed. The 47 building-
# MAGIC characteristic columns in `application_train` are useless for credit risk
# MAGIC and they still land here intact.
# MAGIC
# MAGIC ### What this notebook guarantees
# MAGIC
# MAGIC | Guarantee | How |
# MAGIC |---|---|
# MAGIC | No schema inference | Reads the committed `schemas/*.json` produced by `scripts/generate_schemas.py` |
# MAGIC | No silent parse loss | `PERMISSIVE` mode with a `_corrupt_record` column that is counted, not ignored |
# MAGIC | Reproducible | Schema fingerprint recorded per run; drift is detectable |
# MAGIC | Auditable | Row counts checked against reference; every run appends to `reporting.ingestion_audit` |
# MAGIC | Traceable | `_ingested_at`, `_source_file`, `_row_hash` on every row |
# MAGIC | Fast to join | `OPTIMIZE ... ZORDER BY` on the join keys |
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC ```bash
# MAGIC python scripts/download_data.py      # fetch + extract
# MAGIC python scripts/generate_schemas.py   # commit schemas/*.json
# MAGIC python scripts/verify_raw.py         # data-quality gate
# MAGIC databricks fs cp --recursive data/raw dbfs:/Volumes/credit_risk/bronze/raw_files/
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json  # noqa: E402
import time  # noqa: E402
from hashlib import sha256  # noqa: E402

from pyspark.sql import DataFrame, functions as F  # noqa: E402
from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402

# COMMAND ----------

dbutils.widgets.dropdown("mode", "overwrite", ["overwrite", "skip_existing"], "Write mode")
dbutils.widgets.text("tables", "", "Tables (comma-separated, blank = all)")
dbutils.widgets.dropdown("run_optimize", "true", ["true", "false"], "Run OPTIMIZE ZORDER")

WRITE_MODE = dbutils.widgets.get("mode")
RUN_OPTIMIZE = dbutils.widgets.get("run_optimize") == "true"
_requested = [t.strip() for t in dbutils.widgets.get("tables").split(",") if t.strip()]

SPECS = [s for s in cfg.TABLE_SPECS if not _requested or s.name in _requested]
if _requested and (unknown := set(_requested) - {s.name for s in cfg.TABLE_SPECS}):
    raise ValueError(f"Unknown table(s): {sorted(unknown)}")

print(f"mode     : {WRITE_MODE}")
print(f"optimize : {RUN_OPTIMIZE}")
print(f"tables   : {', '.join(s.name for s in SPECS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read options
# MAGIC
# MAGIC Defaults apply to the seven data files. `columns_description` needs its own
# MAGIC treatment: it is Latin-1 encoded and its Description field contains quoted
# MAGIC commas and embedded newlines, so it must be read multi-line. Reading it with
# MAGIC the defaults yields a table that looks fine and is quietly shredded.

# COMMAND ----------

BASE_READ_OPTIONS = {
    "header": "true",
    "mode": "PERMISSIVE",
    "columnNameOfCorruptRecord": sch.CORRUPT_RECORD_COL,
    "nullValue": "",
    "quote": '"',
    "escape": '"',
    "multiLine": "false",
    "encoding": "UTF-8",
}

READ_OPTION_OVERRIDES = {
    "columns_description": {"encoding": "ISO-8859-1", "multiLine": "true"},
}


def read_options_for(table: str) -> dict[str, str]:
    return BASE_READ_OPTIONS | READ_OPTION_OVERRIDES.get(table, {})


# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion

# COMMAND ----------


def load_schema(table: str) -> tuple[StructType, str]:
    """Load the committed schema and return it plus a fingerprint of it.

    The fingerprint is stored with each run so a schema change between runs is
    visible in the audit table rather than being something you discover three
    notebooks later.
    """
    schema_json = sch.read_schema(table, SCHEMA_DIR)
    fingerprint = sha256(
        json.dumps(
            [(f["name"], f["type"]) for f in schema_json["fields"]], sort_keys=True
        ).encode()
    ).hexdigest()[:16]

    struct = StructType.fromJson(schema_json)
    # PERMISSIVE mode requires the corrupt-record column to exist in the schema.
    struct = struct.add(StructField(sch.CORRUPT_RECORD_COL, StringType(), True))
    return struct, fingerprint


def add_metadata(df: DataFrame, data_columns: list[str]) -> DataFrame:
    """Attach lineage columns.

    `_row_hash` coalesces nulls to a sentinel before hashing. Without that,
    concat_ws silently skips nulls and (a, NULL, b) hashes identically to
    (a, b, NULL) — which would make the duplicate detection in silver quietly
    wrong on exactly the rows most likely to be duplicated.
    """
    hashed = F.sha2(
        F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("\\N")) for c in data_columns]),
        256,
    )
    return (
        df.withColumn(sch.INGEST_TIMESTAMP_COL, F.current_timestamp())
        .withColumn(sch.SOURCE_FILE_COL, F.col("_metadata.file_name"))
        .withColumn(sch.ROW_HASH_COL, hashed)
    )


def ingest(spec) -> dict:
    """Ingest one source file to a bronze Delta table. Returns an audit record."""
    target = cfg.bronze(spec.name)
    source = f"{RAW_VOLUME}/{spec.filename}"
    started = time.time()

    if WRITE_MODE == "skip_existing" and spark.catalog.tableExists(target):
        n_rows = spark.table(target).count()
        print(f"  {spec.name:<26} exists with {n_rows:,} rows — skipped")
        return {
            "table": spec.name, "target": target, "source_file": spec.filename,
            "status": "skipped", "n_rows": n_rows, "n_corrupt": None,
            "expected_rows": spec.expected_rows, "row_count_ok": None,
            "schema_fingerprint": None, "n_columns": None, "duration_sec": 0.0,
        }

    struct, fingerprint = load_schema(spec.name)
    data_columns = [f.name for f in struct.fields if f.name not in sch.METADATA_COLUMNS]

    df = (
        spark.read.format("csv")
        .schema(struct)
        .options(**read_options_for(spec.name))
        .load(source)
    )

    (
        add_metadata(df, data_columns)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target)
    )

    # Count from Delta, not from the DataFrame. Spark refuses to evaluate a
    # filter on _corrupt_record against a lazily-parsed CSV source; once the
    # data is materialised in Delta the column is an ordinary column.
    written = spark.table(target)
    n_rows = written.count()
    n_corrupt = written.filter(F.col(sch.CORRUPT_RECORD_COL).isNotNull()).count()

    comment = f"{spec.description} | grain: {spec.grain}".replace("'", "''")
    spark.sql(f"COMMENT ON TABLE {target} IS '{comment}'")

    row_count_ok = None if spec.expected_rows is None else n_rows == spec.expected_rows
    flag = "" if row_count_ok is not False else f"  <-- expected {spec.expected_rows:,}"
    corrupt_flag = "" if n_corrupt == 0 else f", {n_corrupt:,} CORRUPT"
    print(f"  {spec.name:<26} {n_rows:>12,} rows, {len(data_columns):>3} cols{corrupt_flag}{flag}")

    return {
        "table": spec.name, "target": target, "source_file": spec.filename,
        "status": "ingested", "n_rows": n_rows, "n_corrupt": n_corrupt,
        "expected_rows": spec.expected_rows, "row_count_ok": row_count_ok,
        "schema_fingerprint": fingerprint, "n_columns": len(data_columns),
        "duration_sec": round(time.time() - started, 1),
    }


# COMMAND ----------

print(f"Ingesting {len(SPECS)} table(s) from {RAW_VOLUME}\n")
audit = [ingest(spec) for spec in SPECS]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compaction and clustering
# MAGIC
# MAGIC `ZORDER` on the join keys. The panels are joined on `SK_ID_CURR` /
# MAGIC `SK_ID_PREV` in every feature notebook, and `bureau_balance` is joined on
# MAGIC `SK_ID_BUREAU` — note that it carries no `SK_ID_CURR` at all, so clustering
# MAGIC it by customer is impossible and reaching a customer means routing through
# MAGIC `bureau`.
# MAGIC
# MAGIC Deliberately not partitioning. Databricks guidance is not to partition
# MAGIC tables below roughly 1 TB; on a 27M-row table, partitioning by month bucket
# MAGIC would create small files and slow the joins it is meant to help.

# COMMAND ----------

if RUN_OPTIMIZE:
    for spec in SPECS:
        if not spec.zorder_by:
            continue
        target = cfg.bronze(spec.name)
        keys = ", ".join(spec.zorder_by)
        try:
            started = time.time()
            spark.sql(f"OPTIMIZE {target} ZORDER BY ({keys})")
            print(f"  {spec.name:<26} zordered by ({keys}) in {time.time() - started:,.0f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  {spec.name:<26} OPTIMIZE failed, non-fatal: {str(exc)[:120]}")
else:
    print("OPTIMIZE skipped (run_optimize=false)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Audit
# MAGIC
# MAGIC Appended, never overwritten. The history of ingestion runs is itself an
# MAGIC artefact — when a downstream number moves, the first question is whether
# MAGIC the input moved.

# COMMAND ----------

AUDIT_FIELDS = [
    ("table", "string"), ("target", "string"), ("source_file", "string"),
    ("status", "string"), ("n_rows", "long"), ("n_corrupt", "long"),
    ("expected_rows", "long"), ("row_count_ok", "boolean"),
    ("schema_fingerprint", "string"), ("n_columns", "int"), ("duration_sec", "double"),
]
audit_schema = ", ".join(f"{name} {dtype}" for name, dtype in AUDIT_FIELDS)

# Build rows positionally rather than handing Spark a list of dicts -- dict
# ordering against a string schema is version-dependent and silently shuffles
# columns when it disagrees.
audit_rows = [tuple(record[name] for name, _ in AUDIT_FIELDS) for record in audit]

audit_df = spark.createDataFrame(audit_rows, audit_schema).withColumn(
    "run_at", F.current_timestamp()
)

(
    audit_df.write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(cfg.reporting("ingestion_audit"))
)

display(audit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate
# MAGIC
# MAGIC Fails the notebook rather than letting a bad load flow into silver. A
# MAGIC truncated `bureau_balance` does not announce itself downstream; it just
# MAGIC produces a slightly optimistic bad rate and a scorecard nobody can
# MAGIC reproduce.

# COMMAND ----------

problems = []

for record in audit:
    if record["status"] == "skipped":
        continue
    if record["row_count_ok"] is False:
        problems.append(
            f"{record['table']}: {record['n_rows']:,} rows, expected {record['expected_rows']:,}"
        )
    if record["n_corrupt"]:
        problems.append(f"{record['table']}: {record['n_corrupt']:,} unparseable rows")

if problems:
    raise RuntimeError(
        "Bronze ingestion gate failed:\n  - " + "\n  - ".join(problems)
        + "\n\nInspect the _corrupt_record column on the affected tables, or re-run "
        "scripts/verify_raw.py locally. Do not proceed to silver."
    )

total_rows = sum(r["n_rows"] for r in audit)
print(f"Bronze ingestion complete: {len(audit)} tables, {total_rows:,} rows.")
print("Next: notebooks/02_clean_silver.py")

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT table_name, comment
        FROM {CATALOG}.information_schema.tables
        WHERE table_schema = '{cfg.BRONZE_SCHEMA}'
        ORDER BY table_name
    """)
)
