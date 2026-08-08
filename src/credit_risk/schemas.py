"""Schema handling for the bronze ingestion layer.

Design
------
Bronze must never fail to read and must never silently mistype a join key.
Those two goals pull in opposite directions, so we split them:

1.  `scripts/generate_schemas.py` infers a schema from the *full* downloaded
    CSV using pyarrow (C++ reader, full-file inference -- not a sample), then
    writes it to `schemas/<table>.json` in Spark's StructType JSON format.
    That JSON is a committed build artefact: the schema stops being something
    Spark guesses at read time and becomes something the repo asserts.

2.  The override rules below are applied on top of the inferred types. They
    force the types we refuse to leave to inference -- join keys, month
    indices, DPD counters -- and widen the numeric columns that carry nulls.

Bronze widens, silver narrows. A column read as double in bronze and cast to
int in silver is correct medallion practice; a join key read as double because
one row had a stray decimal is a defect that surfaces three phases later as
mysteriously missing customers.

Why we do NOT use inferSchema in Spark
--------------------------------------
`inferSchema=true` triggers a second full pass over the file. On the 27M-row
bureau_balance that is a meaningful cost, and it mistypes nullable integer
columns as double inconsistently between runs, which makes the bronze schema
non-deterministic. Reading from a committed schema file is both faster and
reproducible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from credit_risk.config import SCHEMA_DIR

# --------------------------------------------------------------------------- #
# Spark type names used in StructType JSON
# --------------------------------------------------------------------------- #

LONG = "long"
INTEGER = "integer"
DOUBLE = "double"
STRING = "string"
BOOLEAN = "boolean"

# --------------------------------------------------------------------------- #
# Exact-name overrides
# --------------------------------------------------------------------------- #
# Join keys and the panel index. These are never left to inference.

EXACT_OVERRIDES: dict[str, str] = {
    "SK_ID_CURR": LONG,
    "SK_ID_PREV": LONG,
    "SK_ID_BUREAU": LONG,
    "MONTHS_BALANCE": INTEGER,
    "SK_DPD": INTEGER,
    "SK_DPD_DEF": INTEGER,
    "TARGET": INTEGER,
    "NUM_INSTALMENT_NUMBER": INTEGER,
    "HOUR_APPR_PROCESS_START": INTEGER,
    "CNT_CHILDREN": INTEGER,
}

# --------------------------------------------------------------------------- #
# Prefix overrides
# --------------------------------------------------------------------------- #
# Applied only if the column is not in EXACT_OVERRIDES. All of these carry
# nulls somewhere in the dataset, so they are read as double in bronze and
# narrowed in silver once the nulls have been handled explicitly.
#
# Deliberately absent: FLAG_*. Home Credit uses that prefix for two different
# things -- FLAG_OWN_CAR / FLAG_OWN_REALTY are 'Y'/'N' strings while
# FLAG_DOCUMENT_2..21 and FLAG_MOBIL are 0/1 integers. A blanket rule would
# corrupt one group or the other, so FLAG_* is left to inference.

PREFIX_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("AMT_", DOUBLE),
    ("DAYS_", DOUBLE),  # sentinel 365243 is scrubbed in silver, not here
    ("CNT_", DOUBLE),
    ("NUM_INSTALMENT_", DOUBLE),
    ("EXT_SOURCE_", DOUBLE),
    ("RATE_", DOUBLE),
    ("OBS_", DOUBLE),
    ("DEF_", DOUBLE),
    ("APARTMENTS_", DOUBLE),
    ("BASEMENTAREA_", DOUBLE),
    ("YEARS_", DOUBLE),
    ("COMMONAREA_", DOUBLE),
    ("ELEVATORS_", DOUBLE),
    ("ENTRANCES_", DOUBLE),
    ("FLOORSMAX_", DOUBLE),
    ("FLOORSMIN_", DOUBLE),
    ("LANDAREA_", DOUBLE),
    ("LIVINGAPARTMENTS_", DOUBLE),
    ("LIVINGAREA_", DOUBLE),
    ("NONLIVINGAPARTMENTS_", DOUBLE),
    ("NONLIVINGAREA_", DOUBLE),
    ("TOTALAREA_", DOUBLE),
)

# --------------------------------------------------------------------------- #
# Bronze metadata columns appended to every table
# --------------------------------------------------------------------------- #

INGEST_TIMESTAMP_COL = "_ingested_at"
SOURCE_FILE_COL = "_source_file"
ROW_HASH_COL = "_row_hash"
CORRUPT_RECORD_COL = "_corrupt_record"

METADATA_COLUMNS: tuple[str, ...] = (
    INGEST_TIMESTAMP_COL,
    SOURCE_FILE_COL,
    ROW_HASH_COL,
    CORRUPT_RECORD_COL,
)

# --------------------------------------------------------------------------- #
# Arrow -> Spark type mapping
# --------------------------------------------------------------------------- #

_ARROW_PREFIX_TO_SPARK: tuple[tuple[str, str], ...] = (
    ("int8", INTEGER),
    ("int16", INTEGER),
    ("int32", INTEGER),
    ("int64", LONG),
    ("uint", LONG),
    ("float", DOUBLE),
    ("double", DOUBLE),
    ("decimal", DOUBLE),
    ("bool", BOOLEAN),
    ("date", "date"),
    ("timestamp", "timestamp"),
    ("string", STRING),
    ("large_string", STRING),
    ("null", STRING),  # an all-null column has no inferable type; string is safe
)


def arrow_type_to_spark(arrow_type: str) -> str:
    """Map a pyarrow type name onto a Spark StructType JSON type name."""
    name = str(arrow_type).lower()
    for prefix, spark_type in _ARROW_PREFIX_TO_SPARK:
        if name.startswith(prefix):
            return spark_type
    return STRING


# --------------------------------------------------------------------------- #
# Override resolution
# --------------------------------------------------------------------------- #


def resolve_type(column: str, inferred: str) -> tuple[str, str]:
    """Return the (final_type, reason) for one column.

    Precedence: exact override > prefix override > inferred.
    The reason string is recorded in the schema manifest so every type in the
    lakehouse can be traced back to why it is what it is.
    """
    if column in EXACT_OVERRIDES:
        return EXACT_OVERRIDES[column], "exact-override"

    for prefix, spark_type in PREFIX_OVERRIDES:
        if column.startswith(prefix):
            return spark_type, f"prefix-override:{prefix}"

    return inferred, "inferred"


# --------------------------------------------------------------------------- #
# Column name sanitisation
# --------------------------------------------------------------------------- #
# Unity Catalog rejects column names that are empty or contain spaces, periods,
# forward slashes or control characters. Parquet additionally dislikes a handful
# of others. Source CSVs do not care about any of this:
# HomeCredit_columns_description.csv ships an unnamed index column, whose header
# is the empty string -- which UC refuses with
#   "At columns.0: name "" is not a valid name"
#
# Sanitising here rather than at read time means the committed schema is already
# a valid target schema, so what the repo asserts and what lands in the
# lakehouse cannot disagree. The original header is preserved in field metadata
# so the mapping back to the source file is never lost.

_INVALID_COLUMN_CHARS = re.compile(r"[ ,;{}()\n\t=./\\\x00-\x1f]")


def sanitise_column_name(name: str, position: int) -> str:
    """Return a Unity-Catalog-safe column name.

    Empty or whitespace-only headers become `col_<position>`; illegal characters
    become underscores. Leading underscores are preserved -- they are meaningful
    for the bronze metadata columns.
    """
    cleaned = _INVALID_COLUMN_CHARS.sub("_", (name or "").strip())
    return cleaned if cleaned.strip("_") else f"col_{position}"


def build_struct_field(
    column: str, arrow_type: str, *, position: int = 0, nullable: bool = True
) -> dict[str, Any]:
    """Build one Spark StructType JSON field dict, with provenance in metadata.

    `arrow_type` is pyarrow's inferred type name; it is mapped to a Spark type
    first, then the override rules get the final say. The column name is
    sanitised for Unity Catalog, with the original header kept in metadata.
    """
    safe_name = sanitise_column_name(column, position)
    inferred = arrow_type_to_spark(arrow_type)
    # Overrides key off the SANITISED name, because that is the name the column
    # actually has in the lakehouse and the one every downstream notebook will
    # reference. A source header of "AMT BALANCE" becomes AMT_BALANCE and should
    # pick up the AMT_ rule; keying off the raw header would silently miss it.
    final_type, reason = resolve_type(safe_name, inferred)

    metadata: dict[str, Any] = {
        "arrow_type": arrow_type,
        "inferred_type": inferred,
        "type_source": reason,
    }
    if safe_name != column:
        metadata["source_column"] = column
        metadata["renamed_reason"] = (
            "empty header" if not (column or "").strip() else "illegal characters"
        )

    return {
        "name": safe_name,
        "type": final_type,
        "nullable": nullable,
        "metadata": metadata,
    }


def build_struct_schema(columns: dict[str, str]) -> dict[str, Any]:
    """Build a full Spark StructType JSON dict from {column: arrow_type_name}."""
    return {
        "type": "struct",
        "fields": [
            build_struct_field(col, typ, position=i)
            for i, (col, typ) in enumerate(columns.items())
        ],
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def schema_path(table: str, schema_dir: Path | str = SCHEMA_DIR) -> Path:
    return Path(schema_dir) / f"{table}.json"


def write_schema(
    table: str, schema: dict[str, Any], schema_dir: Path | str = SCHEMA_DIR
) -> Path:
    path = schema_path(table, schema_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def read_schema(table: str, schema_dir: Path | str = SCHEMA_DIR) -> dict[str, Any]:
    """Load a committed schema. Feed the result to `StructType.fromJson` in Spark."""
    path = schema_path(table, schema_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No committed schema for '{table}' at {path}. "
            "Run `python scripts/generate_schemas.py` after downloading the data."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def schema_columns(table: str, schema_dir: Path | str = SCHEMA_DIR) -> list[str]:
    return [f["name"] for f in read_schema(table, schema_dir)["fields"]]
