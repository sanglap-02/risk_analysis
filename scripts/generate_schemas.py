#!/usr/bin/env python3
"""Infer, override and commit the bronze schemas; profile the raw files.

Usage
-----
    python scripts/generate_schemas.py
    python scripts/generate_schemas.py --tables bureau bureau_balance

What it does
------------
For every source CSV it streams the *entire* file through pyarrow's CSV reader
(block by block, so memory stays bounded regardless of the 27M-row
bureau_balance), then:

  * captures pyarrow's inferred type per column
  * applies the override rules in `credit_risk.schemas`
  * writes `schemas/<table>.json` in Spark StructType JSON format
  * writes `data/manifests/raw_profile.json` with row counts, null counts,
    file sizes and SHA-256 digests

Streaming the whole file is deliberate. It is simultaneously the type
inference, a full-file parse validation, and the authoritative row count --
three things we would otherwise pay for separately. If a later block contains
a value that contradicts the type inferred from the first block, pyarrow
raises; we catch that, widen the offending column to string, and retry. Every
such widening is recorded in the manifest rather than being silently absorbed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402

from credit_risk.config import (  # noqa: E402
    MANIFEST_DIR,
    RAW_DIR,
    SCHEMA_DIR,
    TABLE_SPECS,
    TableSpec,
)
from credit_risk.schemas import build_struct_schema, write_schema  # noqa: E402

BLOCK_SIZE = 64 << 20  # 64 MB read blocks
MAX_WIDEN_RETRIES = 12
PROFILE_PATH = MANIFEST_DIR / "raw_profile.json"

# pyarrow reports conflicts as e.g.
#   "In CSV column #4: CSV conversion error to int64: invalid value 'XNA'"
# and, when column names are available,
#   "Could not parse 'XNA' as int64 ... column 'FOO'"
_COLUMN_INDEX_RE = re.compile(r"column #(\d+)", re.IGNORECASE)
_COLUMN_NAME_RE = re.compile(r"column ['\"]([^'\"]+)['\"]", re.IGNORECASE)


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_header(path: Path) -> list[str]:
    """Read just the column names, without parsing the body.

    Uses csv.reader rather than a naive split so quoted headers containing
    commas survive intact.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return next(csv.reader(fh))


def _offending_column(message: str, columns: list[str]) -> str | None:
    """Extract the column pyarrow choked on, by name or by positional index."""
    if match := _COLUMN_NAME_RE.search(message):
        name = match.group(1)
        if name in columns:
            return name
    if match := _COLUMN_INDEX_RE.search(message):
        index = int(match.group(1))
        if 0 <= index < len(columns):
            return columns[index]
    return None


def scan_file(path: Path) -> dict[str, Any]:
    """Stream the whole CSV. Returns inferred types, row count, null counts.

    Columns that fail conversion partway through the file are widened to string
    and the file is re-scanned. Each widening is reported back to the caller.
    """
    columns = read_header(path)
    forced_string: dict[str, pa.DataType] = {}
    widened: list[dict[str, str]] = []

    for attempt in range(MAX_WIDEN_RETRIES + 1):
        convert = pacsv.ConvertOptions(column_types=dict(forced_string))
        read = pacsv.ReadOptions(block_size=BLOCK_SIZE)
        try:
            with pacsv.open_csv(path, read_options=read, convert_options=convert) as reader:
                schema = reader.schema
                n_rows = 0
                null_counts = [0] * len(schema)
                for batch in reader:
                    n_rows += batch.num_rows
                    for i in range(batch.num_columns):
                        null_counts[i] += batch.column(i).null_count
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            message = str(exc)
            column = _offending_column(message, columns)
            if column is None or column in forced_string or attempt == MAX_WIDEN_RETRIES:
                raise RuntimeError(
                    f"Could not resolve a schema for {path.name}: {message}"
                ) from exc
            forced_string[column] = pa.string()
            widened.append({"column": column, "reason": message.strip()[:200]})
            print(f"    widening '{column}' to string and re-scanning ({message.strip()[:90]}...)")
            continue

        return {
            "columns": {name: str(dtype) for name, dtype in zip(schema.names, schema.types)},
            "n_rows": n_rows,
            "null_counts": dict(zip(schema.names, null_counts)),
            "widened_columns": widened,
        }

    raise RuntimeError(f"Exceeded widening retries for {path.name}")


def profile_table(spec: TableSpec, raw_dir: Path) -> dict[str, Any] | None:
    path = raw_dir / spec.filename
    if not path.exists():
        print(f"  {spec.name:<26} SKIPPED (file not found: {spec.filename})")
        return None

    size_mb = path.stat().st_size / 1024**2
    print(f"  {spec.name:<26} scanning {spec.filename} ({size_mb:,.1f} MB) ...")

    scan = scan_file(path)
    schema_json = build_struct_schema(scan["columns"])
    write_schema(spec.name, schema_json)

    n_rows = scan["n_rows"]
    delta = ""
    if spec.expected_rows is not None:
        diff = n_rows - spec.expected_rows
        delta = " (matches reference)" if diff == 0 else f" (reference {spec.expected_rows:,}, diff {diff:+,})"
    print(f"  {spec.name:<26} {n_rows:,} rows, {len(scan['columns'])} columns{delta}")

    overridden = [
        f["name"]
        for f in schema_json["fields"]
        if f["metadata"]["type_source"] != "inferred"
        and f["metadata"]["inferred_type"] != f["type"]
    ]

    return {
        "table": spec.name,
        "filename": spec.filename,
        "role": spec.role,
        "grain": spec.grain,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "n_rows": n_rows,
        "n_columns": len(scan["columns"]),
        "expected_rows": spec.expected_rows,
        "row_count_matches_reference": (
            None if spec.expected_rows is None else n_rows == spec.expected_rows
        ),
        "natural_key": list(spec.natural_key),
        "arrow_types": scan["columns"],
        "spark_types": {f["name"]: f["type"] for f in schema_json["fields"]},
        "null_counts": scan["null_counts"],
        "null_rate": {
            col: round(count / n_rows, 6) if n_rows else 0.0
            for col, count in scan["null_counts"].items()
        },
        "retyped_columns": overridden,
        "widened_columns": scan["widened_columns"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--tables", nargs="*", help="subset of table names to process")
    args = parser.parse_args()

    specs = TABLE_SPECS
    if args.tables:
        wanted = set(args.tables)
        specs = tuple(s for s in TABLE_SPECS if s.name in wanted)
        if unknown := wanted - {s.name for s in TABLE_SPECS}:
            print(f"Unknown table(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Profiling {len(specs)} table(s) from {args.raw_dir}\n")
    profiles = [p for spec in specs if (p := profile_table(spec, args.raw_dir))]

    if not profiles:
        print(
            "\nNothing profiled -- no source files found. "
            "Run `python scripts/download_data.py` first.",
            file=sys.stderr,
        )
        return 1

    # Preserve profiles for tables not processed in this run.
    existing: dict[str, Any] = {}
    if PROFILE_PATH.exists():
        existing = json.loads(PROFILE_PATH.read_text(encoding="utf-8")).get("tables", {})
    existing.update({p["table"]: p for p in profiles})

    PROFILE_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "raw_dir": str(args.raw_dir),
                "tables": existing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSchemas written to {SCHEMA_DIR}")
    print(f"Profile written to {PROFILE_PATH}")

    mismatches = [p for p in profiles if p["row_count_matches_reference"] is False]
    if mismatches:
        print("\nRow counts differ from the published reference:", file=sys.stderr)
        for p in mismatches:
            print(
                f"  {p['table']}: got {p['n_rows']:,}, reference {p['expected_rows']:,}",
                file=sys.stderr,
            )
        print(
            "\nIf the download is complete and the files parse, the reference "
            "constants in config.TABLE_SPECS are the thing to update -- not the data.",
            file=sys.stderr,
        )

    widened = [p for p in profiles if p["widened_columns"]]
    if widened:
        print("\nColumns widened to string during scanning (inspect these):")
        for p in widened:
            for w in p["widened_columns"]:
                print(f"  {p['table']}.{w['column']}")

    print("\nNext: python scripts/verify_raw.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
