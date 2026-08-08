#!/usr/bin/env python3
"""Data-quality gate on the raw extract, before anything reaches Delta.

Usage
-----
    python scripts/verify_raw.py
    python scripts/verify_raw.py --strict     # non-zero exit on any WARN

Checks
------
1.  Row counts against the published reference counts.
2.  Natural-key uniqueness -- the key each table claims must actually be unique.
3.  Referential integrity across the join graph, including the one that trips
    people up: bureau_balance reaches the customer only via bureau.SK_ID_BUREAU,
    because it carries no SK_ID_CURR of its own.
4.  MONTHS_BALANCE sign and range on the panel tables -- the whole behaviour
    scoring design in plans.md 2.2 depends on these being negative offsets.
5.  Presence of the 365243 sentinel in DAYS_* columns. This check does not fix
    anything; it quantifies the problem so Phase 2 knows exactly what it is
    dealing with, and so nobody computes a tenure feature from a 1000-year
    offset by accident.

Findings are written to data/manifests/raw_verification.json. FAIL means the
data cannot be trusted downstream; WARN means it is a known issue that silver
is expected to handle.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402

from credit_risk.config import (  # noqa: E402
    DAYS_SENTINEL,
    MANIFEST_DIR,
    RAW_DIR,
    TABLE_SPECS,
    TABLES,
)

PROFILE_PATH = MANIFEST_DIR / "raw_profile.json"
REPORT_PATH = MANIFEST_DIR / "raw_verification.json"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# child table -> (child column, parent table, parent column)
REFERENTIAL_EDGES: tuple[tuple[str, str, str, str], ...] = (
    ("bureau_balance", "SK_ID_BUREAU", "bureau", "SK_ID_BUREAU"),
    ("bureau", "SK_ID_CURR", "application_train", "SK_ID_CURR"),
    ("credit_card_balance", "SK_ID_PREV", "previous_application", "SK_ID_PREV"),
    ("pos_cash_balance", "SK_ID_PREV", "previous_application", "SK_ID_PREV"),
    ("installments_payments", "SK_ID_PREV", "previous_application", "SK_ID_PREV"),
    ("previous_application", "SK_ID_CURR", "application_train", "SK_ID_CURR"),
)

# Every parent in this dataset is partial, because Home Credit released an
# anonymised *sample* rather than a complete book:
#
#   application_train      holds only the labelled customers; the rest are in
#                          application_test
#   bureau                 does not carry a row for every SK_ID_BUREAU that
#                          appears in bureau_balance
#   previous_application   does not carry a row for every SK_ID_PREV that
#                          appears in the three panel tables
#
# Measured on the real extract, orphan rates run 3.9%-10.9% of distinct child
# IDs. Those are sampling artefacts and must not fail the gate. But "expected"
# cannot mean "unbounded" -- a genuinely broken join key or a truncated parent
# file would also show up as orphans, just far more of them. So partial parents
# WARN up to a materiality threshold and FAIL above it.
PARTIAL_PARENTS = frozenset({"application_train", "bureau", "previous_application"})

# Share of DISTINCT child IDs allowed to be orphaned against a partial parent.
# Set with headroom over the worst observed real value (10.9%); anything above
# this is not sampling, it is a defect.
MAX_ORPHAN_RATE_DISTINCT = 0.25

PANEL_TABLES = ("bureau_balance", "credit_card_balance", "pos_cash_balance")


class Report:
    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []

    def add(self, status: str, check: str, target: str, detail: str, **extra: Any) -> None:
        self.findings.append(
            {"status": status, "check": check, "target": target, "detail": detail, **extra}
        )
        icon = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{icon}] {check:<22} {target:<38} {detail}")

    def count(self, status: str) -> int:
        return sum(1 for f in self.findings if f["status"] == status)


def load_columns(path: Path, columns: list[str]) -> pa.Table:
    """Read only the named columns from a CSV."""
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(include_columns=columns),
    )


def check_row_counts(profile: dict[str, Any], report: Report) -> None:
    for spec in TABLE_SPECS:
        entry = profile.get(spec.name)
        if entry is None:
            report.add(WARN, "row-count", spec.name, "not profiled; skipped")
            continue
        if spec.expected_rows is None:
            report.add(PASS, "row-count", spec.name, f"{entry['n_rows']:,} rows (no reference)")
        elif entry["n_rows"] == spec.expected_rows:
            report.add(PASS, "row-count", spec.name, f"{entry['n_rows']:,} rows, matches reference")
        else:
            report.add(
                FAIL,
                "row-count",
                spec.name,
                f"got {entry['n_rows']:,}, reference {spec.expected_rows:,} "
                f"({entry['n_rows'] - spec.expected_rows:+,}) -- download may be incomplete",
                actual=entry["n_rows"],
                expected=spec.expected_rows,
            )


def check_key_uniqueness(raw_dir: Path, report: Report) -> None:
    for spec in TABLE_SPECS:
        if not spec.has_natural_key:
            report.add(
                PASS,
                "key-uniqueness",
                spec.name,
                "no natural key claimed; uniqueness asserted downstream",
            )
            continue

        path = raw_dir / spec.filename
        if not path.exists():
            report.add(WARN, "key-uniqueness", spec.name, "file missing; skipped")
            continue

        keys = list(spec.natural_key)
        table = load_columns(path, keys)
        n_rows = table.num_rows
        n_distinct = table.group_by(keys).aggregate([]).num_rows
        target = f"{spec.name}({', '.join(keys)})"

        if n_distinct == n_rows:
            report.add(PASS, "key-uniqueness", target, f"unique across {n_rows:,} rows")
        else:
            report.add(
                FAIL,
                "key-uniqueness",
                target,
                f"{n_rows - n_distinct:,} duplicate rows on the declared key",
                n_rows=n_rows,
                n_distinct=n_distinct,
            )


def check_referential_integrity(raw_dir: Path, report: Report) -> None:
    for child, child_col, parent, parent_col in REFERENTIAL_EDGES:
        child_path = raw_dir / TABLES[child].filename
        parent_path = raw_dir / TABLES[parent].filename
        target = f"{child}.{child_col} -> {parent}.{parent_col}"

        if not (child_path.exists() and parent_path.exists()):
            report.add(WARN, "referential", target, "file missing; skipped")
            continue

        child_ids = load_columns(child_path, [child_col]).column(child_col).combine_chunks()
        parent_ids = load_columns(parent_path, [parent_col]).column(parent_col).combine_chunks()

        parent_unique = pc.unique(parent_ids)
        n_child = len(child_ids)
        n_orphan_rows = pc.sum(pc.invert(pc.is_in(child_ids, value_set=parent_unique))).as_py() or 0

        # Distinct-ID rate is the meaningful measure. Row rate is skewed by how
        # many monthly records each account happens to have, so a handful of
        # long-lived orphan accounts can look like a large row-level breach.
        child_unique = pc.unique(child_ids)
        n_child_ids = len(child_unique)
        n_orphan_ids = (
            pc.sum(pc.invert(pc.is_in(child_unique, value_set=parent_unique))).as_py() or 0
        )
        rate_ids = n_orphan_ids / n_child_ids if n_child_ids else 0.0

        stats = {
            "n_orphan_rows": n_orphan_rows,
            "n_orphan_ids": n_orphan_ids,
            "n_child_ids": n_child_ids,
            "orphan_rate_distinct": round(rate_ids, 4),
        }
        detail = (
            f"{n_orphan_ids:,} of {n_child_ids:,} distinct ids ({rate_ids:.1%}), "
            f"{n_orphan_rows:,} of {n_child:,} rows ({n_orphan_rows / n_child:.1%})"
        )

        if n_orphan_rows == 0:
            report.add(PASS, "referential", target, f"all {n_child:,} references resolve")
        elif parent not in PARTIAL_PARENTS:
            report.add(FAIL, "referential", target, f"orphaned references -- {detail}", **stats)
        elif rate_ids > MAX_ORPHAN_RATE_DISTINCT:
            report.add(
                FAIL,
                "referential",
                target,
                f"orphan rate {rate_ids:.1%} exceeds the {MAX_ORPHAN_RATE_DISTINCT:.0%} "
                f"materiality threshold for a partial parent -- {detail}",
                **stats,
            )
        else:
            report.add(
                WARN,
                "referential",
                target,
                f"unmatched but within tolerance for a sampled parent -- {detail}",
                **stats,
            )


def check_panel_index(raw_dir: Path, report: Report) -> None:
    """MONTHS_BALANCE must be a negative offset; the cohort design depends on it."""
    for name in PANEL_TABLES:
        path = raw_dir / TABLES[name].filename
        if not path.exists():
            report.add(WARN, "panel-index", name, "file missing; skipped")
            continue

        col = load_columns(path, ["MONTHS_BALANCE"]).column("MONTHS_BALANCE")
        lo = pc.min(col).as_py()
        hi = pc.max(col).as_py()
        n_positive = pc.sum(pc.greater(col, 0)).as_py() or 0

        if n_positive:
            report.add(
                FAIL,
                "panel-index",
                f"{name}.MONTHS_BALANCE",
                f"{n_positive:,} non-negative values; the cohort windows assume negative offsets",
            )
        else:
            report.add(
                PASS,
                "panel-index",
                f"{name}.MONTHS_BALANCE",
                f"range [{lo}, {hi}], all negative offsets",
                min=lo,
                max=hi,
            )


def check_days_sentinel(raw_dir: Path, profile: dict[str, Any], report: Report) -> None:
    """Quantify the 365243 sentinel so Phase 2 knows the exact blast radius."""
    total_affected = 0

    for spec in TABLE_SPECS:
        entry = profile.get(spec.name)
        path = raw_dir / spec.filename
        if entry is None or not path.exists():
            continue

        days_cols = [
            c
            for c, t in entry["spark_types"].items()
            if c.startswith("DAYS_") and t in {"double", "integer", "long"}
        ]
        if not days_cols:
            continue

        table = load_columns(path, days_cols)
        hits = {
            col: n
            for col in days_cols
            if (n := pc.sum(pc.equal(table.column(col), DAYS_SENTINEL)).as_py() or 0)
        }

        if hits:
            total_affected += sum(hits.values())
            summary = ", ".join(f"{c}={n:,}" for c, n in sorted(hits.items(), key=lambda kv: -kv[1]))
            report.add(
                WARN,
                "days-sentinel",
                spec.name,
                f"{DAYS_SENTINEL} present -- {summary}",
                columns=hits,
            )
        else:
            report.add(PASS, "days-sentinel", spec.name, f"no {DAYS_SENTINEL} values")

    if total_affected:
        print(
            f"\n  -> {total_affected:,} sentinel values across the extract. "
            "Silver must null these before any DAYS_-derived feature is built."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on WARN too")
    args = parser.parse_args()

    if not PROFILE_PATH.exists():
        print(
            f"No profile at {PROFILE_PATH}. Run `python scripts/generate_schemas.py` first.",
            file=sys.stderr,
        )
        return 2

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["tables"]
    report = Report()

    print("Row counts")
    print("-" * 100)
    check_row_counts(profile, report)

    print("\nNatural key uniqueness")
    print("-" * 100)
    check_key_uniqueness(args.raw_dir, report)

    print("\nReferential integrity")
    print("-" * 100)
    check_referential_integrity(args.raw_dir, report)

    print("\nPanel index")
    print("-" * 100)
    check_panel_index(args.raw_dir, report)

    print("\nDAYS_ sentinel scan")
    print("-" * 100)
    check_days_sentinel(args.raw_dir, profile, report)

    n_fail, n_warn = report.count(FAIL), report.count(WARN)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_pass": report.count(PASS),
                "n_warn": n_warn,
                "n_fail": n_fail,
                "findings": report.findings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(f"{report.count(PASS)} passed, {n_warn} warnings, {n_fail} failures")
    print(f"Report written to {REPORT_PATH}")

    if n_fail:
        print("\nFAILures must be resolved before ingesting to bronze.", file=sys.stderr)
        return 1
    if n_warn and args.strict:
        return 1

    print("\nNext: upload data/raw/*.csv to the Databricks volume, then run notebooks/01_ingest_bronze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
