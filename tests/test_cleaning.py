#!/usr/bin/env python3
"""Tests for the bronze -> silver cleaning decisions (plans.md Phase 2).

Run with:  python tests/test_cleaning.py

These cover the *decisions*, not the Spark plumbing — which bureau status means
what, which sentinels earn an indicator, how the two panel products reconcile.
Those are the things a model validator questions, so they are the things that
need to be pinned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from credit_risk import cleaning as cln
from credit_risk import config as cfg

FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 78}")


# --------------------------------------------------------------------------- #


def test_bureau_status_decoding() -> None:
    section("Bureau status decoding")

    check(cln.decode_bureau_status("0") == 0, "'0' (current) decodes to bucket 0")
    check(cln.decode_bureau_status("5") == 5, "'5' (120+/written off) decodes to bucket 5")
    check(cln.decode_bureau_status("C") == 0, "'C' (closed) decodes to bucket 0")

    # The one that matters most: unknown must not read as current. Collapsing
    # X into 0 understates risk on exactly the accounts the bureau can see least.
    check(cln.decode_bureau_status("X") is None, "'X' (unknown) decodes to NULL, never 0")

    check(cln.decode_bureau_status(None) is None, "NULL status stays NULL")
    check(cln.decode_bureau_status(" 3 ") == 3, "whitespace tolerated")
    check(cln.decode_bureau_status("c") == 0, "lower case tolerated")
    check(cln.decode_bureau_status("?") is None, "unrecognised status is NULL, not a guess")

    # Buckets must be ordered, since Phase 4 compares against thresholds.
    buckets = [cln.decode_bureau_status(s) for s in ("0", "1", "2", "3", "4", "5")]
    check(buckets == sorted(buckets), "buckets increase monotonically with severity")
    check(
        cln.BUREAU_BUCKET_90_PLUS > cln.BUREAU_DELINQUENT_FROM_BUCKET,
        "the 90+ threshold sits above the delinquency threshold",
    )


def test_sentinel_rules() -> None:
    section("Sentinel handling")

    check(cfg.DAYS_SENTINEL == 365243, "sentinel value pinned at 365243")

    # 56% and 18% sentinel respectively -- both far above the threshold.
    check(cln.needs_indicator(934_444, 1_670_214), "DAYS_FIRST_DRAWING (56%) earns an indicator")
    check(cln.needs_indicator(55_374, 307_511), "DAYS_EMPLOYED (18%) earns an indicator")
    check(not cln.needs_indicator(40_645, 1_670_214), "DAYS_FIRST_DUE (2.4%) does not")
    check(not cln.needs_indicator(0, 1_000), "a clean column never earns an indicator")
    check(not cln.needs_indicator(5, 0), "zero rows does not divide by zero")

    # Indicators for the two meaningful cases are named for what they mean, not
    # for the mechanical fact that a value was absent.
    check(cln.indicator_name("DAYS_EMPLOYED") == "is_not_employed", "DAYS_EMPLOYED -> is_not_employed")
    check(cln.indicator_name("DAYS_FIRST_DRAWING") == "never_drawn", "DAYS_FIRST_DRAWING -> never_drawn")
    check(
        cln.indicator_name("DAYS_TERMINATION") == "days_termination_is_missing",
        "unmapped columns fall back to a mechanical suffix",
    )


def test_sentinel_rules_track_the_committed_report() -> None:
    """The cleaning must follow the measurement, not a hand-maintained list."""
    section("Sentinel rules driven by the committed verification report")

    report_path = cfg.MANIFEST_DIR / "raw_verification.json"
    if not report_path.exists():
        check(True, "no verification report present; skipped")
        return

    report = json.loads(report_path.read_text())
    found = cln.sentinel_columns_from_report(report)

    check(bool(found), f"sentinel findings extracted for {len(found)} table(s)")
    check(
        "previous_application" in found and "DAYS_FIRST_DRAWING" in found["previous_application"],
        "previous_application.DAYS_FIRST_DRAWING picked up from the report",
    )
    check(
        "application_train" in found and "DAYS_EMPLOYED" in found["application_train"],
        "application_train.DAYS_EMPLOYED picked up from the report",
    )

    # Tables the gate found clean must not appear -- scrubbing them would be
    # wasted work and would imply a problem that does not exist.
    check("bureau" not in found, "clean tables are absent from the scrub list")

    total = sum(sum(cols.values()) for cols in found.values())
    check(total == 1_570_735, f"total sentinel count matches the measured 1,570,735 (got {total:,})")

    filtered = cln.sentinel_columns_from_report(report, table="application_train")
    check(set(filtered) == {"application_train"}, "single-table filter works")


def test_panel_pooling() -> None:
    section("Pooled panel schema")

    card_only = set(cln.PANEL_CARD_ONLY_COLUMNS)
    pos_only = set(cln.PANEL_POS_ONLY_COLUMNS)
    keys = set(cln.PANEL_KEY_COLUMNS)
    shared = set(cln.PANEL_SHARED_COLUMNS)

    check(not (card_only & pos_only), "card-only and POS-only column sets are disjoint")
    check(not (card_only & shared), "card-only columns are not also listed as shared")
    check(not (keys & card_only) and not (keys & pos_only), "keys are not duplicated in either product set")

    output = cln.panel_output_columns()
    check(len(output) == len(set(output)), "no duplicate columns in the pooled output")
    check(all(c in output for c in keys), "all key columns present in the output")
    check("product_type" in output, "product_type is carried as a characteristic")
    check("utilisation" in output, "utilisation derived once, in silver")

    # The attributes the brief specifically asked for must survive pooling.
    for attribute in (
        "AMT_DRAWINGS_ATM_CURRENT",  # cash advance
        "CNT_DRAWINGS_ATM_CURRENT",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "SK_DPD",
    ):
        check(attribute in output, f"{attribute} survives pooling")

    check(cfg.DPD_COLUMN in output, f"the target column {cfg.DPD_COLUMN} is present in the panel")

    # Both panel products must be registered source tables.
    check(
        set(cfg.BEHAVIOUR_PANELS) == {"credit_card_balance", "pos_cash_balance"},
        "the pooled products match the configured behaviour panels",
    )


def test_bureau_month_alignment() -> None:
    section("Bureau panel alignment")

    # Measured: bureau_balance spans [-96, 0]; internal panels span [-96, -1].
    check(cln.BUREAU_PANEL_MAX_MONTH == 0, "bureau panel reaches month 0")
    check(cln.INTERNAL_PANEL_MAX_MONTH == -1, "internal panels stop at month -1")
    check(cln.BUREAU_MONTH_OFFSET == 1, "the offset between them is one month")

    # Bureau month 0 is contemporaneous with internal month -1.
    check(cln.align_bureau_month(0) == -1, "bureau month 0 aligns to internal month -1")
    check(cln.align_bureau_month(-96) == -97, "the whole axis shifts, not just the edge")

    # After alignment a bureau window can be compared with an internal one
    # without an off-by-one. This is the property the whole shift exists for.
    obs_point = cfg.COHORT_DEV.obs_point
    check(
        cln.align_bureau_month(obs_point + cln.BUREAU_MONTH_OFFSET) == obs_point,
        "an aligned bureau month lands exactly on the observation point",
    )


def test_capping_rules() -> None:
    section("Outlier capping")

    check("AMT_INCOME_TOTAL" in cln.CAP_COLUMNS, "AMT_INCOME_TOTAL is capped")
    check(
        cln.CAP_COLUMNS["AMT_INCOME_TOTAL"] == cfg.INCOME_CAP_PERCENTILE,
        "the cap percentile comes from config, not a literal in the cleaning module",
    )
    check(
        cln.cap_flag_name("AMT_INCOME_TOTAL") == "amt_income_total_was_capped",
        "capping is flagged so the fact stays visible downstream",
    )
    check(all(0 < p < 1 for p in cln.CAP_COLUMNS.values()), "every cap percentile is a valid quantile")


def test_type_narrowing() -> None:
    section("Type narrowing")

    check(cln.should_narrow_to_int("DAYS_BIRTH"), "DAYS_ columns narrow to int in silver")
    # AMT_ values are genuinely decimal and must stay double.
    check(not cln.should_narrow_to_int("AMT_BALANCE"), "AMT_ columns stay double")
    check(not cln.should_narrow_to_int("SK_ID_CURR"), "keys are untouched by narrowing")


def test_silver_table_registry() -> None:
    section("Silver table registry")

    check(len(cln.SILVER_TABLES) == len(set(cln.SILVER_TABLES)), "silver table names are unique")
    check("panel_pooled" in cln.SILVER_TABLES, "the pooled panel is registered")
    check(
        "application_holdout" in cln.SILVER_TABLES,
        "the Kaggle scoring set stays a separate table so it cannot be trained on by accident",
    )


# --------------------------------------------------------------------------- #


def main() -> int:
    test_bureau_status_decoding()
    test_sentinel_rules()
    test_sentinel_rules_track_the_committed_report()
    test_panel_pooling()
    test_bureau_month_alignment()
    test_capping_rules()
    test_type_narrowing()
    test_silver_table_registry()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
