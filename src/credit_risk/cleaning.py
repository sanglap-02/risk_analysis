"""Cleaning rules for the bronze -> silver transformation (plans.md Phase 2).

Deliberately free of PySpark imports. Everything here is a pure decision —
which columns get scrubbed, which get an indicator, how a bureau status maps to
a delinquency bucket, what the pooled panel looks like. The Spark transforms
that apply these rules live in `notebooks/02_clean_silver.py`.

Keeping the two apart means every cleaning *decision* is unit-testable on a
laptop with no cluster, which matters because these decisions are the ones a
model validator will question.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Bureau status decoding
# --------------------------------------------------------------------------- #
# bureau_balance.STATUS is a single character per tradeline-month:
#
#   C  closed
#   X  status unknown
#   0  no days past due
#   1  1-30 DPD
#   2  31-60 DPD
#   3  61-90 DPD
#   4  91-120 DPD
#   5  120+ DPD, or written off
#
# We map to the BUCKET NUMBER, not to a day count. Converting '3' into "61 days"
# or "90 days" would fabricate precision the source does not contain, and every
# feature built on it would inherit that invention. Phase 4 counts buckets.
#
# 'X' becomes NULL rather than 0: unknown is not the same as current, and
# collapsing the two would understate risk on exactly the accounts where the
# bureau has the least visibility.

BUREAU_STATUS_MAP: dict[str, int | None] = {
    "C": 0,  # closed with no delinquency outstanding
    "X": None,  # unknown -- must not be read as "fine"
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

# A tradeline-month counts as delinquent from bucket 1 upward.
BUREAU_DELINQUENT_FROM_BUCKET = 1

# Buckets 3+ correspond to 61 DPD and worse; 4+ is the 90+ equivalent.
BUREAU_BUCKET_90_PLUS = 4


def decode_bureau_status(status: str | None) -> int | None:
    """Map a bureau_balance STATUS character to its bucket number."""
    if status is None:
        return None
    return BUREAU_STATUS_MAP.get(status.strip().upper())


# --------------------------------------------------------------------------- #
# Sentinel handling
# --------------------------------------------------------------------------- #
# Home Credit encodes "missing" in DAYS_* columns as 365243 (~1000 years).
# Measured across the extract: 1,570,735 occurrences.
#
# Every occurrence is nulled. Separately, columns where the sentinel is common
# enough to be a signal in its own right get an explicit boolean indicator --
# because for those, "no value" is not an accident of data collection, it is a
# fact about the customer. DAYS_EMPLOYED at 18% sentinel means "not employed";
# DAYS_FIRST_DRAWING at 56% means the facility was never drawn on.

SENTINEL_INDICATOR_MIN_RATE = 0.05

# Columns whose sentinel carries a specific business meaning, so the indicator
# gets a name that says what it means rather than a mechanical suffix.
SENTINEL_INDICATOR_NAMES: dict[str, str] = {
    "DAYS_EMPLOYED": "is_not_employed",
    "DAYS_FIRST_DRAWING": "never_drawn",
}

SENTINEL_INDICATOR_SUFFIX = "_is_missing"


def indicator_name(column: str) -> str:
    """Name for the boolean indicator accompanying a scrubbed sentinel column."""
    return SENTINEL_INDICATOR_NAMES.get(column, f"{column.lower()}{SENTINEL_INDICATOR_SUFFIX}")


def sentinel_columns_from_report(
    report: dict[str, Any], *, table: str | None = None
) -> dict[str, dict[str, int]]:
    """Extract per-table sentinel counts from `raw_verification.json`.

    Driving the cleaning off the committed verification report means the rules
    follow the measurement rather than a hand-maintained list that silently goes
    stale when the source data changes.
    """
    found: dict[str, dict[str, int]] = {}
    for finding in report.get("findings", []):
        if finding.get("check") != "days-sentinel":
            continue
        columns = finding.get("columns")
        if not columns:
            continue
        target = finding["target"]
        if table is None or target == table:
            found[target] = dict(columns)
    return found


def needs_indicator(sentinel_count: int, n_rows: int, threshold: float = SENTINEL_INDICATOR_MIN_RATE) -> bool:
    """Is this column's sentinel common enough to warrant its own indicator?"""
    if n_rows <= 0:
        return False
    return sentinel_count / n_rows >= threshold


# --------------------------------------------------------------------------- #
# Type narrowing
# --------------------------------------------------------------------------- #
# Bronze widened these to double so nothing could fail on read. Once the
# sentinel is nulled they are safe to narrow. AMT_* deliberately stays double --
# those are genuinely decimal.

NARROW_TO_INT_PREFIXES: tuple[str, ...] = ("DAYS_",)


def should_narrow_to_int(column: str) -> bool:
    return column.startswith(NARROW_TO_INT_PREFIXES)


# --------------------------------------------------------------------------- #
# Outlier capping
# --------------------------------------------------------------------------- #
# AMT_INCOME_TOTAL contains a 117,000,000 value. Capping rather than dropping,
# with a flag, so the row survives and the fact of capping stays visible.

CAP_COLUMNS: dict[str, float] = {"AMT_INCOME_TOTAL": 0.995}
CAP_FLAG_SUFFIX = "_was_capped"


def cap_flag_name(column: str) -> str:
    return f"{column.lower()}{CAP_FLAG_SUFFIX}"


# --------------------------------------------------------------------------- #
# The pooled behaviour panel
# --------------------------------------------------------------------------- #
# Card alone yields 870 bads, below the ~1,000 needed for a stable scorecard.
# Pooling card and POS gives 1,812 (plans.md 1.4b).
#
# The two products do not share a column set. Rather than force them into a
# lowest-common-denominator schema and throw away the card behaviour that most
# of the requested attributes depend on, both column sets are carried and the
# absent ones are NULL. That is not a compromise -- WOE gives "missing" its own
# bin with its own weight, and PRODUCT_TYPE enters the model as a characteristic
# in its own right.

PRODUCT_CARD = "CC"
PRODUCT_POS = "POS"

PANEL_KEY_COLUMNS: tuple[str, ...] = ("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE")

PANEL_SHARED_COLUMNS: tuple[str, ...] = ("NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF")

# Present only on credit_card_balance.
PANEL_CARD_ONLY_COLUMNS: tuple[str, ...] = (
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE",  # sic -- the typo is in the source data
    "AMT_TOTAL_RECEIVABLE",
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT",
    "CNT_DRAWINGS_POS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
)

# Present only on POS_CASH_balance.
PANEL_POS_ONLY_COLUMNS: tuple[str, ...] = ("CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE")

# Row-level derivations. These are cleaning, not feature engineering: each one
# encodes a decision about invalid source values that every downstream notebook
# would otherwise have to re-implement, and re-implement identically.
PANEL_DERIVED_COLUMNS: tuple[str, ...] = (
    "product_type",
    "has_valid_limit",
    "utilisation",
    "is_overlimit",
    "overlimit_amount",
)


def panel_output_columns() -> list[str]:
    return [
        *PANEL_KEY_COLUMNS,
        *PANEL_DERIVED_COLUMNS,
        *PANEL_SHARED_COLUMNS,
        *PANEL_CARD_ONLY_COLUMNS,
        *PANEL_POS_ONLY_COLUMNS,
    ]


# --------------------------------------------------------------------------- #
# Panel index alignment
# --------------------------------------------------------------------------- #
# Measured: bureau_balance spans [-96, 0] while both internal panels span
# [-96, -1]. Bureau windows that reuse the internal window arithmetic without
# accounting for that extra month are silently off by one.

BUREAU_PANEL_MAX_MONTH = 0
INTERNAL_PANEL_MAX_MONTH = -1
BUREAU_MONTH_OFFSET = BUREAU_PANEL_MAX_MONTH - INTERNAL_PANEL_MAX_MONTH  # = 1


def align_bureau_month(months_balance: int) -> int:
    """Shift a bureau month onto the internal panel's convention.

    bureau_balance's month 0 is contemporaneous with the internal panels'
    month -1, so bureau months are shifted back by one to make the two
    directly comparable.
    """
    return months_balance - BUREAU_MONTH_OFFSET


# --------------------------------------------------------------------------- #
# Silver table names
# --------------------------------------------------------------------------- #

SILVER_TABLES: tuple[str, ...] = (
    "application",
    "application_holdout",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "panel_pooled",
)
