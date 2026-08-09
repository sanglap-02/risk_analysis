#!/usr/bin/env python3
"""Step 02 — is this dataset modellable, and how big is the population?

    python lc/steps/02_explore.py

Runs before any feature work, deliberately. The most expensive failure in credit
risk is discovering in week five that the target was never modellable; on the
main project this exact check caught a bad definition that would have yielded 14
bads out of 49,796 accounts.

Four questions, in order of how badly a wrong answer would hurt:

  1. What does `loan_status` actually contain, and how many loans are unmatured?
  2. How many bads survive once unmatured loans are excluded?
  3. Does the vintage range support a genuine out-of-time split?
  4. Is there real recovery data for the LGD model?
"""

from __future__ import annotations

import sys
from pathlib import Path

LC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LC_ROOT / "src"))

import pandas as pd  # noqa: E402

from lending_club import config as cfg  # noqa: E402

PROBE_COLUMNS = [
    "id", "issue_d", "loan_status", "funded_amnt", "grade",
    "recoveries", "total_rec_prncp", "last_pymnt_d", "term",
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * 78}")


def main() -> int:
    if not cfg.RAW_FILE.exists():
        print(f"{cfg.RAW_FILE} not found. Run lc/steps/01_download.py first.", file=sys.stderr)
        return 1

    header = pd.read_csv(cfg.RAW_FILE, nrows=0)
    print(f"File     : {cfg.RAW_FILE.name} ({cfg.RAW_FILE.stat().st_size / 1024**2:,.0f} MB)")
    print(f"Columns  : {len(header.columns)}")

    print(f"\nReading {len(PROBE_COLUMNS)} columns over the full file ...")
    df = pd.read_csv(cfg.RAW_FILE, usecols=PROBE_COLUMNS, low_memory=False)
    df = df[df["id"].notna()]  # the file carries a few trailing junk rows
    print(f"Rows     : {len(df):,}")

    # ---------------------------------------------------------------- statuses
    rule("1. loan_status — what outcomes exist")
    counts = df["loan_status"].value_counts(dropna=False)
    for status, n in counts.items():
        if status in cfg.BAD_STATUSES:
            tag = "BAD"
        elif status in cfg.GOOD_STATUSES:
            tag = "GOOD"
        elif status in cfg.INDETERMINATE_STATUSES:
            tag = "indeterminate"
        elif status in cfg.EXCLUDED_STATUSES:
            tag = "EXCLUDED (unmatured)"
        else:
            tag = "unmapped"
        print(f"  {str(status):<52} {n:>9,}  {n / len(df):>6.1%}  {tag}")

    unmapped = set(counts.index.dropna()) - set(
        cfg.BAD_STATUSES + cfg.GOOD_STATUSES + cfg.INDETERMINATE_STATUSES + cfg.EXCLUDED_STATUSES
    )
    if unmapped:
        print(f"\n  UNMAPPED statuses need a decision: {sorted(unmapped)}")

    n_current = df["loan_status"].isin(cfg.EXCLUDED_STATUSES).sum()
    print(
        f"\n  {n_current:,} loans ({n_current / len(df):.1%}) are still in repayment.\n"
        "  The reference project labels every one of these GOOD. They have not\n"
        "  survived anything yet -- we exclude them instead."
    )

    # ---------------------------------------------------------------- maturity
    rule("2. Modellable population once unmatured loans are excluded")
    matured = df[df["loan_status"].isin(cfg.BAD_STATUSES + cfg.GOOD_STATUSES)].copy()
    matured["target"] = matured["loan_status"].isin(cfg.BAD_STATUSES).astype(int)
    n_bad = int(matured["target"].sum())
    print(f"  matured loans : {len(matured):>9,}")
    print(f"  bads          : {n_bad:>9,}  ({n_bad / len(matured):.2%})")
    print(f"  goods         : {len(matured) - n_bad:>9,}")
    verdict = "PASS" if n_bad >= cfg.MIN_BADS_FOR_SCORECARD else "FAIL"
    print(f"\n  gate (>= {cfg.MIN_BADS_FOR_SCORECARD:,} bads): {verdict}")

    # ---------------------------------------------------------------- vintages
    rule("3. Vintages — can we split out of time?")
    matured["issue_dt"] = pd.to_datetime(matured["issue_d"], format="%b-%Y", errors="coerce")
    by_year = matured.groupby(matured["issue_dt"].dt.year).agg(
        n=("target", "size"), bads=("target", "sum"), bad_rate=("target", "mean")
    )
    by_year["bad_rate"] = (by_year["bad_rate"] * 100).round(2)
    print(by_year.to_string())

    dev = matured[
        (matured["issue_dt"] >= cfg.DEV_ISSUE_FROM) & (matured["issue_dt"] < cfg.OOT_ISSUE_FROM)
    ]
    oot = matured[
        (matured["issue_dt"] >= cfg.OOT_ISSUE_FROM) & (matured["issue_dt"] < cfg.OOT_ISSUE_TO)
    ]
    print(
        f"\n  DEV  {cfg.DEV_ISSUE_FROM[:7]} to {cfg.OOT_ISSUE_FROM[:7]}   "
        f"{len(dev):>8,} loans  {int(dev['target'].sum()):>7,} bads  {dev['target'].mean():.2%}"
    )
    print(
        f"  OOT  {cfg.OOT_ISSUE_FROM[:7]} to {cfg.OOT_ISSUE_TO[:7]}   "
        f"{len(oot):>8,} loans  {int(oot['target'].sum()):>7,} bads  "
        f"{oot['target'].mean():.2%}" if len(oot) else "  OOT  EMPTY"
    )
    if len(dev) and len(oot):
        drift = abs(dev["target"].mean() - oot["target"].mean()) / dev["target"].mean()
        print(f"  relative bad-rate drift dev->oot: {drift:.1%}")

    # ---------------------------------------------------------------- recovery
    rule("4. Recovery data — can LGD be modelled rather than assumed?")
    bad = matured[matured["target"] == 1].copy()
    bad["recovery_rate"] = (bad["recoveries"] / bad["funded_amnt"]).clip(0, 1)
    print(f"  charged-off loans        : {len(bad):>9,}")
    print(f"  with recoveries recorded : {int((bad['recoveries'] > 0).sum()):>9,} "
          f"({(bad['recoveries'] > 0).mean():.1%})")
    print(f"  zero recovery            : {(bad['recovery_rate'] <= 0).mean():>9.1%}")
    print(f"  full recovery (=1)       : {(bad['recovery_rate'] >= 0.999).mean():>9.1%}")
    print(f"\n  recovery_rate distribution:")
    print(bad["recovery_rate"].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(4).to_string())
    print(
        "\n  Heavy mass at zero is expected and is precisely why LGD is fitted in\n"
        "  two stages: P(any recovery), then E[recovery | recovery > 0]."
    )

    print(f"\n{'=' * 78}")
    if n_bad < cfg.MIN_BADS_FOR_SCORECARD:
        print("Gate FAILED — revisit the target before building features.")
        return 1
    print("Population gate passed. Next: python lc/steps/03_clean_and_target.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
