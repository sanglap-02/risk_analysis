#!/usr/bin/env python3
"""Tests for the feature catalogue and window arithmetic (plans.md Phase 4).

Run with:  python tests/test_features.py

The window arithmetic is the anti-leakage guarantee. Leakage does not announce
itself — it makes results look better, not worse — so the guarantee is asserted
here rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from credit_risk import config as cfg
from credit_risk import features as fx

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


def test_window_arithmetic() -> None:
    section("Window arithmetic")

    check(fx.window_for(-13, 12) == (-24, -13), "dev 12m window is [-24, -13]")
    check(fx.window_for(-13, 6) == (-18, -13), "dev 6m window is [-18, -13]")
    check(fx.window_for(-13, 3) == (-15, -13), "dev 3m window is [-15, -13]")
    check(fx.window_for(-19, 12) == (-30, -19), "oot 12m window is [-30, -19]")
    check(fx.window_for(-13, 1) == (-13, -13), "a 1-month window is the observation month itself")

    # The window length must be exactly what was asked for.
    for months in fx.WINDOWS:
        lo, hi = fx.window_for(-13, months)
        check(hi - lo + 1 == months, f"{months}m window spans exactly {months} months")

    try:
        fx.window_for(-13, 0)
        check(False, "a zero-length window is rejected")
    except ValueError:
        check(True, "a zero-length window is rejected")


def test_no_window_reaches_the_future() -> None:
    """The single property the whole design rests on."""
    section("Anti-leakage: no window may pass the observation point")

    for cohort in cfg.COHORTS:
        for months in (*fx.WINDOWS, fx.TREND_WINDOW, 1, 24):
            lo, hi = fx.window_for(cohort.obs_point, months)
            check(
                hi <= cohort.obs_point,
                f"{cohort.name} {months}m window ends at or before the observation point",
            )
            check(
                fx.window_is_safe(cohort.obs_point, months),
                f"{cohort.name} {months}m window passes the safety check",
            )

    # A feature window must never overlap its own cohort's label window.
    for cohort in cfg.COHORTS:
        perf_lo, _ = cohort.perf_window
        for months in fx.WINDOWS:
            _, hi = fx.window_for(cohort.obs_point, months)
            check(
                hi < perf_lo,
                f"{cohort.name} {months}m feature window does not touch the label window",
            )

    # The trend window must sit inside the observation window, or slope features
    # would draw on months the cohort has not confirmed are present.
    for cohort in cfg.COHORTS:
        obs_lo, obs_hi = cohort.obs_window
        lo, hi = fx.window_for(cohort.obs_point, fx.TREND_WINDOW)
        check(
            lo >= obs_lo and hi <= obs_hi,
            f"{cohort.name} trend window sits inside the confirmed observation window",
        )


def test_catalogue_integrity() -> None:
    section("Feature catalogue")

    names = fx.feature_names()
    check(len(names) == len(set(names)), "feature names are unique after window expansion")
    check(len(names) > 50, f"catalogue is substantive ({len(names)} features)")

    check(
        all(s.direction in fx.DIRECTIONS for s in fx.INTERNAL_FEATURES),
        "every feature declares a valid expected direction",
    )
    check(
        all(s.description for s in fx.INTERNAL_FEATURES),
        "every feature carries a description for the data dictionary",
    )
    check(
        all(s.family in fx.INTERNAL_FAMILIES for s in fx.INTERNAL_FEATURES),
        "every feature belongs to a declared family",
    )
    check(
        not any(s.windowed for s in fx.INTERNAL_FEATURES),
        "expansion leaves no unexpanded windowed specs",
    )

    try:
        fx.FeatureSpec("bad", fx.TENURE, "d", "up")
        check(False, "an invalid direction is rejected at construction")
    except ValueError:
        check(True, "an invalid direction is rejected at construction")


def test_expansion() -> None:
    section("Window expansion")

    spec = fx.FeatureSpec("max_dpd", fx.DELINQUENCY, "Worst DPD", fx.RISKIER, windowed=True)
    expanded = fx.expand((spec,))
    check(len(expanded) == len(fx.WINDOWS), "one concrete feature per window")
    check(
        {s.name for s in expanded} == {f"max_dpd_{m}m" for m in fx.WINDOWS},
        "expanded names carry the window suffix",
    )
    check(
        all(s.direction == fx.RISKIER for s in expanded),
        "expansion preserves the expected direction",
    )
    check(all(not s.windowed for s in expanded), "expanded specs are no longer marked windowed")

    unwindowed = fx.FeatureSpec("util_at_obs", fx.UTILISATION, "d", fx.RISKIER, card_only=True)
    result = fx.expand((unwindowed,))
    check(len(result) == 1 and result[0].card_only, "non-windowed specs pass through with flags intact")


def test_domain_expectations() -> None:
    """The directions encode real credit risk priors. Pin the important ones."""
    section("Expected directions encode credit risk priors")

    by_name = {s.name: s for s in fx.INTERNAL_FEATURES}

    # Utilisation rising is the classic distress signal.
    check(by_name["util_trend_slope_6m"].direction == fx.RISKIER, "rising utilisation means more risk")
    check(by_name["util_std_12m"].direction == fx.RISKIER, "volatile utilisation means more risk")

    # Cash advance usage is a textbook early warning.
    check(
        by_name["cash_adv_to_spend_ratio_12m"].direction == fx.RISKIER,
        "cash advance share means more risk",
    )

    # Recency is protective; more months since an event is safer.
    check(
        by_name["months_since_last_delinq"].direction == fx.SAFER,
        "longer since the last delinquency means less risk",
    )
    check(
        by_name["n_consecutive_clean_months"].direction == fx.SAFER,
        "a longer clean streak means less risk",
    )

    # Counts of bad events go the other way.
    check(by_name["n_overlimit_months_12m"].direction == fx.RISKIER, "overlimit months mean more risk")
    check(by_name["ontime_pay_ratio_12m"].direction == fx.SAFER, "paying on time means less risk")

    # Balance level genuinely has no clean prior -- a high balance can mean a
    # heavy user or a distressed one. Claiming a direction here would be false
    # confidence.
    check(by_name["bal_avg_12m"].direction == fx.UNSURE, "balance level carries no confident prior")


def test_family_coverage() -> None:
    section("Family coverage and caps")

    grouped = fx.by_family()
    check(set(grouped) == set(fx.INTERNAL_FAMILIES), "every declared family has features")
    for family, specs in sorted(grouped.items()):
        check(len(specs) >= 3, f"{family} has enough candidates to select from ({len(specs)})")

    # Utilisation variants correlate at 0.95+. Without a cap the scorecard
    # becomes seven flavours of one signal -- strong in development, fragile in
    # production.
    check(fx.MAX_FEATURES_PER_FAMILY <= 3, "family cap keeps one signal from dominating")
    check(
        fx.MAX_FEATURES_PER_FAMILY * len(fx.INTERNAL_FAMILIES) >= cfg.TARGET_N_FEATURES[1],
        "the family caps still allow a full-length scorecard",
    )

    # The attributes the brief specifically named must be represented.
    names = set(fx.feature_names())
    for family, probe in [
        ("cash advance", "cash_adv_amt_12m"),
        ("utilisation trend", "util_trend_slope_6m"),
        ("overlimit", "n_overlimit_months_12m"),
        ("payment history", "ontime_pay_ratio_12m"),
        ("delinquency recency", "months_since_last_delinq"),
        ("cross-product depth", "cust_n_product_types"),
        ("account tenure", "acct_tenure_months"),
    ]:
        check(probe in names, f"requested attribute covered: {family}")


# --------------------------------------------------------------------------- #


def main() -> int:
    test_window_arithmetic()
    test_no_window_reaches_the_future()
    test_catalogue_integrity()
    test_expansion()
    test_domain_expectations()
    test_family_coverage()

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
