#!/usr/bin/env python3
"""Tests for the population and target definition (plans.md Phase 3).

Run with:  python tests/test_population.py

The label definition is the single most consequential decision in the project —
everything downstream inherits it, and it cannot be fixed later without redoing
the feature engineering. So it is pinned here rather than left implicit in a
Spark expression.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from credit_risk import config as cfg
from credit_risk import population as pop

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


def test_classification() -> None:
    section("Target classification")

    check(pop.classify(0) == pop.GOOD, "0 DPD is good")
    check(pop.classify(29) == pop.GOOD, "29 DPD is still good")
    check(pop.classify(30) == pop.INDETERMINATE, "30 DPD is indeterminate")
    check(pop.classify(89) == pop.INDETERMINATE, "89 DPD is indeterminate")
    check(pop.classify(90) == pop.BAD, "90 DPD is bad")
    check(pop.classify(3260) == pop.BAD, "the observed maximum (3,260) is bad")

    # A covered window with no delinquency record is evidence of good behaviour,
    # not an absence of evidence. Treating it as unknown would silently drop
    # well-behaved accounts from training.
    check(pop.classify(None) == pop.GOOD, "no delinquency record in a covered window is good")

    check(pop.is_bad(90) and not pop.is_bad(89), "is_bad matches the threshold exactly")
    check(
        pop.is_indeterminate(45) and not pop.is_indeterminate(90),
        "is_indeterminate covers the band and stops at bad",
    )

    # The three classes must partition the space -- no value unclassified, none
    # in two classes.
    for dpd in range(0, 200):
        classes = [pop.classify(dpd)]
        check_partition = len(classes) == 1 and classes[0] in (pop.GOOD, pop.INDETERMINATE, pop.BAD)
        if not check_partition:
            check(False, f"DPD {dpd} classified as {classes}")
            return
    check(True, "every DPD value 0-199 falls in exactly one class")

    # Boundaries come from config, not from literals repeated here.
    check(
        pop.classify(cfg.BAD_DPD_THRESHOLD) == pop.BAD
        and pop.classify(cfg.BAD_DPD_THRESHOLD - 1) != pop.BAD,
        "the bad boundary tracks config.BAD_DPD_THRESHOLD",
    )
    check(
        pop.classify(cfg.INDETERMINATE_DPD_LOW) == pop.INDETERMINATE
        and pop.classify(cfg.INDETERMINATE_DPD_LOW - 1) == pop.GOOD,
        "the indeterminate boundary tracks config.INDETERMINATE_DPD_LOW",
    )


def test_exclusion_steps() -> None:
    section("Exclusion waterfall definition")

    keys = [s.key for s in pop.EXCLUSION_STEPS]
    check(len(keys) == len(set(keys)), "step keys are unique")
    check(all(s.kind in (pop.AVAILABILITY, pop.POLICY) for s in pop.EXCLUSION_STEPS),
          "every step is classified as availability or policy")
    check(all(s.rationale for s in pop.EXCLUSION_STEPS),
          "every step records why it exists — the waterfall is a deliverable, not a diagnostic")

    # Availability comes first so the policy counts that follow are expressed
    # over a population we can actually measure.
    kinds = [s.kind for s in pop.EXCLUSION_STEPS]
    check(kinds[0] == pop.AVAILABILITY, "the waterfall opens with an availability step")

    check("not_already_bad" in keys, "already-bad accounts are excluded")
    check("full_obs_window" in keys, "a full observation window is required")
    check("min_perf_coverage" in keys, "performance coverage is required")
    check(pop.EXCLUSION_STEP_KEYS == tuple(keys), "the exported key tuple matches the steps")

    check(
        "Completed" in pop.CLOSED_STATUSES and "Canceled" in pop.CLOSED_STATUSES,
        "closed statuses cover the observed contract states",
    )


def test_gate_thresholds() -> None:
    section("Population gate")

    # The check that would have caught SK_DPD_DEF in hour three.
    check(not pop.has_enough_bads(14), "14 bads (the SK_DPD_DEF outcome) fails the gate")
    check(not pop.has_enough_bads(870), "870 bads (card only) fails the gate")
    check(pop.has_enough_bads(1_812), "1,812 bads (pooled card+POS) passes the gate")
    check(
        pop.has_enough_bads(pop.MIN_BADS_FOR_SCORECARD),
        "the threshold itself passes (inclusive boundary)",
    )

    # Bad count caps scorecard length however many features were engineered.
    check(pop.supportable_characteristics(1_812) == 10, "1,812 bads supports ~10 characteristics")
    check(pop.supportable_characteristics(0) == 0, "zero bads supports nothing")
    check(pop.supportable_characteristics(100, bads_per_bin=0) == 0, "no division by zero")
    check(
        pop.supportable_characteristics(5_000) > pop.supportable_characteristics(1_812),
        "more bads supports more characteristics",
    )

    check(pop.scorecard_length_warning(1_812) is None, "1,812 bads meets the target length")
    check(pop.scorecard_length_warning(900) is not None, "a thin cohort warns about scorecard length")


def test_drift() -> None:
    section("Dev vs OOT comparability")

    # Measured: dev 2.28%, OOT 1.86%.
    drift = pop.bad_rate_drift(0.0228, 0.0186)
    check(0.17 < drift < 0.20, f"measured dev/OOT drift is ~18% (got {drift:.1%})")
    check(not pop.drift_is_material(0.0228, 0.0186), "the measured drift is within tolerance")

    check(pop.bad_rate_drift(0.02, 0.02) == 0.0, "identical rates have zero drift")
    check(pop.drift_is_material(0.02, 0.05), "a 150% swing is material")
    check(pop.drift_is_material(0.05, 0.02), "drift is symmetric in direction")
    check(pop.bad_rate_drift(0.0, 0.02) == 0.0, "a zero dev rate does not divide by zero")


def test_loss_components() -> None:
    """Phase 12: what is modelled, what is assumed, and the caveat on each."""
    section("Loss components (LGD / EAD)")

    check(cfg.WORKOUT_WINDOW_MONTHS > 0, "a workout window is defined")
    check(
        cfg.MIN_WORKOUT_COVERAGE_MONTHS >= cfg.WORKOUT_WINDOW_MONTHS,
        "eligibility requires at least as much panel as the window observes",
    )

    # The recovery distribution is what forces the two-stage design. If these
    # measured shares ever move, the model structure needs revisiting.
    check(
        cfg.EXPECTED_ZERO_RECOVERY_SHARE > 0.5,
        "measured zero-recovery mass justifies a separate stage-1 model",
    )
    check(
        cfg.EXPECTED_ZERO_RECOVERY_SHARE + cfg.EXPECTED_FULL_RECOVERY_SHARE > 0.9,
        "recovery is concentrated at the boundaries, so stage 2 must respect [0,1]",
    )
    check(cfg.RECOVERY_FLOOR == 0.0 and cfg.RECOVERY_CAP == 1.0, "recovery is clipped to [0,1]")

    # The textbook CCF was measured and found degenerate; the constant records
    # that so nobody silently reinstates it.
    check(
        cfg.EXPECTED_CCF_IN_BOUNDS_SHARE < 0.25,
        "the classical CCF is recorded as degenerate on this data",
    )
    check(cfg.EAD_RATIO_CAP > 1.0, "the EAD ratio cap allows for overlimit accounts")

    components = {(c.component, c.applies_to): c for c in cfg.LOSS_COMPONENTS}
    check(len(components) == len(cfg.LOSS_COMPONENTS), "one spec per component x product")
    for component in ("LGD", "EAD"):
        for product in ("card", "pos"):
            check((component, product) in components, f"{component} defined for {product}")

    check(
        components[("LGD", "card")].is_modelled and not components[("LGD", "pos")].is_modelled,
        "LGD is modelled on card and assumed on POS",
    )
    check(
        components[("EAD", "card")].is_modelled and not components[("EAD", "pos")].is_modelled,
        "EAD is modelled on card and assumed on POS",
    )

    # Same rule as ModelSpec: nothing ships without a stated caveat.
    check(
        all(c.limitation for c in cfg.LOSS_COMPONENTS),
        "every loss component records its limitation",
    )
    check(
        "not economic LGD" in components[("LGD", "card")].limitation,
        "the LGD proxy is explicitly not claimed as economic LGD",
    )


def test_config_alignment() -> None:
    section("Alignment with the model config")

    check(
        cfg.MODEL_BEHAVIOUR.expected_bads >= pop.MIN_BADS_FOR_SCORECARD,
        "the configured behaviour model clears its own gate",
    )
    check(
        cfg.MODEL_APPLICATION.expected_bads >= pop.MIN_BADS_FOR_SCORECARD,
        "the configured application model clears the gate",
    )
    check(
        pop.TARGET_BEHAVIOUR_TABLE != pop.TARGET_APPLICATION_TABLE,
        "the two models write to separate target tables",
    )


# --------------------------------------------------------------------------- #


def main() -> int:
    test_classification()
    test_exclusion_steps()
    test_gate_thresholds()
    test_drift()
    test_loss_components()
    test_config_alignment()

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
