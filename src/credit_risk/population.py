"""Population and target definition rules (plans.md Phase 3).

Like `cleaning`, this module is deliberately free of PySpark. The label
definition, the exclusion policy and the gate thresholds are the decisions a
model validator interrogates, so they live somewhere they can be unit-tested on
a laptop rather than buried inside Spark expressions in a notebook.

`notebooks/03_population_and_target.py` applies them.
"""

from __future__ import annotations

from dataclasses import dataclass

from credit_risk.config import (
    BAD_DPD_THRESHOLD,
    INDETERMINATE_DPD_LOW,
    MAX_N_BINS,
    MIN_BIN_N_EVENT,
    TARGET_N_FEATURES,
)

# --------------------------------------------------------------------------- #
# The label
# --------------------------------------------------------------------------- #

BAD = "bad"
GOOD = "good"
INDETERMINATE = "indeterminate"


def classify(max_dpd: int | float | None) -> str:
    """Classify an account from its worst delinquency in the performance window.

    A missing max_dpd means the account was present but never delinquent, which
    is good — not unknown. Absence of a delinquency record in a window we have
    already confirmed is covered is evidence, not a gap.
    """
    dpd = 0 if max_dpd is None else max_dpd
    if dpd >= BAD_DPD_THRESHOLD:
        return BAD
    if dpd >= INDETERMINATE_DPD_LOW:
        return INDETERMINATE
    return GOOD


def is_bad(max_dpd: int | float | None) -> bool:
    return classify(max_dpd) == BAD


def is_indeterminate(max_dpd: int | float | None) -> bool:
    """Genuinely ambiguous accounts, excluded from training.

    A model learns the boundary between two classes. Pushing ambiguous cases
    into the good pile blurs that boundary and makes the model worse at
    separating the clear ones. They are kept and scored at validation: their
    mean score must land between goods and bads, or the bad definition is wrong.
    """
    return classify(max_dpd) == INDETERMINATE


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #
# Two kinds, and the distinction matters when explaining the population.
#
#   availability -- we cannot build features or observe an outcome. A limit of
#                   the data, not a statement about risk.
#   policy       -- the account is not a candidate for the decision this model
#                   drives.

AVAILABILITY = "availability"
POLICY = "policy"

# NAME_CONTRACT_STATUS values meaning the account is no longer open.
CLOSED_STATUSES: tuple[str, ...] = (
    "Completed",
    "Canceled",
    "Amortized debt",
    "Returned to the store",
)


@dataclass(frozen=True)
class ExclusionStep:
    key: str
    name: str
    kind: str
    rationale: str


# Order is meaningful: availability first, so the policy counts that follow are
# expressed over a population we can actually measure.
EXCLUSION_STEPS: tuple[ExclusionStep, ...] = (
    ExclusionStep(
        "has_history",
        "has panel history",
        AVAILABILITY,
        "No panel rows at all means no features and no outcome.",
    ),
    ExclusionStep(
        "min_tenure",
        "account open long enough at observation point",
        POLICY,
        "Very young accounts have too little behaviour to score; they route to "
        "an application score instead.",
    ),
    ExclusionStep(
        "full_obs_window",
        "full observation window present",
        AVAILABILITY,
        "Rolling 12-month features computed from partial windows quietly weaken "
        "exactly the trend and volatility signals that carry the most weight.",
    ),
    ExclusionStep(
        "min_perf_coverage",
        "sufficient performance coverage",
        AVAILABILITY,
        "Too few forward months to observe the outcome; the label would be an "
        "assumption rather than a measurement.",
    ),
    ExclusionStep(
        "not_closed",
        "not closed at observation point",
        POLICY,
        "A closed account is not a candidate for any line or authorisation "
        "decision.",
    ),
    ExclusionStep(
        "not_already_bad",
        "not already at the bad threshold at observation point",
        POLICY,
        "Nothing left to predict -- it has already happened. Leaving these in "
        "inflates every metric, and the model looks superb at identifying "
        "customers who already defaulted.",
    ),
    ExclusionStep(
        "not_dormant",
        "not dormant through the observation window",
        POLICY,
        "Zero balance and zero activity means no behaviour to score. Measurable "
        "on card accounts only; POS accounts are amortising loans and cannot go "
        "dormant while open.",
    ),
)

EXCLUSION_STEP_KEYS: tuple[str, ...] = tuple(s.key for s in EXCLUSION_STEPS)


# --------------------------------------------------------------------------- #
# Gate thresholds
# --------------------------------------------------------------------------- #
# The check that would have caught the SK_DPD_DEF mistake in hour three instead
# of week five. It is cheap and it is the single highest-value guard in the
# project.

MIN_BADS_FOR_SCORECARD = 1_000
MAX_BAD_RATE_DRIFT = 0.30


def supportable_characteristics(
    n_bads: int, *, bads_per_bin: int = MIN_BIN_N_EVENT, bins: int = MAX_N_BINS
) -> int:
    """How many characteristics this many bads can support.

    Every bin needs a minimum number of bads to give a stable WOE. With `bins`
    bins per characteristic, the bad count puts a hard ceiling on scorecard
    length regardless of how many candidate features were engineered.
    """
    if bads_per_bin <= 0 or bins <= 0:
        return 0
    return max(0, n_bads // (bads_per_bin * bins))


def has_enough_bads(n_bads: int) -> bool:
    return n_bads >= MIN_BADS_FOR_SCORECARD


def scorecard_length_warning(n_bads: int) -> str | None:
    """Warn when the bad count cannot support the intended scorecard length."""
    supportable = supportable_characteristics(n_bads)
    if supportable >= TARGET_N_FEATURES[0]:
        return None
    return (
        f"{n_bads:,} bads supports roughly {supportable} characteristics at "
        f"{MIN_BIN_N_EVENT} bads/bin over {MAX_N_BINS} bins -- below the target of "
        f"{TARGET_N_FEATURES[0]}-{TARGET_N_FEATURES[1]}. Expect a shorter scorecard "
        "and report confidence intervals on KS/Gini rather than point estimates."
    )


def bad_rate_drift(dev_rate: float, oot_rate: float) -> float:
    """Relative difference in bad rate between the development and OOT cohorts.

    If the population moved, the out-of-time test stops being a test of the
    model and becomes a test of whether the book changed underneath it.
    """
    if dev_rate <= 0:
        return 0.0
    return abs(dev_rate - oot_rate) / dev_rate


def drift_is_material(dev_rate: float, oot_rate: float) -> bool:
    return bad_rate_drift(dev_rate, oot_rate) > MAX_BAD_RATE_DRIFT


# --------------------------------------------------------------------------- #
# Output tables
# --------------------------------------------------------------------------- #

TARGET_BEHAVIOUR_TABLE = "target_behaviour"
TARGET_APPLICATION_TABLE = "target_application"
WATERFALL_TABLE = "exclusion_waterfall"
