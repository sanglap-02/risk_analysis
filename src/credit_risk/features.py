"""Feature catalogue and window arithmetic (plans.md Phase 4).

Pure Python, like `cleaning` and `population`. The Spark implementations live in
`notebooks/04_features_internal.py` and `05_features_bureau.py`.

Two things here earn their keep.

**Window arithmetic in one place.** Every windowed feature resolves its month
range through `window_for()`. The anti-leakage guarantee is that no window may
extend past the observation point, and that is asserted once here rather than
re-derived per feature — which is exactly the kind of repetition that produces
one subtly wrong feature out of a hundred and fifty.

**Expected direction, written down before seeing the data.** Each feature
declares whether more of it should mean more risk. Phase 6 plots observed bad
rate by decile and compares. Every contradiction is then either a data bug or a
genuine insight, and both are worth an hour of anyone's time. Without the
prediction recorded up front, you rationalise whatever you observe.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

WINDOWS: tuple[int, ...] = (3, 6, 12)
TREND_WINDOW = 6  # months used for slope features


def window_for(obs_point: int, months: int) -> tuple[int, int]:
    """Inclusive (start, end) MONTHS_BALANCE range ending at the observation point.

    Ends AT the observation point, never after it. The observation month itself
    is known at decision time, so it is included.
    """
    if months < 1:
        raise ValueError(f"window must be at least 1 month, got {months}")
    return (obs_point - months + 1, obs_point)


def window_is_safe(obs_point: int, months: int) -> bool:
    """No part of the window may fall after the observation point."""
    return window_for(obs_point, months)[1] <= obs_point


# --------------------------------------------------------------------------- #
# Direction of the expected relationship with risk
# --------------------------------------------------------------------------- #

RISKIER = "+"  # more of this should mean more risk
SAFER = "-"  # more of this should mean less risk
UNSURE = "?"  # no confident prior, or a non-monotone relationship expected

DIRECTIONS = (RISKIER, SAFER, UNSURE)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    description: str
    direction: str
    windowed: bool = False
    card_only: bool = False

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.name}: direction must be one of {DIRECTIONS}")


# --------------------------------------------------------------------------- #
# Internal / account-level families
# --------------------------------------------------------------------------- #

TENURE = "tenure_relationship"
DELINQUENCY = "delinquency"
UTILISATION = "utilisation"
BALANCE = "balance"
SPEND = "spend"
DISTRESS = "distress"
PAYMENT = "payment_history"

_INTERNAL: tuple[FeatureSpec, ...] = (
    # --- tenure and relationship depth -------------------------------------
    FeatureSpec("acct_tenure_months", TENURE, "Months since the account first appeared", SAFER),
    FeatureSpec("n_months_observed", TENURE, "Panel months present in the observation window", SAFER),
    FeatureSpec("cust_n_accounts", TENURE, "Accounts this customer holds", UNSURE),
    FeatureSpec("cust_n_product_types", TENURE, "Distinct product types held", SAFER),
    FeatureSpec("cust_total_balance", TENURE, "Total balance across the relationship", UNSURE),
    FeatureSpec("cust_tenure_months", TENURE, "Months since the customer's first account", SAFER),

    # --- delinquency --------------------------------------------------------
    # Recency beats count in most credit models: a 90+ event two months ago is
    # far more predictive than three events two years ago.
    FeatureSpec("max_dpd", DELINQUENCY, "Worst days past due in the window", RISKIER, windowed=True),
    FeatureSpec("n_months_dpd30", DELINQUENCY, "Months at 30+ DPD", RISKIER, windowed=True),
    FeatureSpec("n_months_dpd60", DELINQUENCY, "Months at 60+ DPD", RISKIER, windowed=True),
    FeatureSpec("n_months_dpd90", DELINQUENCY, "Months at 90+ DPD", RISKIER, windowed=True),
    FeatureSpec("months_since_last_delinq", DELINQUENCY, "Months since the most recent 30+ event", SAFER),
    FeatureSpec("n_consecutive_clean_months", DELINQUENCY, "Clean months immediately before observation", SAFER),
    FeatureSpec("dpd_trend_3m_vs_12m", DELINQUENCY, "Recent worst DPD less the 12-month worst", RISKIER),
    FeatureSpec("ever_delinquent", DELINQUENCY, "Any 30+ event in the window", RISKIER),

    # --- utilisation --------------------------------------------------------
    # The highest-value family. Trend and volatility routinely out-predict the
    # level: steady at 60% is materially safer than 20% climbing to 60%.
    FeatureSpec("util_at_obs", UTILISATION, "Utilisation at the observation point", RISKIER, card_only=True),
    FeatureSpec("util_avg", UTILISATION, "Mean utilisation over the window", RISKIER, windowed=True, card_only=True),
    FeatureSpec("util_max", UTILISATION, "Peak utilisation over the window", RISKIER, windowed=True, card_only=True),
    FeatureSpec("util_min_12m", UTILISATION, "Trough utilisation over 12 months", RISKIER, card_only=True),
    FeatureSpec("util_std_12m", UTILISATION, "Volatility of monthly utilisation", RISKIER, card_only=True),
    FeatureSpec("util_trend_slope_6m", UTILISATION, "OLS slope of utilisation over 6 months", RISKIER, card_only=True),
    FeatureSpec("util_delta_3m_vs_12m", UTILISATION, "Recent mean utilisation less the 12-month mean", RISKIER, card_only=True),
    FeatureSpec("n_months_util_gt80_12m", UTILISATION, "Months above 80% utilisation", RISKIER, card_only=True),
    FeatureSpec("n_months_util_gt100_12m", UTILISATION, "Months above the limit", RISKIER, card_only=True),

    # --- balance ------------------------------------------------------------
    FeatureSpec("bal_avg", BALANCE, "Mean balance over the window", UNSURE, windowed=True, card_only=True),
    FeatureSpec("bal_max_12m", BALANCE, "Peak balance over 12 months", UNSURE, card_only=True),
    FeatureSpec("bal_min_12m", BALANCE, "Trough balance over 12 months", UNSURE, card_only=True),
    FeatureSpec("bal_trend_slope_6m", BALANCE, "OLS slope of balance over 6 months", RISKIER, card_only=True),
    FeatureSpec("bal_growth_6m", BALANCE, "Balance change across the 6-month window", RISKIER, card_only=True),

    # --- spend and transaction behaviour ------------------------------------
    FeatureSpec("spend_avg", SPEND, "Mean monthly drawings", UNSURE, windowed=True, card_only=True),
    FeatureSpec("n_txn", SPEND, "Drawing count over the window", SAFER, windowed=True, card_only=True),
    FeatureSpec("spend_std_12m", SPEND, "Volatility of monthly spend", RISKIER, card_only=True),
    FeatureSpec("n_zero_spend_months_12m", SPEND, "Months with no drawings", RISKIER, card_only=True),
    FeatureSpec("pct_atm_drawings_12m", SPEND, "Share of drawings taken at an ATM", RISKIER, card_only=True),
    FeatureSpec("pct_pos_drawings_12m", SPEND, "Share of drawings made at point of sale", SAFER, card_only=True),

    # --- distress signals ---------------------------------------------------
    # Cash advance usage and overlimit behaviour are the classic early warnings:
    # a customer funding day-to-day life on expensive revolving credit.
    FeatureSpec("cash_adv_amt", DISTRESS, "Cash advance amount over the window", RISKIER, windowed=True, card_only=True),
    FeatureSpec("cash_adv_cnt", DISTRESS, "Cash advance count over the window", RISKIER, windowed=True, card_only=True),
    FeatureSpec("cash_adv_to_spend_ratio_12m", DISTRESS, "Cash advances as a share of total drawings", RISKIER, card_only=True),
    FeatureSpec("cash_adv_trend_6m", DISTRESS, "OLS slope of cash advance amount", RISKIER, card_only=True),
    FeatureSpec("n_overlimit_months_12m", DISTRESS, "Months spent over the credit limit", RISKIER, card_only=True),
    FeatureSpec("max_overlimit_amt_12m", DISTRESS, "Largest overlimit amount", RISKIER, card_only=True),
    FeatureSpec("months_since_last_overlimit", DISTRESS, "Months since the last overlimit month", SAFER, card_only=True),
    FeatureSpec("n_min_payment_only_12m", DISTRESS, "Months paying at or near the minimum due", RISKIER, card_only=True),

    # --- payment history ----------------------------------------------------
    FeatureSpec("ontime_pay_ratio_12m", PAYMENT, "Share of instalments paid on or before the due date", SAFER),
    FeatureSpec("avg_days_late_12m", PAYMENT, "Mean lateness across instalments", RISKIER),
    FeatureSpec("max_days_late_12m", PAYMENT, "Worst lateness across instalments", RISKIER),
    FeatureSpec("n_late_payments_12m", PAYMENT, "Count of late instalments", RISKIER),
    FeatureSpec("n_short_payments_12m", PAYMENT, "Instalments paid short of the amount due", RISKIER),
    FeatureSpec("pay_shortfall_ratio_12m", PAYMENT, "Total paid over total due", SAFER),
    # Near-zero variance in payment timing looks like an autodraft, which is a
    # behavioural proxy for payment reliability -- one of the requested
    # attributes that has no direct column in this dataset.
    FeatureSpec("pay_regularity_std_12m", PAYMENT, "Volatility of payment timing (autopay proxy)", RISKIER),
)


def expand(specs: tuple[FeatureSpec, ...], windows: tuple[int, ...] = WINDOWS) -> tuple[FeatureSpec, ...]:
    """Expand windowed specs into one concrete spec per window."""
    out: list[FeatureSpec] = []
    for spec in specs:
        if not spec.windowed:
            out.append(spec)
            continue
        for months in windows:
            out.append(
                FeatureSpec(
                    name=f"{spec.name}_{months}m",
                    family=spec.family,
                    description=f"{spec.description} ({months}-month window)",
                    direction=spec.direction,
                    windowed=False,
                    card_only=spec.card_only,
                )
            )
    return tuple(out)


INTERNAL_FEATURES: tuple[FeatureSpec, ...] = expand(_INTERNAL)

INTERNAL_FAMILIES: tuple[str, ...] = (
    TENURE, DELINQUENCY, UTILISATION, BALANCE, SPEND, DISTRESS, PAYMENT,
)


def by_family(specs: tuple[FeatureSpec, ...] = INTERNAL_FEATURES) -> dict[str, list[FeatureSpec]]:
    grouped: dict[str, list[FeatureSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.family, []).append(spec)
    return grouped


def feature_names(specs: tuple[FeatureSpec, ...] = INTERNAL_FEATURES) -> list[str]:
    return [s.name for s in specs]


# --------------------------------------------------------------------------- #
# Family caps for feature selection (plans.md 7.5)
# --------------------------------------------------------------------------- #
# Utilisation variants correlate at 0.95+. Without a cap the whole scorecard
# becomes seven flavours of the same signal, which looks strong in development
# and is fragile in production.

MAX_FEATURES_PER_FAMILY = 3

FEATURE_TABLE_INTERNAL = "features_internal"
FEATURE_TABLE_BUREAU = "features_bureau"
FEATURE_DICTIONARY_TABLE = "feature_dictionary"
