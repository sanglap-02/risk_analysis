"""Central configuration for the credit risk strategy build.

Every constant that governs the shape of the model lives here, not scattered
through notebooks. Phase 3 (target definition) and Phase 8 (score scaling) both
read from this module so the two can never drift apart.

See plans.md sections 1.3-1.5 (time framework, bad definition, exclusions) and
2.1-2.2 (dataset, behaviour-scoring reframe).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (local)
# --------------------------------------------------------------------------- #
# The artefact directories are env-overridable so tests can redirect them to a
# temp dir. Without that, running the test suite would overwrite the committed
# schemas with schemas derived from fixture data -- a quiet, nasty failure.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CREDIT_RISK_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = Path(os.environ.get("CREDIT_RISK_RAW_DIR", DATA_DIR / "raw"))
MANIFEST_DIR = Path(os.environ.get("CREDIT_RISK_MANIFEST_DIR", DATA_DIR / "manifests"))
SCHEMA_DIR = Path(os.environ.get("CREDIT_RISK_SCHEMA_DIR", PROJECT_ROOT / "schemas"))

# --------------------------------------------------------------------------- #
# Kaggle source
# --------------------------------------------------------------------------- #

KAGGLE_COMPETITION = "home-credit-default-risk"

# --------------------------------------------------------------------------- #
# Unity Catalog layout (plans.md 3.3)
# --------------------------------------------------------------------------- #

CATALOG = "credit_risk"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
REPORTING_SCHEMA = "reporting"

# Unity Catalog volume holding the raw CSVs before they become Delta tables.
RAW_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw_files"


def bronze(table: str) -> str:
    return f"{CATALOG}.{BRONZE_SCHEMA}.{table}"


def silver(table: str) -> str:
    return f"{CATALOG}.{SILVER_SCHEMA}.{table}"


def gold(table: str) -> str:
    return f"{CATALOG}.{GOLD_SCHEMA}.{table}"


def reporting(table: str) -> str:
    return f"{CATALOG}.{REPORTING_SCHEMA}.{table}"


# --------------------------------------------------------------------------- #
# Time framework (plans.md 1.3, 2.2)
# --------------------------------------------------------------------------- #
# MONTHS_BALANCE in the Home Credit panels runs from -96 (oldest) to -1 (most
# recent), relative to each customer's application date. Placing the observation
# point partway back lets us build a genuine behaviour scorecard: features from
# strictly before the point, label from strictly after it.


@dataclass(frozen=True)
class Cohort:
    """One observation cohort: where we stand, what we look back on, what we predict."""

    name: str
    obs_point: int  # MONTHS_BALANCE value that acts as "today"
    obs_window_months: int  # length of the look-back feature window
    perf_window_months: int  # length of the forward performance window

    @property
    def obs_window(self) -> tuple[int, int]:
        """Inclusive (start, end) MONTHS_BALANCE range for feature construction."""
        return (self.obs_point - self.obs_window_months + 1, self.obs_point)

    @property
    def perf_window(self) -> tuple[int, int]:
        """Inclusive (start, end) MONTHS_BALANCE range for label observation."""
        return (self.obs_point + 1, self.obs_point + self.perf_window_months)


# Development cohort: features from [-24, -13], label from [-12, -1].
COHORT_DEV = Cohort("dev", obs_point=-13, obs_window_months=12, perf_window_months=12)

# Out-of-time cohort: an earlier observation point, so the OOT sample is
# genuinely out of time rather than a random split wearing a date label.
# Features from [-30, -19], label from [-18, -7].
COHORT_OOT = Cohort("oot", obs_point=-19, obs_window_months=12, perf_window_months=12)

COHORTS = (COHORT_DEV, COHORT_OOT)

# --------------------------------------------------------------------------- #
# Target definition (plans.md 1.4)
# --------------------------------------------------------------------------- #

BAD_DPD_THRESHOLD = 90  # 90+ DPD in the performance window => bad
INDETERMINATE_DPD_LOW = 30  # 30-89 DPD => indeterminate, excluded from training

# Measured 2026-08-08 on the full extract, across 104,307 card accounts:
#
#                       ever 30+     ever 60+     ever 90+
#   SK_DPD                 3,162        2,192        1,806   (1.73%)
#   SK_DPD_DEF               393          114           39   (0.04%)
#
# SK_DPD_DEF nets off tolerance-level delinquency, which is why the plan
# originally chose it. On this dataset that netting suppresses essentially all
# severe delinquency and leaves 14 bads in the development cohort -- not a
# modellable target. Raw SK_DPD is the only viable choice here.
#
# Worth carrying into the model documentation as a limitation: SK_DPD includes
# delinquency the lender would have tolerated, so the target is marginally
# broader than a strict contractual default.
DPD_COLUMN = "SK_DPD"

# 90 / 60 / 30 give 1,812 / 1,848 / 1,917 bads on the pooled dev cohort. The
# near-indifference means delinquency here is highly persistent -- accounts
# that reach 30 DPD mostly roll to 90+. Only ~105 accounts fall in the
# indeterminate band, so excluding them costs almost nothing.

# --------------------------------------------------------------------------- #
# Behaviour-scoring population: pooled products
# --------------------------------------------------------------------------- #
# Card alone yields 870 bads -- below the ~1,000 needed for a stable 10-15
# variable scorecard. Pooling the card and POS panels doubles that to 1,812.
#
# The cost is product heterogeneity: utilisation, cash advance and overlimit
# features do not exist for POS accounts. That is handled the way credit risk
# always handles it -- "missing" becomes its own WOE bin, and PRODUCT_TYPE
# enters the model as a characteristic in its own right.

BEHAVIOUR_PANELS: tuple[str, ...] = ("credit_card_balance", "pos_cash_balance")
PRODUCT_TYPE_COLUMN = "product_type"

# Measured expectations for the dev cohort at 12-month minimum history.
# Treated like expected_rows: a tripwire, reported not enforced.
EXPECTED_DEV_ELIGIBLE = 79_327
EXPECTED_DEV_BADS = 1_812
EXPECTED_DEV_BAD_RATE = 0.0228

# --------------------------------------------------------------------------- #
# The two models
# --------------------------------------------------------------------------- #
# Two scorecards are built from the same feature layer. They differ only in
# population and label, which is exactly the point: it demonstrates that the
# feature engineering is reusable and that the choice of target is a design
# decision rather than an accident of the data.


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label_source: str
    description: str
    has_true_oot: bool
    expected_bads: int
    expected_bad_rate: float
    limitation: str


MODEL_BEHAVIOUR = ModelSpec(
    key="behaviour",
    label_source="max(SK_DPD) >= 90 over the performance window",
    description=(
        "Primary. Behaviour scorecard on pooled card + POS accounts. Preserves "
        "the full time framework, so out-of-time validation is genuine."
    ),
    has_true_oot=True,
    expected_bads=1_812,
    expected_bad_rate=0.0228,
    limitation=(
        "Thin on bads. Expect a final scorecard nearer 8-10 characteristics than "
        "15, and report confidence intervals on KS and Gini rather than point "
        "estimates alone."
    ),
)

MODEL_APPLICATION = ModelSpec(
    key="application",
    label_source="application_train.TARGET",
    description=(
        "Secondary. Known-customer application scorecard: behavioural features "
        "from the panels plus bureau, predicting performance on the new loan. "
        "Every panel month is strictly before the application, so there is no "
        "leakage."
    ),
    has_true_oot=False,
    expected_bads=24_825,
    expected_bad_rate=0.0807,
    limitation=(
        "Home Credit ships no application date, so no genuine out-of-time split "
        "exists. Validation is on a random stratified holdout, and the model "
        "document must say so plainly rather than presenting it as OOT."
    ),
)

MODELS: tuple[ModelSpec, ...] = (MODEL_BEHAVIOUR, MODEL_APPLICATION)

# --------------------------------------------------------------------------- #
# Population exclusions (plans.md 1.5)
# --------------------------------------------------------------------------- #

MIN_HISTORY_MONTHS_BEFORE_OBS = 12  # need a full look-back window
MIN_COVERAGE_MONTHS_AFTER_OBS = 6  # need enough forward months to observe outcome
MIN_ACCOUNT_TENURE_MONTHS = 6  # younger accounts route to the application score

# --------------------------------------------------------------------------- #
# Data quality (plans.md Phase 2)
# --------------------------------------------------------------------------- #

# Home Credit encodes "missing" in DAYS_* columns as 365243 (~1000 years).
# Left in place it silently poisons every tenure and recency feature.
DAYS_SENTINEL = 365243

INCOME_CAP_PERCENTILE = 0.995  # AMT_INCOME_TOTAL has a 117,000,000 outlier

# --------------------------------------------------------------------------- #
# Binning / feature selection thresholds (plans.md 7.1, 7.3, 7.4)
# --------------------------------------------------------------------------- #

MIN_PREBIN_SIZE = 0.05  # each bin >= 5% of population
MIN_BIN_N_EVENT = 30  # each bin >= 30 bads
MAX_N_BINS = 6
IV_DROP_BELOW = 0.02
IV_LEAKAGE_ALARM = 0.50  # investigate, do not celebrate
MAX_ABS_CORRELATION = 0.70
MAX_VIF = 5.0
TARGET_N_FEATURES = (10, 15)  # final scorecard size

# --------------------------------------------------------------------------- #
# Score scaling (plans.md 8.3)
# --------------------------------------------------------------------------- #

PDO = 20  # points to double the odds
BASE_SCORE = 600
BASE_ODDS = 50  # 50:1 good:bad at BASE_SCORE

# --------------------------------------------------------------------------- #
# Validation acceptance thresholds (plans.md Phase 10)
# --------------------------------------------------------------------------- #

MIN_KS = 0.25
MIN_GINI = 0.40
MAX_TRAIN_OOT_GINI_GAP = 0.10
PSI_STABLE = 0.10
PSI_UNUSABLE = 0.25

# --------------------------------------------------------------------------- #
# Loss components (plans.md Phase 12)
# --------------------------------------------------------------------------- #
# Fallback constants, used where a component cannot be modelled. POS accounts
# carry no balance or credit limit, so neither LGD nor EAD is observable for
# them and they fall back to these.

LGD_UNSECURED_REVOLVING = 0.87
CCF_REVOLVING = 0.50  # undrawn limit expected to be drawn before default

# --- the workout window ---------------------------------------------------- #
# A second time framework, distinct from the PD framework above. The default
# month is the first month an account reaches the bad threshold; recovery is
# observed over the following WORKOUT_WINDOW_MONTHS.
#
# Measured across all 1,806 ever-90+ card accounts: 90.8% have at least 12
# months of post-default panel, so a 12-month window costs under 10% of the
# defaulted population.

WORKOUT_WINDOW_MONTHS = 12
MIN_WORKOUT_COVERAGE_MONTHS = 12

# --- LGD ------------------------------------------------------------------- #
# recovery_rate = clip((bal_at_default - bal_after_workout) / bal_at_default, 0, 1)
#
# Measured on 1,624 eligible accounts:
#     zero recovery  73.1%      any recovery  26.9%      full recovery  21.9%
#     mean 0.248, median 0.000
#
# A 73% mass at zero and a 22% mass at one is why the model must be two-stage.
# Fitted as one regression it would predict ~0.25 for nearly everybody: wrong
# for the 73% who recover nothing and wrong for the 22% who recover everything.

RECOVERY_FLOOR = 0.0
RECOVERY_CAP = 1.0
EXPECTED_ZERO_RECOVERY_SHARE = 0.731
EXPECTED_FULL_RECOVERY_SHARE = 0.219

# --- EAD ------------------------------------------------------------------- #
# The textbook CCF is degenerate on this data. Measured over 1,418 defaulted
# accounts with a computable value:
#
#     mean -7.22   median -0.47   within [0,1] only 12.3%
#
# The denominator (limit - balance) collapses toward zero for accounts already
# near their limit -- exactly the population most likely to default -- so the
# ratio explodes and flips sign. Model exposure directly instead:
#
#     ead_ratio = balance_at_default / limit_at_obs
#
# Bounded, stable, interpretable. The classical CCF is still computed and
# reported as a diagnostic, with its instability shown, because demonstrating
# that the textbook quantity was tried and found degenerate is a stronger
# result than quietly presenting a winsorised version of it.

EAD_RATIO_CAP = 1.5  # overlimit accounts can exceed 1.0; cap well above it
CCF_DIAGNOSTIC_BOUNDS = (0.0, 1.0)
EXPECTED_CCF_IN_BOUNDS_SHARE = 0.123


@dataclass(frozen=True)
class LossComponentSpec:
    """How one loss component is obtained, and what that costs in credibility."""

    component: str  # "LGD" | "EAD"
    method: str
    is_modelled: bool
    applies_to: str
    limitation: str  # required -- no component ships without a stated caveat


LOSS_COMPONENTS: tuple[LossComponentSpec, ...] = (
    LossComponentSpec(
        component="LGD",
        method="Two-stage: logistic P(recovery>0), then fractional logit E[recovery | >0]",
        is_modelled=True,
        applies_to="card",
        limitation=(
            "A balance-reduction proxy, not economic LGD. It captures neither "
            "collections costs, the time value of delayed recovery, debt sale "
            "proceeds, nor the difference between genuine repayment and a "
            "write-off that removes the balance. Home Credit ships none of "
            "those. Directionally meaningful; not Basel-compliant."
        ),
    ),
    LossComponentSpec(
        component="LGD",
        method=f"Assumed constant {LGD_UNSECURED_REVOLVING}",
        is_modelled=False,
        applies_to="pos",
        limitation="POS accounts carry no balance column, so recovery is unobservable.",
    ),
    LossComponentSpec(
        component="EAD",
        method="Fractional logit on ead_ratio = balance_at_default / limit_at_obs",
        is_modelled=True,
        applies_to="card",
        limitation=(
            "Departs from the textbook CCF form, which is degenerate here "
            "(12.3% of values within [0,1]). The classical CCF is reported "
            "alongside as a diagnostic so the departure is visible, not hidden."
        ),
    ),
    LossComponentSpec(
        component="EAD",
        method=f"Assumed CCF {CCF_REVOLVING} on outstanding principal",
        is_modelled=False,
        applies_to="pos",
        limitation="POS accounts carry no credit limit, so there is no undrawn amount to convert.",
    ),
)

# --------------------------------------------------------------------------- #
# Splits (plans.md Phase 5)
# --------------------------------------------------------------------------- #

TRAIN_FRACTION = 0.70
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Source table registry (plans.md 2.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableSpec:
    """Everything the ingestion layer needs to know about one source file."""

    name: str  # bronze table name (lower_snake)
    filename: str  # CSV filename as Kaggle ships it
    description: str
    grain: str
    expected_rows: int | None  # reference count; see note below
    natural_key: tuple[str, ...]  # columns that should uniquely identify a row
    zorder_by: tuple[str, ...]  # join keys, for OPTIMIZE ZORDER
    required: bool = True
    role: str = "modelling"  # modelling | reference | holdout

    @property
    def has_natural_key(self) -> bool:
        return bool(self.natural_key)


# NOTE ON expected_rows: these are the widely published counts for this
# competition, used as a tripwire against a truncated or partial download.
# verify_raw.py reports mismatches rather than hard-failing, so a discrepancy
# tells you the truth about your download instead of blocking you on a constant
# that may have drifted. Update these from the manifest once you have verified
# your own extract.

TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="application_train",
        filename="application_train.csv",
        description="Application-level attributes, demographics, external scores, bureau enquiry counts",
        grain="one row per customer",
        expected_rows=307_511,
        natural_key=("SK_ID_CURR",),
        zorder_by=("SK_ID_CURR",),
    ),
    TableSpec(
        name="application_test",
        filename="application_test.csv",
        description="Kaggle scoring set; retained for schema parity only, never used for training",
        grain="one row per customer",
        expected_rows=48_744,
        natural_key=("SK_ID_CURR",),
        zorder_by=("SK_ID_CURR",),
        role="holdout",
    ),
    TableSpec(
        name="bureau",
        filename="bureau.csv",
        description="Tradelines reported to the credit bureau by OTHER institutions",
        grain="one row per external tradeline",
        expected_rows=1_716_428,
        natural_key=("SK_ID_BUREAU",),
        zorder_by=("SK_ID_CURR", "SK_ID_BUREAU"),
    ),
    TableSpec(
        name="bureau_balance",
        filename="bureau_balance.csv",
        description="Monthly DPD status panel for each bureau tradeline",
        grain="tradeline x month",
        expected_rows=27_299_925,
        natural_key=("SK_ID_BUREAU", "MONTHS_BALANCE"),
        # Joins to the customer only via bureau.SK_ID_BUREAU -- there is no
        # SK_ID_CURR on this table. Easy trap.
        zorder_by=("SK_ID_BUREAU",),
    ),
    TableSpec(
        name="credit_card_balance",
        filename="credit_card_balance.csv",
        description="Internal credit card monthly panel: balance, limit, drawings, DPD. The core behaviour table.",
        grain="account x month",
        expected_rows=3_840_312,
        natural_key=("SK_ID_PREV", "MONTHS_BALANCE"),
        zorder_by=("SK_ID_CURR", "SK_ID_PREV"),
    ),
    TableSpec(
        name="pos_cash_balance",
        filename="POS_CASH_balance.csv",
        description="Internal POS / instalment loan monthly panel",
        grain="account x month",
        expected_rows=10_001_358,
        natural_key=("SK_ID_PREV", "MONTHS_BALANCE"),
        zorder_by=("SK_ID_CURR", "SK_ID_PREV"),
    ),
    TableSpec(
        name="installments_payments",
        filename="installments_payments.csv",
        description="Payment-level record: due date vs actual date, due amount vs paid amount",
        grain="one row per payment",
        expected_rows=13_605_401,
        # A single instalment can be paid across multiple entries, so there is
        # no clean natural key here. Uniqueness is asserted downstream instead.
        natural_key=(),
        zorder_by=("SK_ID_CURR", "SK_ID_PREV"),
    ),
    TableSpec(
        name="previous_application",
        filename="previous_application.csv",
        description="Prior applications with Home Credit; source of cross-product relationship depth",
        grain="one row per prior application",
        expected_rows=1_670_214,
        natural_key=("SK_ID_PREV",),
        zorder_by=("SK_ID_CURR", "SK_ID_PREV"),
    ),
    TableSpec(
        name="columns_description",
        filename="HomeCredit_columns_description.csv",
        description="Official data dictionary. Read this before writing a single feature.",
        grain="one row per column",
        expected_rows=None,
        natural_key=(),
        zorder_by=(),
        role="reference",
    ),
)

TABLES: dict[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}

MODELLING_TABLES: tuple[TableSpec, ...] = tuple(
    s for s in TABLE_SPECS if s.role == "modelling"
)

# Files Kaggle ships that we deliberately do not ingest.
IGNORED_FILES = frozenset({"sample_submission.csv"})
