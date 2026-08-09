"""Configuration for the Lending Club rehearsal build.

Same discipline as the main project: every constant that governs the model lives
here, and nothing redefines it downstream.

This is a REHEARSAL. Lending Club has no monthly panel, so none of the 15
internal behavioural attributes the brief asked for exist in it. Its purpose is
to teach the WOE -> logistic -> scorecard -> cut-off -> PSI -> LGD/EAD spine on
a dataset small enough to iterate on in seconds. The Home Credit build remains
the deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LC_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
# The same code runs locally and in a Kaggle notebook. On Kaggle the dataset is
# mounted read-only at /kaggle/input, so there is nothing to download; outputs
# go to /kaggle/working, which persists between runs of the same notebook.
#
# Detecting rather than configuring means no edit is needed when moving between
# the two, and no chance of a Kaggle copy quietly drifting from the repo.

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
ON_KAGGLE = KAGGLE_INPUT.exists() and KAGGLE_WORKING.exists()

RAW_FILENAME = "accepted_2007_to_2018Q4.csv.gz"

# Candidate locations, in preference order. The Kaggle dataset ships both a
# gzipped file and an uncompressed copy inside a folder; pandas reads either.
_RAW_CANDIDATES = [
    KAGGLE_INPUT / "lending-club" / RAW_FILENAME,
    KAGGLE_INPUT / "lending-club" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv",
    LC_ROOT / "data" / "raw" / RAW_FILENAME,
]


def _resolve_raw() -> Path:
    for candidate in _RAW_CANDIDATES:
        if candidate.exists():
            return candidate
    # Nothing found yet -- return the local path so the download step has a
    # target and the error message points somewhere useful.
    return LC_ROOT / "data" / "raw" / RAW_FILENAME


RAW_DIR = LC_ROOT / "data" / "raw"
RAW_FILE = _resolve_raw()

# Artifacts must land somewhere writeable. On Kaggle the repo clone sits under
# /kaggle/working already, but writing to an explicit artifacts directory keeps
# the two environments symmetrical.
ARTIFACT_DIR = (KAGGLE_WORKING / "artifacts") if ON_KAGGLE else (LC_ROOT / "data" / "artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
# Lending Club gives an outcome per loan in `loan_status`. The classification
# below differs from the reference project in one consequential way, explained
# under EXCLUDED_STATUSES.

BAD_STATUSES: tuple[str, ...] = (
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
)

GOOD_STATUSES: tuple[str, ...] = (
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
)

# Genuinely ambiguous: currently delinquent but not yet resolved. Excluded from
# training, scored at validation. Their mean score must land between goods and
# bads -- if it does not, the bad definition is wrong.
INDETERMINATE_STATUSES: tuple[str, ...] = (
    "Late (31-120 days)",
    "Late (16-30 days)",
    "In Grace Period",
)

# ---------------------------------------------------------------------------
# The correction that matters most in this rehearsal.
#
# `Current` means the loan is still being repaid and its outcome is NOT YET
# KNOWN. It is neither good nor bad; it is unmatured.
#
# The reference project (levist7/Credit_Risk_Modelling) labels everything
# outside its bad list as good:
#
#     loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(bad_def), 0, 1)
#
# which silently labels every `Current` loan GOOD. A loan three months into a
# sixty-month term has not survived anything yet, and some meaningful share of
# them will charge off later. Counting them as good contaminates the good
# population with future bads, depresses the apparent bad rate, and makes the
# model look better than it is.
#
# This is the same class of mistake as having no performance window at all: the
# target answers "what is the status right now" rather than "did this loan go
# bad". We exclude unmatured loans instead.
# ---------------------------------------------------------------------------
EXCLUDED_STATUSES: tuple[str, ...] = ("Current",)

TARGET = "target"  # 1 = bad
INDETERMINATE_FLAG = "is_indeterminate"

# --------------------------------------------------------------------------- #
# Maturity — the constraint that governs which vintages are usable
# --------------------------------------------------------------------------- #
# The extract ends 2018-12. A loan is only fully matured once
# issue_d + term <= 2018-12; before that, the loans that have *resolved* are a
# biased subset, because a charge-off resolves in months while a good loan only
# becomes "Fully Paid" at the end of its term.
#
# Measured bad rate for 36-month loans, by vintage:
#
#     vintage   % resolved   bad rate
#     2012        100.0%      13.58%
#     2013        100.0%      12.33%
#     2014        100.0%      13.73%
#     2015         99.9%      14.89%
#     2016         71.8%      19.85%   <- inflated by maturity bias
#     2017         40.1%      20.02%   <- inflated
#     2018         12.0%      13.25%   <- unusable
#
# Fully-resolved vintages sit in a stable 12-15% band. The jump at 2016 is not
# the book deteriorating; it is the denominator being incomplete. Using 2016 as
# the out-of-time sample would have produced a 24.9% "bad rate drift" that is
# entirely an artefact -- and it would have been reported as a genuine finding.

DATA_END = "2018-12-01"

# Restricted to 36-month loans. Two reasons, both real:
#   1. Maturity. 60-month loans are only fully matured for the 2012-2013
#      vintages, which is too thin for a dev/OOT split.
#   2. Homogeneity. 36 and 60-month loans are different products with different
#      risk horizons -- the 60-month bad rate runs roughly double. Mixing them
#      in one scorecard means the model spends capacity learning the term split
#      instead of learning risk.
LOAN_TERM_MONTHS = 36

# Both windows are fully matured, so their bad rates are directly comparable and
# any drift observed is real.
DEV_ISSUE_FROM = "2012-01-01"
DEV_ISSUE_TO = "2015-01-01"  # 2012-2014, ~306k loans, ~13.2% bad
OOT_ISSUE_FROM = "2015-01-01"
OOT_ISSUE_TO = "2016-01-01"  # 2015, ~283k loans, ~14.9% bad

TRAIN_FRACTION = 0.70
RANDOM_SEED = 42


def is_fully_matured(issue_date: str, term_months: int = LOAN_TERM_MONTHS) -> bool:
    """Would a loan issued then have completed its term within the extract?"""
    import pandas as pd

    return pd.Timestamp(issue_date) + pd.DateOffset(months=term_months) <= pd.Timestamp(DATA_END)

# --------------------------------------------------------------------------- #
# Binning and selection (mirrors the main project)
# --------------------------------------------------------------------------- #

MIN_PREBIN_SIZE = 0.05
MIN_BIN_N_EVENT = 30
MAX_N_BINS = 6
IV_DROP_BELOW = 0.02
IV_LEAKAGE_ALARM = 0.50
MAX_ABS_CORRELATION = 0.70
MAX_VIF = 5.0
TARGET_N_FEATURES = (10, 15)

# --------------------------------------------------------------------------- #
# Score scaling
# --------------------------------------------------------------------------- #
# PDO scaling, not the min/max rescale the reference project uses. With
# PDO = 20 and 600 at 50:1 odds, every 20 points doubles the odds of being good
# and each characteristic's points carry an odds interpretation. A linear
# rescale onto 300-850 produces FICO-looking numbers whose points mean nothing.

PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50

# --------------------------------------------------------------------------- #
# Validation acceptance
# --------------------------------------------------------------------------- #

MIN_KS = 0.25
MIN_GINI = 0.30  # application scorecards run lower than behaviour scorecards
MAX_TRAIN_OOT_GINI_GAP = 0.10
PSI_STABLE = 0.10
PSI_UNUSABLE = 0.25

MIN_BADS_FOR_SCORECARD = 1_000

# --------------------------------------------------------------------------- #
# Loss components
# --------------------------------------------------------------------------- #
# Unlike Home Credit, Lending Club carries real post-charge-off recovery
# amounts, so LGD and EAD are genuinely modelled here rather than proxied.
#
#   recovery_rate = recoveries / funded_amnt
#   LGD           = 1 - recovery_rate
#   EAD           = funded_amnt - total_rec_prncp   (principal outstanding)
#
# Expect heavy zero-inflation in recoveries, which is exactly why LGD is fitted
# in two stages.

RECOVERY_FLOOR = 0.0
RECOVERY_CAP = 1.0
LGD_FALLBACK = 0.87


@dataclass(frozen=True)
class ArtifactPaths:
    """Everything the pipeline writes, in one place."""

    clean = ARTIFACT_DIR / "clean.parquet"
    modelling = ARTIFACT_DIR / "modelling.parquet"
    binning = ARTIFACT_DIR / "binning_process.pkl"
    iv_table = ARTIFACT_DIR / "iv_table.csv"
    scorecard = ARTIFACT_DIR / "scorecard.csv"
    model = ARTIFACT_DIR / "pd_model.pkl"
    scored = ARTIFACT_DIR / "scored.parquet"
    validation = ARTIFACT_DIR / "validation.csv"
    buckets = ARTIFACT_DIR / "risk_buckets.csv"
    strategy_tree = ARTIFACT_DIR / "strategy_tree.csv"
    simulation = ARTIFACT_DIR / "simulation.csv"
    psi = ARTIFACT_DIR / "psi.csv"
    loss_models = ARTIFACT_DIR / "loss_models.csv"


PATHS = ArtifactPaths()
