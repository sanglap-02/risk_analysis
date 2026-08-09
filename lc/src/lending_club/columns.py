"""Column classification — which of the 151 Lending Club fields may be used.

This is the most important module in the rehearsal, and the one most public
Lending Club notebooks get wrong.

The dataset mixes three kinds of column and gives no hint which is which:

    APPLICATION     known when the credit decision was made -> usable
    POST_OUTCOME    only knowable after the loan ran -> LEAKS
    LENDER_DECISION known at application, but encodes Lending Club's own
                    risk assessment -> usable with a caveat

`recoveries` is the obvious trap: it is non-zero only for charged-off loans, so
a model given it scores a perfect AUC and is worthless. The subtler ones do the
same thing more quietly. `last_fico_range_low` is a FICO score refreshed
*during* the loan, so it has already absorbed the delinquency being predicted.
`out_prncp` is the balance still outstanding, which is zero for every loan that
finished. `total_rec_prncp` is how much principal came back.

Nothing here is dropped because it is unhelpful. Everything is dropped because
using it would be cheating, and the reason is recorded per column.
"""

from __future__ import annotations

from dataclasses import dataclass

APPLICATION = "application"
POST_OUTCOME = "post_outcome"
LENDER_DECISION = "lender_decision"
TARGET_SOURCE = "target_source"
IDENTIFIER = "identifier"
FREE_TEXT = "free_text"
SPARSE = "sparse"


@dataclass(frozen=True)
class ColumnRule:
    column: str
    kind: str
    reason: str

    @property
    def usable(self) -> bool:
        return self.kind in (APPLICATION, LENDER_DECISION)


# --------------------------------------------------------------------------- #
# Excluded — post-outcome. Using any of these is leakage.
# --------------------------------------------------------------------------- #

_POST_OUTCOME: dict[str, str] = {
    "out_prncp": "Principal still outstanding. Zero for every completed loan.",
    "out_prncp_inv": "As out_prncp, investor share.",
    "total_pymnt": "Total received. Directly encodes whether the loan was repaid.",
    "total_pymnt_inv": "As total_pymnt, investor share.",
    "total_rec_prncp": "Principal recovered over the life of the loan.",
    "total_rec_int": "Interest received. Higher for loans that ran to term.",
    "total_rec_late_fee": "Late fees charged -- only accrue when payments were missed.",
    "recoveries": "Post charge-off recovery. Non-zero ONLY for bads. The single "
                  "most destructive field in this dataset.",
    "collection_recovery_fee": "Fee on post charge-off recovery. Same problem.",
    "last_pymnt_d": "Date of final payment. Reveals whether the loan completed.",
    "last_pymnt_amnt": "Final payment amount. A large balloon means it was paid off.",
    "next_pymnt_d": "Only populated for loans still running.",
    "last_credit_pull_d": "Bureau pull refreshed during servicing.",
    "last_fico_range_high": "FICO refreshed DURING the loan -- has already absorbed "
                            "the delinquency we are trying to predict.",
    "last_fico_range_low": "As last_fico_range_high.",
    "pymnt_plan": "Set when a borrower is moved onto a payment plan, i.e. in trouble.",
    "debt_settlement_flag": "Settlement only happens after default.",
    "debt_settlement_flag_date": "As debt_settlement_flag.",
    "settlement_status": "As debt_settlement_flag.",
    "settlement_date": "As debt_settlement_flag.",
    "settlement_amount": "As debt_settlement_flag.",
    "settlement_percentage": "As debt_settlement_flag.",
    "settlement_term": "As debt_settlement_flag.",
    "hardship_flag": "Hardship programmes are granted mid-loan to struggling borrowers.",
    "hardship_type": "As hardship_flag.",
    "hardship_reason": "As hardship_flag.",
    "hardship_status": "As hardship_flag.",
    "hardship_dpd": "Days past due during hardship. This is the outcome itself.",
    "hardship_loan_status": "Loan status during hardship.",
    "hardship_amount": "As hardship_flag.",
    "hardship_start_date": "As hardship_flag.",
    "hardship_end_date": "As hardship_flag.",
    "hardship_length": "As hardship_flag.",
    "hardship_payoff_balance_amount": "As hardship_flag.",
    "hardship_last_payment_amount": "As hardship_flag.",
    "deferral_term": "Deferral is granted mid-loan.",
    "payment_plan_start_date": "As pymnt_plan.",
    "orig_projected_additional_accrued_interest": "Projected during hardship.",
    "collection_recovery_fee ": "Whitespace variant guard.",
}

# --------------------------------------------------------------------------- #
# Excluded — identifiers, free text, and administrative constants
# --------------------------------------------------------------------------- #

_IDENTIFIER: dict[str, str] = {
    "id": "Loan identifier.",
    "member_id": "Borrower identifier (entirely null in this extract).",
    "url": "Link to the listing.",
    "policy_code": "Constant.",
    "disbursement_method": "Operational, not a risk attribute.",
    "funded_amnt_inv": "Investor-funded portion. An outcome of the listing, not the borrower.",
}

_FREE_TEXT: dict[str, str] = {
    "emp_title": "Free text, ~300k distinct values. Usable only with NLP; out of scope.",
    "desc": "Borrower's free-text description. Mostly null after 2013.",
    "title": "Free text duplicate of `purpose`.",
    "zip_code": "Truncated to 3 digits. Proxy for geography and, in the US lending "
                "context, a fair-lending risk. Excluded deliberately.",
}

# Secondary-applicant and joint fields are populated for a small minority of
# loans. Kept out of the rehearsal to avoid a block of ~95%-null columns
# dominating the binning; a production build would treat them as a segment.
_SPARSE_PREFIXES = ("sec_app_", "revol_bal_joint", "annual_inc_joint", "dti_joint",
                    "verification_status_joint")

# --------------------------------------------------------------------------- #
# Lender decision — usable, with a caveat that must be stated
# --------------------------------------------------------------------------- #
# grade, sub_grade and int_rate are known at application, so they do not leak.
# But Lending Club SETS the interest rate from the grade, and the grade is LC's
# own risk model. Including them means a large part of the scorecard's
# performance is re-learning LC's existing decision rather than adding
# independent signal -- and the model then cannot be deployed to replace the
# grading process it depends on.
#
# The reference project includes all three without comment. We build both ways
# and report the Gini difference, which turns an unexamined assumption into a
# measured result.

_LENDER_DECISION: dict[str, str] = {
    "grade": "LC's own risk grade. Known at application; encodes LC's model.",
    "sub_grade": "Finer version of grade.",
    "int_rate": "Priced FROM the grade, so largely a restatement of it.",
    "installment": "Derived from loan_amnt, term and int_rate.",
}

_TARGET_SOURCE: dict[str, str] = {
    "loan_status": "The outcome. Becomes the target, never a feature.",
}


def _build() -> dict[str, ColumnRule]:
    rules: dict[str, ColumnRule] = {}
    for mapping, kind in [
        (_POST_OUTCOME, POST_OUTCOME),
        (_IDENTIFIER, IDENTIFIER),
        (_FREE_TEXT, FREE_TEXT),
        (_LENDER_DECISION, LENDER_DECISION),
        (_TARGET_SOURCE, TARGET_SOURCE),
    ]:
        for column, reason in mapping.items():
            rules[column.strip()] = ColumnRule(column.strip(), kind, reason)
    return rules


RULES: dict[str, ColumnRule] = _build()

# Columns needed for mechanics (splitting, loss models) but never as features.
CONTROL_COLUMNS: tuple[str, ...] = ("issue_d", "term", "loan_status")

# Needed by the LGD/EAD step only. Post-outcome by nature -- that is fine,
# because there the outcome IS the modelling target.
LOSS_MODEL_COLUMNS: tuple[str, ...] = (
    "funded_amnt", "recoveries", "total_rec_prncp", "out_prncp",
)


def classify(column: str) -> ColumnRule:
    """Classify a column. Anything unlisted is treated as an application field."""
    if column in RULES:
        return RULES[column]
    if column.startswith(_SPARSE_PREFIXES):
        return ColumnRule(
            column, SPARSE,
            "Secondary applicant / joint field, populated for a small minority of loans.",
        )
    return ColumnRule(column, APPLICATION, "Known at application time.")


def feature_columns(all_columns: list[str], *, include_lender_decision: bool = True) -> list[str]:
    """The columns usable as model features.

    `include_lender_decision=False` builds the independent-signal variant, so the
    contribution of LC's own grade can be measured rather than assumed.
    """
    out = []
    for column in all_columns:
        rule = classify(column)
        if not rule.usable:
            continue
        if rule.kind == LENDER_DECISION and not include_lender_decision:
            continue
        if column in CONTROL_COLUMNS:
            continue
        out.append(column)
    return out


def excluded_summary(all_columns: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for column in all_columns:
        rule = classify(column)
        if not rule.usable or column in CONTROL_COLUMNS:
            grouped.setdefault(rule.kind, []).append(column)
    return grouped
