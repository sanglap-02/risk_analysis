# Credit Risk Strategy Build — Project Plan new

**Owner:** Sanglap Kundu
**Reviewer / requester:** Shreyendra Garg
**Created:** 2026-08-07
**Platform:** Databricks (Free Edition or workspace), PySpark + Python
**Status:** Planning

---

## 0. Executive Summary

Build an end-to-end, industry-grade **behaviour scorecard and credit strategy engine** on Databricks. The system ingests bureau attributes and internal bank account-level attributes, engineers a point-in-time-correct feature set, trains a probability-of-default (PD) model, converts it into a scaled scorecard, segments the population into risk buckets, generates a **strategy decision tree** that assigns an action to every segment, and simulates the portfolio P&L impact of each candidate cut-off.

The deliverable is not "a model." It is a **decisioning system** with the artefacts a real credit risk shop produces:

| # | Artefact | Why it matters |
|---|---|---|
| 1 | Documented target definition and performance window | First question any validator asks |
| 2 | Point-in-time correct feature store | Prevents the leakage that kills most portfolio projects |
| 3 | WOE-binned, IV-screened feature set | The standard credit risk feature pipeline |
| 4 | Champion logistic scorecard (scaled to points) | Regulator-friendly, reason-code capable |
| 5 | Challenger GBM + SHAP | Shows the performance ceiling being left on the table |
| 6 | Validation pack (KS, Gini, PSI, calibration, rank order, OOT) | The governance layer |
| 7 | Risk buckets with monotone bad rates | The output the business consumes |
| 8 | Strategy decision tree with an action per leaf | The literal ask from Shreyendra |
| 9 | Portfolio simulation + swap-set analysis | Chooses the cut-off; the business-facing answer |
| 10 | Monitoring dashboard + model documentation | What makes it "industry level" not "Kaggle level" |

---

## 1. Key Design Decisions (decide these before writing any code)

### 1.1 Behaviour scoring, not application scoring

The attribute list supplied (account tenure, utilisation trend, cash advance frequency, overlimit occurrences, NSF incidents, rolling balances) describes **existing accounts with observable history**. That makes this a **behaviour scorecard** — scoring customers you already have, to drive line management, authorisation, collections prioritisation and pre-delinquency treatment.

Consequences of this choice:

- The observation point is a **snapshot date on an existing account**, not an application date.
- **No reject inference is required.** (Reject inference only applies to application scoring, where you never observe the performance of declined applicants.) Note this explicitly in the model doc — knowing *when it doesn't apply* is a stronger signal than applying it blindly.
- Actions available at the leaf nodes are **line increase / line decrease / hold / authorisation strategy / pre-collections treatment**, not approve/decline.

### 1.2 Champion / challenger, not one model

| | Champion | Challenger |
|---|---|---|
| Algorithm | Logistic regression on WOE-transformed bins | LightGBM / XGBoost on raw + engineered features |
| Purpose | Production scorecard | Performance benchmark, feature discovery |
| Explainability | Points per attribute → direct adverse-action reason codes | SHAP values |
| Why | US regulated lending (ECOA / Regulation B) requires specific reasons for adverse action; a points-based scorecard produces these mechanically | Quantifies how much signal linear+monotone binning discards |

Building only a GBM is the single most common tell that someone has not worked in regulated credit risk. Building only a logistic is the tell that someone has not kept up. Build both, report the Gini gap, and recommend consciously.

### 1.3 Time framework (the thing that was missing from the original plan)

```
  ◄──── Observation Window (24m) ────►│◄──── Performance Window (12m) ────►
  t-24                              t=0                                  t+12
   │                                 │                                    │
   └── features computed from here ──┘                                    │
                                     └──────── label observed here ───────┘
                                     ▲
                              Observation Point
                          (snapshot date, "as of")
```

**Every feature must be computable using only data available at or before t=0.** No exceptions. This is enforced structurally via the Feature Store's point-in-time join, not by hand.

### 1.4 Bad definition

> **Revised 2026-08-08 after measuring the real extract.** The original draft of this
> section specified `SK_DPD_DEF`. That was wrong for this dataset and would have
> produced 14 bads. See §1.4a for the measurement and the correction.

| Class | Definition | Treatment |
|---|---|---|
| **Bad** | `max(SK_DPD) >= 90` within the 12-month performance window | Label = 1 |
| **Indeterminate** | `30 <= max(SK_DPD) < 90` | **Excluded from training**, scored and reported at validation |
| **Good** | `max(SK_DPD) < 30` | Label = 0 |

**Do not label indeterminates as good.** They are genuinely ambiguous and including them blurs the decision boundary and depresses KS. Exclude from training; keep them in the OOT scoring set to confirm they rank between goods and bads (a strong sanity check — if they don't, the model is wrong).

### 1.4a Why `SK_DPD`, not `SK_DPD_DEF`

`SK_DPD_DEF` nets off delinquency below the lender's tolerance threshold, which is why it looked like the cleaner choice. Measured across all 104,307 card accounts:

| Column | Ever 30+ | Ever 60+ | Ever 90+ |
|---|---|---|---|
| `SK_DPD` | 3,162 | 2,192 | **1,806 (1.73%)** |
| `SK_DPD_DEF` | 393 | 114 | **39 (0.04%)** |

Under `SK_DPD_DEF` the development cohort contained **14 bads out of 49,796 accounts**. That is not a modellable target — it would have collapsed somewhere around Phase 7, after four weeks of feature engineering.

**Carry into the model documentation as a limitation:** `SK_DPD` includes delinquency the lender would have tolerated, so the target is marginally broader than a strict contractual default.

### 1.4b Threshold choice and product pooling

Measured on the pooled development cohort:

| Threshold | Bads | Bad rate |
|---|---|---|
| 90+ | 1,812 | 2.28% |
| 60+ | 1,848 | 2.33% |
| 30+ | 1,917 | 2.42% |

The near-indifference to the threshold means delinquency here is **highly persistent** — accounts reaching 30 DPD mostly roll to 90+. Two consequences: the 90+ definition is not arbitrary, and only ~105 accounts fall in the indeterminate band, so excluding them costs almost nothing.

**Products are pooled.** Card accounts alone give 870 bads — below the ~1,000 needed for a stable 10–15 variable scorecard. Pooling the card and POS panels gives 1,812 across 79,327 accounts.

| Population | Eligible | Bads | Bad rate |
|---|---|---|---|
| Card only | 49,796 | 870 | 1.75% |
| **Card + POS pooled** | **79,327** | **1,812** | **2.28%** |
| OOT cohort (pooled) | 103,467 | 1,921 | 1.86% |

The cost is product heterogeneity: utilisation, cash advance and overlimit do not exist for POS accounts. Handled the way credit risk always handles it — missing becomes its own WOE bin, and `product_type` enters as a characteristic in its own right.

### 1.4c Two models, not one

The behaviour scorecard is well-specified but thin at 1,812 bads. `application_train.TARGET` offers ~24,800 bads at 8.07%, but Home Credit ships no application date, so no genuine out-of-time split exists for it. Rather than trade one against the other, both are built from the same feature layer:

| | Primary: behaviour | Secondary: application |
|---|---|---|
| Population | Pooled card + POS accounts | Customers with panel history |
| Label | `max(SK_DPD) >= 90` in performance window | `application_train.TARGET` |
| Bads | 1,812 (2.28%) | ~24,800 (8.07%) |
| True OOT | **Yes** — genuine earlier cohort | **No** — random stratified holdout |
| Scorecard size | Expect 8–10 characteristics | Expect 12–15 |
| Stated limitation | Thin on bads; report CIs on KS/Gini, not point estimates | No OOT possible; say so plainly, do not present the holdout as OOT |

They differ *only* in population and label. That is the point: it demonstrates the feature layer is reusable and that the choice of target is a design decision rather than an accident of the data.

For the secondary model, note that every panel month is strictly before the application date, so behavioural features genuinely precede the label — this is a real industry model type (known-customer or existing-relationship application scoring), not a workaround.

### 1.5 Exclusions from the modelling population

Apply at the observation point, before anything else:

- Already 90+ DPD, charged off, or in collections at t=0 (already bad — nothing to predict)
- Bankrupt or deceased at t=0
- Confirmed fraud accounts (different risk process entirely)
- Accounts open less than 6 months at t=0 (insufficient behaviour history — route to application score)
- Dormant / zero-balance-and-zero-activity for the full 12 months prior to t=0
- Closed accounts at t=0
- Staff / employee accounts (non-representative)

Log the row count dropped at every exclusion step. This waterfall table goes into the model documentation and is a standard validator request.

---

## 2. Data Strategy

### 2.1 Chosen dataset: Home Credit Default Risk (primary)

It is the only public dataset carrying **both** sides of the requested attribute list, with genuine monthly panels on each side.

| Table | Rows (approx) | Grain | Role |
|---|---|---|---|
| `application_train.csv` | 307,511 | 1 per customer (`SK_ID_CURR`) | Demographics, income, external scores, bureau enquiry counts |
| `bureau.csv` | 1,716,428 | 1 per external tradeline | **Bureau tradelines from other institutions** |
| `bureau_balance.csv` | 27,299,925 | tradeline × month | **Bureau monthly DPD status panel** |
| `credit_card_balance.csv` | 3,840,312 | account × month | **Internal credit card monthly panel — the core behaviour table** |
| `POS_CASH_balance.csv` | 10,001,358 | account × month | Internal instalment/POS loan monthly panel |
| `installments_payments.csv` | 13,605,401 | 1 per payment | **Payment timing and shortfall — payment history** |
| `previous_application.csv` | 1,670,214 | 1 per prior application | Cross-product relationship depth |
| `HomeCredit_columns_description.csv` | — | — | Data dictionary; read this first |

Total ≈ 2.7 GB uncompressed, ~690 MB zipped. Comfortably handled by Databricks Free Edition serverless.

**Download:** https://www.kaggle.com/c/home-credit-default-risk/data

```bash
pip install kaggle
# place kaggle.json at ~/.kaggle/kaggle.json, chmod 600
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d ./data/raw
```
You must accept the competition rules on the Kaggle site once before the API download works.

### 2.2 Re-framing Home Credit as a behaviour-scoring problem

`application_train.TARGET` is an *application* label. Using it directly would contradict Section 1.1. Instead, construct a genuine behaviour label from the monthly panel:

`credit_card_balance.MONTHS_BALANCE` runs from `-96` (oldest) to `-1` (most recent), relative to each customer's current application. So:

- **Observation point:** `MONTHS_BALANCE = -13`
- **Observation window:** `MONTHS_BALANCE ∈ [-24, -13]` → all features built from here
- **Performance window:** `MONTHS_BALANCE ∈ [-12, -1]` → label observed here
- **Bad flag:** `MAX(SK_DPD) >= 90` over the performance window — see §1.4a for why not `SK_DPD_DEF`
- **Indeterminate:** `30 <= MAX(SK_DPD) < 90`
- **Eligibility:** at least 12 months of history before the observation point *and* at least 6 months of panel coverage after it
- **Products:** card and POS panels pooled — see §1.4b

This produces a real, defensible behaviour-scoring dataset with no leakage, from real data. **Measured:** 79,327 eligible accounts, 1,812 bads, 2.28% bad rate.

The 12-month history requirement was chosen over a 6-month one deliberately. Relaxing to 6 months grows the population to 121,125 but adds only ~4% more bads (1,883), while a large minority of accounts would then have their rolling 12-month features computed from partial windows — quietly weakening exactly the trend and volatility features that carry the most signal.

**Out-of-time split:** shift the observation point to `MONTHS_BALANCE = -19` for a second cohort (features `[-30, -19]`, performance `[-18, -7]`). Train on the earlier cohort, validate out-of-time on the later. This is what makes the OOT genuine rather than a random split with a time label pasted on.

### 2.2a Verified state of the extract

Downloaded and profiled 2026-08-08. All eight modelling tables match the published reference row counts exactly, and every declared natural key is unique. Recorded in `data/manifests/raw_profile.json` and `raw_verification.json` with SHA-256 digests per file.

**Referential integrity — orphans are expected here.** Home Credit released an anonymised *sample*, not a complete book, so every parent table is partial:

| Edge | Orphan distinct IDs | Orphan rows |
|---|---|---|
| `bureau_balance → bureau` | 5.3% | 11.4% |
| `bureau → application_train` | 13.8% | 14.6% |
| `credit_card_balance → previous_application` | 10.9% | 28.2% |
| `pos_cash_balance → previous_application` | 4.0% | 3.4% |
| `installments_payments → previous_application` | 3.9% | 9.2% |
| `previous_application → application_train` | 14.1% | 15.4% |

These are sampling artefacts, not corruption. The verification gate treats them as warnings up to a 25% materiality threshold on distinct IDs — above that it fails, because a broken join key or truncated parent would look the same, only far worse.

Use the **distinct-ID** rate, not the row rate, when reasoning about these. The row rate is inflated by how many monthly records each orphaned account happens to carry — `credit_card_balance` looks like a 28% breach on rows but is 10.9% on accounts.

**The `365243` sentinel is real and widespread** — 1,570,735 values across the extract:

| Table | Columns affected |
|---|---|
| `previous_application` | `DAYS_FIRST_DRAWING` 934,444 · `DAYS_TERMINATION` 225,913 · `DAYS_LAST_DUE` 211,221 · `DAYS_LAST_DUE_1ST_VERSION` 93,864 · `DAYS_FIRST_DUE` 40,645 |
| `application_train` | `DAYS_EMPLOYED` 55,374 |
| `application_test` | `DAYS_EMPLOYED` 9,274 |

Silver must null these before any `DAYS_`-derived feature is built. `DAYS_FIRST_DRAWING` is 56% sentinel — that column is close to unusable and should be treated as a missingness indicator rather than a duration.

**Panel index note:** `bureau_balance.MONTHS_BALANCE` spans `[-96, 0]` — it includes month 0 — while both internal panels span `[-96, -1]`. Bureau feature windows must account for that extra month or they will be off by one relative to the internal ones.

### 2.3 Attribute coverage map

**Internal / account-level (15 requested)**

| # | Requested attribute | Source | Status |
|---|---|---|---|
| 1 | Account tenure | `credit_card_balance`: `MIN(MONTHS_BALANCE)` per `SK_ID_PREV` | ✅ Real |
| 2 | Payment history / on-time ratio | `installments_payments`: `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT` | ✅ Real |
| 3 | Delinquency status (30/60/90 DPD) | `credit_card_balance.SK_DPD_DEF`, `POS_CASH_balance.SK_DPD_DEF` | ✅ Real |
| 4 | Credit utilisation + trend | `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` over months | ✅ Real |
| 5 | Average & peak balance (3/6/12m) | Rolling aggregates of `AMT_BALANCE` | ✅ Real |
| 6 | Transaction / spend behaviour | `AMT_DRAWINGS_CURRENT`, `CNT_DRAWINGS_POS/OTHER_CURRENT` | ✅ Real |
| 7 | Cash advance usage | `AMT_DRAWINGS_ATM_CURRENT`, `CNT_DRAWINGS_ATM_CURRENT` | ✅ Real — direct hit |
| 8 | Overlimit occurrences | Derived: `AMT_BALANCE > AMT_CREDIT_LIMIT_ACTUAL` | ✅ Real (derived) |
| 9 | NSF / returned payment | Proxy: `AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY`; `installments AMT_PAYMENT < AMT_INSTALMENT` | ⚠️ Proxy |
| 10 | Cross-product relationship depth | `previous_application`: distinct `NAME_CONTRACT_TYPE`; presence in CC vs POS panels | ✅ Real |
| 11 | Deposit account cash flow | — | ❌ **Synthesise** |
| 12 | Autopay enrolment | Proxy: low std-dev of payment-date offset ⇒ autopay-like; else synthesise | ⚠️ Proxy |
| 13 | Overall customer tenure | `MIN(DAYS_DECISION)` in `previous_application` | ✅ Real |
| 14 | Prior charge-off / recovery | `bureau.CREDIT_ACTIVE = 'Bad debt'`; `previous_application.NAME_CONTRACT_STATUS` | ⚠️ Partial |
| 15 | Internal behavioural risk score | — | ❌ **Synthesise** (see 2.4) |

**Bureau (9 requested)**

| # | Requested attribute | Source | Status |
|---|---|---|---|
| 1 | Credit score (FICO/Vantage) | `EXT_SOURCE_1/2/3` — normalised external scores | ✅ Real proxy |
| 2 | Length of credit history | `bureau.DAYS_CREDIT` min (oldest) and mean (avg age) | ✅ Real |
| 3 | Total tradelines + trade mix | `bureau` counts by `CREDIT_ACTIVE`, `CREDIT_TYPE` | ✅ Real |
| 4 | Bureau-wide utilisation | `SUM(AMT_CREDIT_SUM_DEBT) / SUM(AMT_CREDIT_SUM_LIMIT)` on revolving | ✅ Real |
| 5 | Delinquency & derogatory marks | `bureau_balance.STATUS ∈ {1,2,3,4,5}`; `CREDIT_DAY_OVERDUE`; `AMT_CREDIT_SUM_OVERDUE`; `CNT_CREDIT_PROLONG` | ✅ Real |
| 6 | Recent credit-seeking | `AMT_REQ_CREDIT_BUREAU_DAY/WEEK/MON/QRT/YEAR`; `bureau.DAYS_CREDIT > -365` | ✅ Real — direct hit |
| 7 | Total debt & DTI | `SUM(AMT_CREDIT_SUM_DEBT) / AMT_INCOME_TOTAL` | ✅ Real |
| 8 | Public records / bankruptcy | — | ❌ **Synthesise** |
| 9 | Months since most recent delinquency | Derived from `bureau_balance` STATUS panel | ✅ Real (derived) |

**Coverage: 20 of 24 attributes from real data.** Four are synthesised.

### 2.4 Rules for the synthesised supplement

Four attributes have no real source. Generate them into a separate `internal_supplement` Delta table keyed on `SK_ID_CURR`, and obey these rules or the project loses credibility:

1. **Never present synthetic fields as real.** Every table, notebook and the model doc must carry a `is_synthetic` column-level tag and a clear README note.
2. **Correlate them to real fields, don't draw them independently.** Deposit inflow should correlate with `AMT_INCOME_TOTAL`; the internal behavioural score should correlate with the real DPD history and utilisation. Independent noise is worse than useless — it teaches the model nothing and dilutes IV.
3. **Do not build the target into them.** The fastest way to ruin the project is to generate a synthetic behavioural score from the label. Generate it from *other features*, then verify its IV sits in a plausible band (0.2–0.5). If a synthetic feature comes out with IV > 0.6, you have leaked.
4. **Run the model twice** — once with and once without the synthetic block — and report both Ginis. If the synthetic features add more than a few Gini points, they are too clean; add noise until they behave like real bank data.
5. **Bankruptcy flag:** generate at a realistic ~1–2% prevalence, concentrated in the tail of the real derogatory-mark distribution.

### 2.5 Fallback / supplementary datasets

| Dataset | Link | When to use |
|---|---|---|
| **Lending Club (2007–2018, accepted + rejected)** | https://www.kaggle.com/datasets/wordsforthewise/lending-club | If you want a genuine **application** scorecard with real rejects — enables a proper reject-inference exercise |
| **Freddie Mac Single-Family Loan-Level** | https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset | Real monthly performance panels 1999–2025; gold standard for vintage curves and roll-rate analysis. Free, registration required |
| **Fannie Mae Single-Family Loan Performance** | https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data | Same, alternative issuer |
| **American Express Default Prediction** | https://www.kaggle.com/competitions/amex-default-prediction | 458,913 customers × 13-month anonymised behaviour panel. Perfect *shape*, but features are anonymised (`D_39`, `S_3`…) so unusable for a business-narrative project |
| **Give Me Some Credit** | https://www.kaggle.com/c/GiveMeSomeCredit | 150k rows, 10 features. Good for a one-day dry run of the WOE→scorecard pipeline before touching Home Credit |
| **UCI Default of Credit Card Clients (Taiwan)** | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients | 30k rows, 6-month repayment panel. Smallest viable end-to-end rehearsal |
| **Curated list of credit risk datasets** | https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html | Broader survey if you want alternatives |

**Recommendation:** rehearse the full pipeline on *Give Me Some Credit* in one day (it will surface every API and syntax problem cheaply), then run the real build on Home Credit.

---

## 3. Environment Setup

### 3.1 Databricks

Sign up for **Databricks Free Edition** — no cloud account, no credit card, no business email required. Serverless compute and default storage are included.

- Sign-up: https://www.databricks.com/learn/free-edition
- Docs (AWS): https://docs.databricks.com/aws/en/getting-started/free-edition
- Docs (Azure): https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition
- Docs (GCP): https://docs.databricks.com/gcp/en/getting-started/free-edition

Free Edition is quota-limited and serverless-only. For a ~3 GB dataset with Spark aggregations this is sufficient. If you hit quota limits, downsample to a stratified 30% of `SK_ID_CURR` — preserve the bad rate exactly when you do, and record the sampling weight.

### 3.2 Libraries

```python
%pip install optbinning scorecardpy shap lightgbm xgboost \
             scikit-learn statsmodels matplotlib seaborn dtreeviz
dbutils.library.restartPython()
```

| Library | Role | Link |
|---|---|---|
| **optbinning** | Optimal monotonic binning, WOE, IV, `Scorecard` class with points scaling. **Primary binning engine.** | https://gnpalencia.org/optbinning/ · https://github.com/guillermo-navas-palencia/optbinning |
| **scorecardpy** | Python port of R `scorecard`. `woebin`, `woebin_adj` (interactive bin editing), `perf_eva` (KS/ROC), `perf_psi`. Excellent for PSI and manual bin overrides. | https://pypi.org/project/scorecardpy/ · https://github.com/ShichenXie/scorecardpy |
| **shap** | Challenger model explainability | https://shap.readthedocs.io/ |
| **lightgbm** | Challenger GBM (faster than XGBoost on wide tabular) | https://lightgbm.readthedocs.io/ |
| **dtreeviz** | Publication-quality decision tree visualisation for the strategy tree | https://github.com/parrt/dtreeviz |
| **statsmodels** | Logistic regression with p-values (sklearn does not give them; you need them for variable culling) | https://www.statsmodels.org/ |

### 3.3 Unity Catalog structure

```
catalog:  credit_risk
├── bronze          -- raw ingested files, schema-on-read, immutable
│   ├── application, bureau, bureau_balance, credit_card_balance,
│   │   pos_cash_balance, installments_payments, previous_application
├── silver          -- cleaned, typed, deduplicated, exclusions applied
│   ├── population_master        -- one row per eligible customer + obs point
│   ├── cc_panel_clean, bureau_panel_clean, installments_clean
│   └── internal_supplement      -- synthetic block, tagged
├── gold            -- modelling-ready
│   ├── features_internal        -- feature table (Feature Store)
│   ├── features_bureau          -- feature table (Feature Store)
│   ├── target_table             -- SK_ID_CURR, obs_date, target, segment
│   ├── modelling_abt            -- the analytical base table
│   └── scored_population        -- score, PD, bucket, strategy leaf, action
└── reporting
    ├── validation_metrics, psi_monitoring, portfolio_simulation, swap_set
```

### 3.4 Notebook layout

```
/Workspace/credit_risk_strategy/
├── 00_config.py                  -- widgets, catalog names, constants, PDO/base score
├── 01_ingest_bronze.py
├── 02_clean_silver.py
├── 03_population_and_target.py   -- exclusion waterfall + bad definition
├── 04_features_internal.py
├── 05_features_bureau.py
├── 06_synthetic_supplement.py
├── 07_build_abt.py               -- point-in-time join, train/test/OOT split
├── 08_eda.py
├── 09_binning_woe_iv.py
├── 10_feature_selection.py       -- IV screen, correlation, VIF, stepwise
├── 11_model_champion_logistic.py
├── 12_model_challenger_gbm.py
├── 13_scorecard_scaling.py       -- points, reason codes
├── 14_validation.py              -- KS, Gini, PSI, calibration, rank order, OOT
├── 15_risk_bucketing.py
├── 16_strategy_tree.py
├── 17_portfolio_simulation.py    -- cut-off selection, swap set, EL
├── 18_monitoring.py
└── 19_model_documentation.py     -- auto-generates the model doc
```

---

## 4. Build Plan — Phase by Phase

### Phase 0 — Setup & rehearsal (Day 1)

**Do:** Create the Databricks workspace, install libraries, download Give Me Some Credit, run a minimal `optbinning` → logistic → KS pipeline end to end.
**Exit criterion:** You have produced one KS statistic from one scorecard. Nothing more.
**Why:** Every API surprise, permission issue and version conflict surfaces here, on a 5 MB dataset, instead of at hour 40 on a 3 GB one.

---

### Phase 1 — Ingestion → Bronze

**Objective:** Land all eight Home Credit CSVs as Delta tables, unmodified.

**Steps**
1. Upload the raw CSVs to a Unity Catalog volume (`/Volumes/credit_risk/bronze/raw_files/`).
2. Read with explicit schemas — do **not** use `inferSchema` on 27M-row `bureau_balance`; it triggers a full extra scan and mistypes nullable integer columns.
3. Write as Delta, partitioned where sensible (`bureau_balance` by `MONTHS_BALANCE` bucket).
4. Add ingestion metadata columns: `_ingested_at`, `_source_file`, `_row_hash`.

**Exit criteria**
- Row counts match the published dataset counts exactly (see table in §2.1).
- `Z-ORDER` applied on join keys (`SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`).
- Every table queryable from Databricks SQL.

**Gotchas**
- `credit_card_balance` and `POS_CASH_balance` share `SK_ID_PREV` — a customer may appear in both. Do not assume one product per customer.
- `bureau_balance` joins to `bureau` on `SK_ID_BUREAU`, **not** `SK_ID_CURR`. You must go through `bureau` to reach the customer.

---

### Phase 2 — Cleaning & QA → Silver

**Objective:** Typed, deduplicated, sentinel-corrected tables with a documented data-quality report.

**Steps**
1. **Sentinel values.** Home Credit encodes missing as `365243` in `DAYS_*` columns (this is 1000 years). Replace with null. Search every `DAYS_` field for it — it appears in `DAYS_EMPLOYED`, `DAYS_FIRST_DRAWING`, `DAYS_LAST_DUE` and others.
2. **Negative-days convention.** All `DAYS_*` are negative offsets from the application date. Convert to positive months for interpretability: `months_ago = -DAYS_X / 30.44`.
3. **Impossible values.** `AMT_CREDIT_LIMIT_ACTUAL = 0` with `AMT_BALANCE > 0` → cannot compute utilisation; flag rather than divide. `AMT_INCOME_TOTAL` has extreme outliers (one record at 117,000,000) → cap at the 99.5th percentile and keep a `was_capped` flag.
4. **Duplicates.** Check `(SK_ID_PREV, MONTHS_BALANCE)` uniqueness in both monthly panels.
5. **Missingness profile.** For every column: null %, distinct count, min/max, and — crucially — **whether missingness itself is predictive**. In credit data it usually is. `EXT_SOURCE_1` is null for ~56% of records and that nullity carries signal. Never blindly impute; create an explicit `_is_missing` indicator and treat "missing" as its own WOE bin.
6. **Referential integrity.** Every `SK_ID_BUREAU` in `bureau_balance` exists in `bureau`; every `SK_ID_PREV` in the panels exists in `previous_application`.

**Exit criteria**
- A `reporting.data_quality_report` table: one row per column with all the above.
- No sentinel `365243` values remain anywhere.
- Written data-quality summary listing every decision made and its justification.

**Gotcha:** Resist imputing with the mean. In credit risk, missing is a category. Mean-imputation of `EXT_SOURCE_1` destroys one of the strongest signals in the dataset.

---

### Phase 3 — Population & Target Definition

**This is the phase the original plan was missing. It is the most important one.**

**Steps**
1. Set the observation point per §2.2 (`MONTHS_BALANCE = -13` for the development cohort).
2. Apply the exclusion waterfall from §1.5, **logging the count dropped at each step**:

   | Step | Exclusion | Rows in | Rows dropped | Rows out |
   |---|---|---|---|---|
   | 0 | Starting population | — | — | N |
   | 1 | No credit card panel | | | |
   | 2 | < 12m history before obs point | | | |
   | 3 | < 6m coverage after obs point | | | |
   | 4 | Already 90+ DPD at obs point | | | |
   | 5 | Closed / dormant at obs point | | | |
   | 6 | Indeterminate (30–60 DPD in perf window) | | | |
   | 7 | **Final modelling population** | | | |

3. Compute the target: `bad = 1 if MAX(SK_DPD_DEF) >= 90 over [-12, -1] else 0`.
4. Compute the indeterminate flag and hold that population aside (do not delete it — it is a validation asset).
5. Build the **second, earlier cohort** at `MONTHS_BALANCE = -19` for the OOT sample.

**Exit criteria**
- `gold.target_table`: `SK_ID_CURR`, `obs_point`, `target`, `is_indeterminate`, `cohort` (dev / oot).
- Bad rate documented per cohort. If dev and OOT bad rates differ by more than ~30% relative, investigate before proceeding — it means the population shifted and your OOT is not comparable.
- The exclusion waterfall table saved to `reporting`.

**Gotcha:** Every subsequent feature must be filtered to `MONTHS_BALANCE <= obs_point`. Write a single reusable helper function that applies this filter, and use it in every feature notebook. Do not re-implement the filter per feature — that is how leakage gets in.

---

### Phase 4 — Feature Engineering

**Objective:** ~120–200 candidate features, all point-in-time correct, registered in the Feature Store.

Organise by attribute family, matching the request list.

#### 4.1 Internal — tenure & relationship
```
acct_tenure_months          = obs_point - MIN(MONTHS_BALANCE)
cust_tenure_months          = -MIN(DAYS_DECISION) / 30.44   [previous_application]
n_products_held             = COUNT(DISTINCT NAME_CONTRACT_TYPE)
n_active_products           = products with activity in last 3m
total_relationship_balance  = SUM(AMT_BALANCE) across all products
has_cc, has_pos, has_cash   = product presence flags
```

#### 4.2 Internal — payment history
```
ontime_pay_ratio_12m   = SUM(DAYS_ENTRY_PAYMENT <= DAYS_INSTALMENT) / COUNT(*)
avg_days_late_12m      = AVG(MAX(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT))
max_days_late_12m      = MAX(...)
pay_shortfall_ratio    = SUM(AMT_PAYMENT) / SUM(AMT_INSTALMENT)
n_partial_payments_12m = COUNT(AMT_PAYMENT < AMT_INSTALMENT * 0.99)
min_pay_only_ratio_6m  = months where payment ≈ minimum due (revolver signal)
pay_regularity_std     = STDDEV(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)   -- autopay proxy
```

#### 4.3 Internal — delinquency
```
max_dpd_3m / 6m / 12m / 24m
n_times_30dpd_12m, n_times_60dpd_12m, n_times_90dpd_12m
months_since_last_delinq          -- recency; usually stronger than counts
worst_dpd_ever
delinq_trend_6m_vs_12m            -- deteriorating vs improving
n_consecutive_clean_months
```

#### 4.4 Internal — utilisation (the highest-value family)
```
util_current            = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL   [at obs point]
util_avg_3m / 6m / 12m
util_max_12m, util_min_12m
util_trend_slope_6m     -- OLS slope of monthly utilisation; direction matters
util_volatility_12m     = STDDEV(monthly utilisation)
util_delta_3m_vs_12m    = util_avg_3m - util_avg_12m
months_util_gt_80pct_12m
months_util_gt_100pct_12m
```
*Note:* utilisation **trend** and **volatility** typically out-predict the level. A customer steady at 60% is materially safer than one who climbed from 20% to 60% in six months. Most naive builds only compute the level — computing the trajectory is what makes this look like professional work.

#### 4.5 Internal — balance
```
bal_avg_3m / 6m / 12m
bal_peak_12m, bal_min_12m
bal_trend_slope_6m
bal_to_limit_peak_ratio
bal_growth_rate_6m
```

#### 4.6 Internal — spend & transaction behaviour
```
n_txn_3m / 6m / 12m               [CNT_DRAWINGS_CURRENT]
spend_avg_3m / 6m / 12m           [AMT_DRAWINGS_CURRENT]
spend_trend_slope_6m
pct_pos_drawings, pct_atm_drawings, pct_other_drawings   -- category mix
n_zero_spend_months_12m
spend_volatility_12m
```

#### 4.7 Internal — distress signals
```
cash_adv_amt_3m / 6m / 12m        [AMT_DRAWINGS_ATM_CURRENT]
cash_adv_cnt_3m / 6m / 12m        [CNT_DRAWINGS_ATM_CURRENT]
cash_adv_to_total_spend_ratio     -- classic distress ratio
cash_adv_trend_6m
n_overlimit_months_12m            [AMT_BALANCE > AMT_CREDIT_LIMIT_ACTUAL]
max_overlimit_amt_12m
overlimit_severity                = MAX(bal - limit) / limit
n_nsf_events_12m                  -- proxy: payment < min due
months_since_last_overlimit
```

#### 4.8 Bureau — score & history
```
ext_score_1, ext_score_2, ext_score_3         -- FICO proxies
ext_score_mean, ext_score_min, ext_score_std
ext_score_n_missing                           -- missingness is predictive
oldest_tradeline_months       = -MIN(DAYS_CREDIT) / 30.44
avg_tradeline_age_months      = AVG(-DAYS_CREDIT) / 30.44
newest_tradeline_months
```

#### 4.9 Bureau — tradelines & mix
```
n_tradelines_total, n_tradelines_active, n_tradelines_closed
n_revolving, n_installment, n_mortgage, n_other      [CREDIT_TYPE]
pct_revolving_of_total
trade_mix_entropy                 -- diversity of credit types
n_tradelines_opened_6m / 12m
```

#### 4.10 Bureau — utilisation & debt
```
bureau_util_revolving  = SUM(AMT_CREDIT_SUM_DEBT) / SUM(AMT_CREDIT_SUM_LIMIT)  [revolving only]
bureau_total_debt      = SUM(AMT_CREDIT_SUM_DEBT)
bureau_total_limit     = SUM(AMT_CREDIT_SUM_LIMIT)
dti_bureau             = bureau_total_debt / AMT_INCOME_TOTAL
max_single_tradeline_util
n_maxed_tradelines                -- tradelines at >90% utilisation
```

#### 4.11 Bureau — delinquency & derogatory
```
bureau_n_30dpd_12m / 24m          [bureau_balance STATUS = '1']
bureau_n_60dpd_12m / 24m          [STATUS = '2']
bureau_n_90plus_12m / 24m         [STATUS IN ('3','4','5')]
bureau_months_since_last_delinq   -- recency beats count
bureau_max_dpd_ever
bureau_amt_overdue                [AMT_CREDIT_SUM_OVERDUE]
bureau_n_bad_debt                 [CREDIT_ACTIVE = 'Bad debt']
bureau_n_prolonged                [CNT_CREDIT_PROLONG]
bureau_worst_status_12m
```

#### 4.12 Bureau — credit-seeking
```
n_inquiries_1m / 3m / 6m / 12m    [AMT_REQ_CREDIT_BUREAU_MON/QRT/YEAR]
inquiry_velocity                  = n_inq_3m / NULLIF(n_inq_12m, 0)
n_new_accounts_6m / 12m
inquiry_to_open_ratio             -- shopping without success = adverse signal
```

#### 4.13 Synthetic supplement (per §2.4 rules)
```
deposit_avg_balance_3m / 6m
deposit_inflow_12m, deposit_outflow_12m
deposit_net_cashflow_ratio
deposit_volatility
n_deposit_od_events_12m
autopay_enrolled_flag
internal_behavioural_score
bankruptcy_flag, months_since_bankruptcy
```

**Feature Store registration**

```python
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

fe.create_table(
    name="credit_risk.gold.features_internal",
    primary_keys=["SK_ID_CURR", "obs_point"],
    timeseries_columns="obs_point",     # <-- enables point-in-time correctness
    df=features_internal_df,
    description="Internal account-level behaviour features, point-in-time as of obs_point",
)
```

Setting `timeseries_columns` is what makes `create_training_set` do a genuine point-in-time (as-of) join. This is the structural leakage guard from §1.3 — use it rather than trusting your own filters.

Docs: https://docs.databricks.com/aws/en/machine-learning/feature-store/

**Exit criteria**
- 120–200 features across the families above, registered in two Feature Store tables.
- A **feature dictionary** table: name, family, definition, source columns, window, expected direction of relationship with risk. Write the expected direction *before* you look at the data — it becomes your sanity check in Phase 6.
- Automated leakage test: for each feature, assert that no source row used has `MONTHS_BALANCE > obs_point`.

---

### Phase 5 — Analytical Base Table & Splits

**Steps**
1. Point-in-time join the two feature tables to `target_table` using `fe.create_training_set`.
2. Split:
   - **Train:** 70% of the dev cohort (stratified on target)
   - **Test:** 30% of the dev cohort (stratified)
   - **OOT:** the entire later cohort — never touched until final validation
3. Verify bad rates are consistent across train and test (they must be, by construction) and note the OOT bad rate.
4. Freeze the split with a fixed seed and persist the assignment as a column. Re-deriving splits on the fly makes results irreproducible.

**Exit criterion:** `gold.modelling_abt` with a `split` column. Row counts and bad rates per split documented.

---

### Phase 6 — Exploratory Data Analysis

Not decorative. EDA in credit risk exists to answer four specific questions.

1. **Does each feature relate to risk in the direction you predicted?** Plot bad rate by decile for every feature. Any feature where the observed direction contradicts your Phase 4 prediction is either a data bug or a genuine insight — investigate every one. This is the highest-yield hour in the entire project.
2. **Is the relationship monotone?** Non-monotone relationships need either coarse binning or a business explanation. A U-shaped bad rate on utilisation (low-utilisation customers being risky) usually means inactive accounts are contaminating the population — go back to exclusions.
3. **Is the bad rate stable over time?** Plot bad rate by observation month. A jump indicates a policy change or data issue that will destroy your OOT validation.
4. **Which features are proxies for each other?** Correlation heat map by family. You will find utilisation features correlated at 0.95+ — that is expected, and Phase 7 resolves it.

Also produce: univariate distributions with outlier flags, missingness heat map by feature family, and target concentration by segment.

---

### Phase 7 — Binning, WOE & Feature Selection

**7.1 Optimal binning**

```python
from optbinning import BinningProcess

binning_process = BinningProcess(
    variable_names=feature_names,
    max_n_prebins=20,
    min_prebin_size=0.05,        # each bin >= 5% of population
    min_n_bins=2,
    max_n_bins=6,
    monotonic_trend="auto_asc_desc",   # enforce monotone bad rate
    min_bin_n_event=30,          # >= 30 bads per bin for stability
)
binning_process.fit(X_train, y_train)
X_train_woe = binning_process.transform(X_train, metric="woe")
```

Fit on **train only**. Apply the fitted bins to test and OOT. Fitting bins on the full data is a subtle, extremely common leak.

**7.2 Understand what WOE and IV are**

```
WOE_bin = ln( (% of goods in bin) / (% of bads in bin) )
IV      = Σ over bins of [ (%goods - %bads) × WOE ]
```

WOE is a **transformation** — it re-expresses each bin as its log-odds contribution. IV is a **selection statistic** — a single number summarising a feature's predictive strength. (The original plan listed "Feature Selection (IV, WOE)" as one step; they are two different things and it is worth being precise about that in the write-up.)

Why WOE is used in credit risk specifically:
- Handles missing values natively (missing becomes its own bin with its own WOE)
- Handles outliers by construction (extremes fall into the edge bin)
- Linearises every relationship with respect to log-odds, which is exactly what logistic regression assumes
- Enforced monotonicity gives business-explainable, defensible score behaviour
- Makes coefficients directly comparable across features

**7.3 IV screening thresholds**

| IV | Interpretation | Action |
|---|---|---|
| < 0.02 | Not predictive | Drop |
| 0.02 – 0.10 | Weak | Keep only if business-meaningful |
| 0.10 – 0.30 | Medium | Keep |
| 0.30 – 0.50 | Strong | Keep |
| > 0.50 | Suspiciously strong | **Investigate for leakage before keeping** |

Treat IV > 0.5 as a leakage alarm, not a win. On this dataset, any feature computed from the performance window will light up here.

**7.4 Multicollinearity**
- Pairwise Pearson on WOE-transformed features; where |r| > 0.7, keep the higher-IV feature
- VIF; drop anything above 5
- Aim for a shortlist of 25–40 features entering model selection

**7.5 Final selection**
- Forward stepwise on the shortlist, using AIC or a p-value threshold of 0.05 (`statsmodels.Logit` gives you p-values; sklearn does not)
- **Target 10–15 features in the final scorecard.** More than ~20 is unmanageable operationally and invariably unstable in production. Fewer than 8 usually means you are leaving signal on the table.
- Enforce family diversity: do not let seven utilisation variants occupy the whole scorecard. Cap each family at 2–3.

**Exit criteria**
- Binning object persisted (needed at scoring time — it is part of the model artefact)
- IV table for all candidate features
- Final feature list with IV, coefficient, p-value, VIF, and business rationale for each

---

### Phase 8 — Champion Model: Logistic Scorecard

**8.1 Fit**
```python
import statsmodels.api as sm
model = sm.Logit(y_train, sm.add_constant(X_train_woe[final_features])).fit()
```

**8.2 Coefficient sign check — mandatory**

With WOE-transformed features and the standard `ln(good/bad)` convention, **every coefficient must carry the same sign**. A sign flip means multicollinearity is producing a nonsensical relationship — e.g. the model saying higher utilisation lowers risk once other variables are controlled. It may be statistically defensible and it is still commercially indefensible. Drop the offending variable and refit. Do not ship a scorecard with a wrong-signed coefficient.

**8.3 Scale to points**

```
factor = PDO / ln(2)
offset = base_score - factor × ln(base_odds)
score  = offset + factor × ln(odds)

points_i = -(β_i × WOE_i + β_0 / n) × factor + offset / n
```

Standard convention: `PDO = 20`, `base_score = 600`, `base_odds = 50:1`. This means a score of 600 corresponds to 50:1 good:bad odds, and every additional 20 points doubles the odds of being good.

`optbinning.Scorecard` implements this directly:
```python
from optbinning import Scorecard
sc = Scorecard(binning_process=binning_process, estimator=LogisticRegression(),
               scaling_method="pdo_odds",
               scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600})
```

**8.4 Reason codes**

For each scored account, the three attributes with the largest **negative** point contribution relative to the population's maximum achievable points on that attribute become the adverse-action reasons. This falls straight out of the points table — which is exactly why the logistic scorecard is the champion.

**Exit criteria**
- Points table: every feature, every bin, points assigned
- Score distribution plot for train / test / OOT
- Reason code generator, tested on sample accounts
- Model logged to MLflow with the binning object bundled in the artefact

---

### Phase 9 — Challenger Model: GBM + SHAP

**Steps**
1. Train LightGBM on the raw (unbinned) shortlist plus the features that failed the linearity assumption.
2. Handle imbalance with `scale_pos_weight` or `is_unbalance=True` — but **do not SMOTE**. Synthetic minority oversampling distorts the PD calibration, and a behaviour scorecard's calibrated PD is the input to the expected-loss calculation in Phase 12. A miscalibrated PD makes the whole economic layer wrong.
3. Tune with Optuna or Hyperopt, tracked in MLflow.
4. SHAP: global bar plot, beeswarm, dependence plots for the top 10 features, and waterfall plots for 3–5 individual accounts.
5. **Report the Gini gap.** Typical: GBM beats a WOE logistic by 3–8 Gini points on this kind of data. Then answer the real question in the model doc: *is that gap worth the explainability, stability and governance cost?* For a behaviour scorecard driving line increases, usually no. For a collections prioritisation model with no adverse-action requirement, often yes. Having a defensible answer here is what separates an analyst from a credit risk strategist.

---

### Phase 10 — Validation

Run every metric on **train, test and OOT** and report all three side by side.

| Metric | What it measures | Acceptance | Red flag |
|---|---|---|---|
| **KS** | Max separation between cumulative good and bad distributions | Behaviour scorecard: 35–55 | < 25 weak; > 65 check leakage |
| **Gini** | `2 × AUC - 1` | > 0.40 | Train − OOT gap > 0.10 ⇒ overfit |
| **AUC / ROC** | Ranking ability | > 0.70 | — |
| **PSI (score)** | Population stability, dev vs OOT | < 0.10 stable | 0.10–0.25 monitor; > 0.25 unusable |
| **PSI (per feature)** | Which feature is driving drift | < 0.10 each | Isolate and investigate any > 0.25 |
| **Rank ordering** | Bad rate strictly decreasing across score deciles | Monotone across all 10 | Any inversion ⇒ refit |
| **Calibration** | Predicted PD vs observed bad rate by decile | Within ±20% relative per decile | Systematic bias ⇒ recalibrate intercept |
| **Divergence** | Separation of good/bad score distributions | Higher is better | — |
| **Indeterminate check** | Mean score of indeterminates | Strictly between goods and bads | If not, the bad definition is wrong |

**PSI formula and thresholds**
```
PSI = Σ over bins of [ (%actual - %expected) × ln(%actual / %expected) ]
```
`scorecardpy.perf_psi` computes this directly.

Also produce:
- **Gains / lift table** by score decile: population %, bad %, cumulative bad capture, lift
- **Vintage curves** — cumulative bad rate by months-on-book per score band
- **Score distribution overlay** — dev vs OOT, visual PSI

**Exit criterion:** a validation pack. If any acceptance criterion fails, you return to Phase 7 — you do not proceed with a caveat.

---

### Phase 11 — Risk Bucketing & Strategy Tree

**11.1 Risk buckets**

Cut the score into 5–7 bands. Requirements:
- **Monotone bad rate** across bands, strictly
- Each band holds a meaningful population share (no band under ~5%)
- Bad rate ratio between adjacent bands is material (roughly 1.5×+) — otherwise merge them
- Band boundaries land on round score numbers for operational usability

| Bucket | Score range | Pop % | Bad rate | Cum bad capture | Indicative action |
|---|---|---|---|---|---|
| A1 — Very Low | 700+ | | | | Proactive line increase |
| A2 — Low | 650–699 | | | | Line increase on request |
| B — Moderate | 600–649 | | | | Hold; monitor |
| C — Elevated | 550–599 | | | | Hold; restrict cash advance |
| D — High | 500–549 | | | | Line decrease; auth restriction |
| E — Severe | < 500 | | | | Line decrease; pre-collections |

**11.2 The strategy decision tree**

This is the specific deliverable Shreyendra asked for, and its placement matters. The tree is **not** a parallel explanation of the PD model — it is a downstream segmentation built on top of the score.

Inputs to the tree:
- The **score band** (primary split — always first)
- A small set of **policy / operational attributes** that the score deliberately does not contain, or that carry actionability beyond risk: current utilisation, current credit limit, customer tenure, cross-product depth, months since last delinquency, deposit relationship flag

Configuration:
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(
    max_depth=4,                  # 3-4 max; deeper is unimplementable operationally
    min_samples_leaf=0.02,        # every leaf >= 2% of population
    criterion="gini",
)
```

Constraints that make it a *strategy* tree rather than a *model* tree:
- **Depth 3–4 maximum.** Operations teams implement these as rules. A depth-8 tree is not a strategy, it is a second model.
- **Every leaf ≥ 2% of population.** Smaller leaves are noise and cannot be staffed.
- **Every leaf gets an explicit action**, not a probability.
- Each leaf reports: population %, bad rate, average score, average balance, expected loss, and assigned action.

Visualise with `dtreeviz` for a presentation-quality output, and separately export the leaf definitions as a **rule table** — because the rule table, not the picture, is what gets implemented.

**11.3 Credit strategy rules layer**

On top of the tree, layer hard policy overrides that sit outside the model:
- Currently 60+ DPD → no line increase regardless of score
- Bankruptcy flag → maximum restriction regardless of score
- Account tenure < 6 months → no proactive line increase
- Overlimit in the last 3 months → cash advance restriction

Document these as a separate, ordered rule set. In production, policy rules always evaluate *after* the tree and can only make a decision more conservative, never less. State that ordering explicitly.

---

### Phase 12 — Loss Components & Portfolio Simulation

This is the layer that produces a business answer, and it should **feed** the cut-off decision, not follow it.

**Revised 2026-08-09.** The original draft treated LGD and EAD as stated constants. They are now **modelled from observed data** wherever the data supports it. The change was prompted by [levist7/Credit_Risk_Modelling](https://github.com/levist7/Credit_Risk_Modelling), which models all three components rather than assuming two of them — a genuinely better approach, and one worth adopting.

The scoping below is ours, because Home Credit ships no recovery, collections or charge-off tables. What follows is what is *actually* observable, measured before being written down.

---

**12.1 The workout window**

Modelling loss needs a second time framework, distinct from the PD framework in §1.3:

```
  ◄─ observation ─►│◄── performance ──►│◄──── workout (12m) ────►
                   │                   │
              obs point            DEFAULT MONTH          recovery observed
                                   (first 90+ DPD)         up to here
```

- **Default month** — the first month an account reaches 90+ DPD
- **Workout window** — 12 months after the default month, where recovery is observed
- **Eligibility** — the account must have ≥12 months of panel after its default month

**Measured feasibility** across all 1,806 ever-90+ card accounts:

| Post-default coverage available | Accounts | Share |
|---|---|---|
| ≥3 months | 1,753 | 97.1% |
| ≥6 months | 1,709 | 94.6% |
| **≥12 months** | **1,639** | **90.8%** |

Losing under 10% to the coverage requirement makes a 12-month workout window comfortably viable.

---

**12.2 LGD — two-stage, and why**

`LGD = 1 − recovery rate`. Our recovery is a **balance-reduction proxy**:

```
recovery_rate = clip( (balance_at_default − balance_after_workout) / balance_at_default , 0, 1 )
```

Measured on 1,624 eligible accounts:

| | Share |
|---|---|
| Zero recovery | **73.1%** |
| Any recovery (>0) | 26.9% |
| **Full recovery (=1)** | **21.9%** |

Mean 0.248, median 0.000.

**This distribution is why the model must be two-stage.** It has a 73% point mass at zero and a 22% point mass at one — it is not remotely continuous, and a single regression fitted to it would predict a nonsense ~0.25 for nearly everybody, wrong for the 73% who recover nothing and wrong for the 22% who recover everything.

```
Stage 1   Logistic:  P(recovery > 0)                    -- will anything come back?
Stage 2   Fractional logit:  E[recovery | recovery > 0]  -- how much, given some does?

LGD = 1 − ( P(recovery > 0) × E[recovery | recovery > 0] )
```

**Stage 2 uses a fractional logit, not OLS.** The reference project describes beta regression in its README but implements linear regression in the notebook. With 22% of the target sitting at exactly 1.0, a linear model predicts outside [0,1] for a large minority — a recovery rate of 1.3 is not a rounding problem, it is a negative loss. A GLM with a binomial family and logit link handles a continuous [0,1] target with boundary mass correctly.

> **Stated limitation, for the model document.** This is a *balance-reduction* proxy, not economic LGD. It does not capture collections costs, the time value of delayed recovery, debt sale proceeds, or the distinction between genuine repayment and a write-off that removes the balance. Home Credit ships none of those. The proxy is directionally meaningful and must not be presented as a Basel-compliant LGD.

---

**12.3 EAD — model utilisation at default, not CCF**

The textbook formulation is:

```
EAD = balance_at_obs + CCF × (limit_at_obs − balance_at_obs)
CCF = (balance_at_default − balance_at_obs) / (limit_at_obs − balance_at_obs)
```

**Measured, this is unusable on our data.** Across 1,418 defaulted accounts with a computable CCF:

| Statistic | Value |
|---|---|
| Mean | −7.22 |
| Median | −0.47 |
| 5th / 95th percentile | −17.66 / 0.00 |
| **Within [0, 1]** | **12.3%** |

The denominator `limit − balance` collapses toward zero for accounts already near their limit at observation — which is precisely the population most likely to default — so the ratio explodes and flips sign. Winsorising to [0,1] would discard the real signal for 88% of cases and keep a number that no longer means anything.

**So model the exposure directly instead:**

```
ead_ratio = balance_at_default / limit_at_obs          -- bounded, stable, interpretable
EAD       = ead_ratio × limit_at_obs
```

Fit with a fractional logit on `ead_ratio` (it is a bounded proportion, occasionally exceeding 1 for overlimit accounts, so allow a modest cap above 1 rather than at 1).

Report the classical CCF alongside as a **diagnostic**, with its instability shown. Demonstrating that you computed the textbook quantity, found it degenerate, and moved to something defensible is a stronger result than quietly presenting a winsorised CCF.

---

**12.4 What is modelled versus assumed**

Honesty here is the whole point:

| Component | Card accounts | POS accounts |
|---|---|---|
| **PD** | Modelled (Phases 7–9) | Modelled |
| **LGD** | Modelled — two-stage on the recovery proxy | **Assumed** — no balance/limit columns |
| **EAD** | Modelled — `ead_ratio` on limit | **Assumed** — outstanding principal |

POS accounts carry no balance or credit limit, so neither component is observable for them. They fall back to stated constants, flagged per row. Every table carries an `is_modelled` column so no downstream chart can silently mix the two.

---

**12.5 Expected loss**
```
EL = PD × LGD × EAD
```
with PD from the calibrated scorecard, and LGD/EAD from §12.2–12.3 where modelled and from constants otherwise.

Report EL two ways — fully modelled, and with the original flat assumptions (`LGD = 0.87`, `CCF = 0.50`) — and show the difference by risk bucket. If modelling the components does not change the cut-off decision, that is worth knowing and worth saying.

**12.6 Unit economics per account**
```
Revenue     = interest margin × avg balance + interchange × annual spend
Cost        = funding cost × avg balance + servicing cost + acquisition amortisation
Expected loss = EL as above
Contribution  = Revenue - Cost - Expected loss
```
State every assumption in a single parameters block and make them widget-driven so a reviewer can flex them.

**12.7 Cut-off selection**

For each candidate score cut-off, compute:
- Approval / line-increase rate
- Portfolio bad rate
- Total expected loss
- Total revenue
- Net contribution
- Bad capture rate at cut-off

Plot **approval rate vs bad rate** and **net contribution vs cut-off**. The optimal cut-off is where net contribution peaks — which is almost never the point where bad rate is minimised. Making that trade-off explicit is the entire point of the exercise.

**12.8 Swap-set analysis**

Compare the new strategy against the incumbent (for this project: a simple utilisation-and-DPD rule, or the raw external score used alone).

| | Incumbent: approve | Incumbent: decline |
|---|---|---|
| **New: approve** | Both approve — no change | **Swap-in**: new business gained. Its bad rate must be below portfolio average or the model is not adding value |
| **New: decline** | **Swap-out**: business rejected. Its bad rate must be well above portfolio average to justify the loss |

The comparison of swap-in bad rate vs swap-out bad rate is the single most persuasive number you can put in front of a credit committee. If swap-ins are cleaner than swap-outs, the new strategy is strictly better at constant volume. Lead with this.

**12.9 Scenario / stress testing**

Rerun the simulation with PD uplifted by 1.5× and 2× (proxying a recession) and report how each risk bucket's contribution behaves. Buckets that flip negative under stress are your concentration risk.

---

### Phase 13 — Monitoring & Dashboard

**Monitoring tables** (refreshed on a schedule via Databricks Workflows):
- Score PSI vs development sample, by month
- Feature-level PSI, top 10 features
- Actual vs expected bad rate by score band
- Approval / action-mix distribution over time
- Vintage curves by origination month

**Dashboard** — build in Databricks AI/BI Dashboards (Databricks SQL). Recommended pages:

1. **Portfolio overview** — population by risk bucket, bad rate by bucket, total exposure, total EL
2. **Model performance** — KS / Gini trend, score distribution over time, PSI gauge with traffic-light thresholds
3. **Strategy tree** — the tree visual plus the leaf rule table with volume and bad rate per leaf
4. **Simulation** — interactive cut-off slider driving approval rate, bad rate and net contribution
5. **Swap-set** — the 2×2 with bad rates and the volume/loss delta

Docs: https://docs.databricks.com/aws/en/dashboards/

**Model documentation** (Phase 19 notebook, auto-generated):
Purpose and scope · Data sources and lineage · Population and exclusion waterfall · Target definition and windows · Feature dictionary · Binning and WOE tables · Model specification and coefficients · Points table · Validation results (all three samples) · Limitations and known weaknesses · Assumptions (including every synthetic field) · Monitoring plan and trigger thresholds for re-development.

The SR 11-7 supervisory guidance is the reference framework for what this document should contain, and Databricks publishes an automated MRM-documentation accelerator worth reading: https://github.com/databricks-industry-solutions/fsi-mrm-generation

---

## 5. Timeline

| Week | Phases | Milestone |
|---|---|---|
| **1** | 0, 1, 2 | Rehearsal complete on Give Me Some Credit. Home Credit landed in Bronze, cleaned to Silver, data-quality report published |
| **2** | 3, 4 | **Target defined and defended.** Exclusion waterfall complete. Feature engineering underway |
| **3** | 4, 5, 6 | Feature Store populated, ABT built with train/test/OOT, EDA complete with directional checks |
| **4** | 7, 8, 9 | Binning and IV screening done. Champion scorecard scaled to points. Challenger GBM with SHAP |
| **5** | 10, 11 | Validation pack passed on OOT. Risk buckets set. Strategy tree built and rule table exported |
| **6** | 12, 13 | LGD/EAD models, portfolio simulation, cut-off chosen, swap-set analysis. Dashboard live. Model doc complete |

Weeks 2 and 5 are the ones that slip. Week 2 because target definition surfaces data problems that force you back to Phase 2. Week 5 because failing an OOT acceptance criterion sends you back to Phase 7. Budget for both.

---

## 6. Resources

### 6.1 Datasets

| Dataset | Link |
|---|---|
| **Home Credit Default Risk** (primary) | https://www.kaggle.com/c/home-credit-default-risk/data |
| Home Credit — column descriptions | Included in the download as `HomeCredit_columns_description.csv` |
| Lending Club — all loan data 2007–2018 | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Lending Club — alternative mirror | https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv |
| Freddie Mac Single-Family Loan-Level | https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset |
| Fannie Mae Single-Family Loan Performance | https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data |
| Fannie Mae Data Dynamics (portal) | https://capitalmarkets.fanniemae.com/tools-applications/data-dynamics |
| Fannie Mae loan performance tutorial (PDF) | https://capitalmarkets.fanniemae.com/media/9066/display |
| American Express Default Prediction | https://www.kaggle.com/competitions/amex-default-prediction |
| Give Me Some Credit (rehearsal set) | https://www.kaggle.com/c/GiveMeSomeCredit |
| UCI Default of Credit Card Clients | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| FHFA datasets hub | https://www.fhfa.gov/data/datasets |
| Curated list of credit risk datasets | https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html |

### 6.2 Databricks — platform

| Topic | Link |
|---|---|
| Free Edition sign-up | https://www.databricks.com/learn/free-edition |
| Free Edition docs (AWS) | https://docs.databricks.com/aws/en/getting-started/free-edition |
| Free Edition docs (Azure) | https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition |
| Free Edition docs (GCP) | https://docs.databricks.com/gcp/en/getting-started/free-edition |
| Serverless compute for notebooks | https://docs.databricks.com/aws/en/compute/serverless/notebooks |
| Reference architectures (downloadable diagrams) | https://docs.databricks.com/aws/en/lakehouse-architecture/reference |
| Medallion architecture | https://docs.databricks.com/aws/en/lakehouse/medallion |
| Unity Catalog | https://docs.databricks.com/aws/en/data-governance/unity-catalog/ |
| Delta Lake | https://docs.databricks.com/aws/en/delta/ |
| Databricks Workflows (orchestration) | https://docs.databricks.com/aws/en/jobs/ |
| Lakeflow Declarative Pipelines / DLT | https://docs.databricks.com/aws/en/dlt/ |
| AI/BI Dashboards | https://docs.databricks.com/aws/en/dashboards/ |

### 6.3 Databricks — ML

| Topic | Link |
|---|---|
| Feature Store overview | https://docs.databricks.com/aws/en/machine-learning/feature-store/ |
| Feature Store concepts & glossary | https://docs.databricks.com/aws/en/machine-learning/feature-store/concepts |
| **Point-in-time feature lookups** (critical for §1.3) | https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series |
| MLflow on Databricks | https://docs.databricks.com/aws/en/mlflow/ |
| Model lifecycle in Unity Catalog | https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/ |
| AutoML | https://docs.databricks.com/aws/en/machine-learning/automl/ |
| Hyperparameter tuning (Optuna / Hyperopt) | https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/ |
| `applyInPandas` for parallel scoring | https://docs.databricks.com/aws/en/pandas/pandas-function-apis |

### 6.4 Databricks — credit risk specific

| Resource | Link |
|---|---|
| **Credit decisioning demo** (DLT → AutoML → dashboard, end to end) | https://www.databricks.com/resources/demos/tutorials/lakehouse-platform/lakehouse-credit-decisioning |
| `dbdemos` installer | https://github.com/databricks-demos/dbdemos |
| Databricks Industry Solutions (GitHub org) | https://github.com/databricks-industry-solutions |
| **FSI Model Risk Management doc generation** | https://github.com/databricks-industry-solutions/fsi-mrm-generation |
| Credit risk modelling for CECL & stress testing | https://medium.com/@databricksfinserv/modernizing-credit-risk-modeling-for-cecl-and-stress-testing-with-databricks-80bb3f66cda8 |
| Databricks for Financial Services | https://www.databricks.com/solutions/industries/financial-services |
| Market / enterprise risk accelerator | https://www.databricks.com/solutions/accelerators/market-risk |

Install the credit decisioning demo first — it is the closest thing to a reference implementation of the platform layer, and it will show you the DLT and dashboard patterns without you having to invent them:
```python
%pip install dbdemos
import dbdemos; dbdemos.install('lakehouse-fsi-credit')
```

### 6.5 Python libraries

| Library | Docs | Repo |
|---|---|---|
| **optbinning** | https://gnpalencia.org/optbinning/ | https://github.com/guillermo-navas-palencia/optbinning |
| optbinning — Scorecard tutorial | https://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html | — |
| **scorecardpy** | https://pypi.org/project/scorecardpy/ | https://github.com/ShichenXie/scorecardpy |
| SHAP | https://shap.readthedocs.io/ | https://github.com/shap/shap |
| LightGBM | https://lightgbm.readthedocs.io/ | https://github.com/microsoft/LightGBM |
| XGBoost | https://xgboost.readthedocs.io/ | https://github.com/dmlc/xgboost |
| dtreeviz | — | https://github.com/parrt/dtreeviz |
| statsmodels (Logit) | https://www.statsmodels.org/stable/discretemod.html | — |
| scikit-learn — DecisionTreeClassifier | https://scikit-learn.org/stable/modules/tree.html | — |

### 6.6 Domain learning

| Resource | Link / Reference |
|---|---|
| **Credit Risk Scorecards** — Naeem Siddiqi | The standard industry text. Read chapters on target definition, binning, and validation before Phase 3 |
| **Intelligent Credit Scoring** — Naeem Siddiqi (2nd ed.) | Updated companion, covers ML challengers |
| **The Credit Scoring Toolkit** — Raymond Anderson | Deepest treatment of strategy trees and swap-set analysis |
| Developing scorecards in Python with OptBinning | https://medium.com/data-science/developing-scorecards-in-python-using-optbinning-ab9a205e1f69 |
| Credit scorecard modelling with optbinning (worked example) | https://medium.com/@aw_marcell/credit-score-modelling-project-11504f7ab530 |
| Reference repo — optbinning scorecard build | https://github.com/marcellinus-witarsah/credit-score-modelling-with-optbinning |
| How to develop a credit scorecard in Python | https://lelesgaray.github.io/blog/scorecard/ |
| Home Credit — business understanding & EDA walkthrough | https://medium.com/analytics-vidhya/home-credit-default-risk-part-1-business-understanding-data-cleaning-and-eda-1203913e979c |
| Home Credit — full-process reference | https://leandeep.com/datalab-kaggle/kb002.html |
| Home Credit — reference solution repo | https://github.com/NoxMoon/home-credit-default-risk |

### 6.7 Regulatory & governance

| Topic | Link |
|---|---|
| **SR 11-7 — Supervisory Guidance on Model Risk Management** (Federal Reserve, 2011) | https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm |
| SR 11-7 attachment (full guidance PDF) | https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf |
| OCC Bulletin 2011-12 (identical guidance, OCC adoption) | https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html |
| ECOA / Regulation B — adverse action requirements (12 CFR 1002) | https://www.consumerfinance.gov/rules-policy/regulations/1002/ |
| CFPB circular on adverse action notices for complex models | https://www.consumerfinance.gov/compliance/circulars/ |
| Basel — IRB approach and PD/LGD/EAD definitions | https://www.bis.org/basel_framework/ |
| SR 11-7 practitioner summary | https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 |

---

## 7. Glossary

| Term | Definition |
|---|---|
| **ABT** | Analytical Base Table — the single modelling-ready table, one row per observation unit |
| **Bad rate** | Proportion of the population meeting the bad definition within the performance window |
| **Behaviour scorecard** | Score for existing accounts using observed account behaviour |
| **CCF** | Credit Conversion Factor — proportion of an undrawn limit expected to be drawn before default |
| **DPD** | Days Past Due |
| **EAD** | Exposure at Default |
| **Gini** | `2 × AUC − 1`; standard discrimination measure in credit risk |
| **Indeterminate** | Account whose performance is ambiguous under the bad definition; excluded from training |
| **IV** | Information Value — single-number measure of a feature's predictive power |
| **KS** | Kolmogorov–Smirnov — maximum separation between cumulative good and bad score distributions |
| **LGD** | Loss Given Default |
| **MOB** | Months on Book — account tenure |
| **OOT** | Out-of-Time sample — validation on a later time period than training |
| **PD** | Probability of Default |
| **PDO** | Points to Double the Odds — the score scaling constant |
| **PSI** | Population Stability Index — measures distribution drift between two samples |
| **Reject inference** | Technique to infer performance of declined applicants; applies to application scoring only |
| **Roll rate** | Probability of moving from one delinquency bucket to the next in a month |
| **Swap set** | Accounts where the new and incumbent strategies disagree; swap-in gained, swap-out lost |
| **Vintage curve** | Cumulative bad rate plotted against months on book, by origination cohort |
| **WOE** | Weight of Evidence — `ln(%goods / %bads)` per bin; the standard credit risk transformation |

---

## 8. Open Items

1. **Confirm the behaviour-scoring frame with Shreyendra.** If he specifically wants an *application* scorecard, switch the primary dataset to Lending Club (which has real rejects) and add a reject-inference phase. Everything else in this plan carries over.
2. **Decide the synthetic supplement scope.** Four attributes need synthesising. Alternative: drop them and deliver 20 of 24 attributes on entirely real data, which is a cleaner story but does not fully satisfy the attribute list.
3. **Confirm the economic assumptions** (interest margin, LGD, servicing cost) or agree to run the simulation on stated illustrative values.
4. **Databricks Free Edition quota** — verify it handles the 27M-row `bureau_balance` aggregation. If not, apply the stratified downsample noted in §3.1.
