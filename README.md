# Credit Risk Strategy Build

A behaviour scorecard and credit decisioning engine built on bureau attributes and
internal bank account-level attributes, running on Databricks.

| Document | What it covers |
|---|---|
| **[plans.md](plans.md)** | Full design, rationale, resource links. Read first. |
| [explain.md](explain.md) | The domain explained for a newcomer — why each decision was made. |
| [project_explain.md](project_explain.md) | The codebase file by file, with code walked through. |
| [walkthrough.md](walkthrough.md) | Complete account of the work to date, with diagrams. Written to be presented. |

---

## Current status

Data downloaded and verified 2026-08-08: 2.50 GB, all 8 modelling tables match
published row counts exactly, 23 checks passed / 9 warnings / 0 failures.

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & rehearsal | **done** |
| 1 | Ingestion → Bronze | **local half done; Spark half awaiting a workspace** |
| 2 | Cleaning & QA → Silver | next |
| 3 | Population & target definition | not started |
| 4 | Feature engineering | not started |
| 5 | ABT & splits | not started |
| 6 | EDA | not started |
| 7 | Binning, WOE, feature selection | not started |
| 8 | Champion logistic scorecard | not started |
| 9 | Challenger GBM + SHAP | not started |
| 10 | Validation | not started |
| 11 | Risk buckets & strategy tree | not started |
| 12 | Portfolio simulation | not started |
| 13 | Monitoring & dashboard | not started |

---

## Layout

```
.
├── plans.md                     Design document — read this first
├── src/credit_risk/
│   ├── config.py                Every constant that governs the model
│   └── schemas.py               Bronze schema inference, overrides, persistence
├── scripts/                     Local, run before touching Databricks
│   ├── download_data.py         Fetch + extract from Kaggle
│   ├── generate_schemas.py      Full-file type inference → schemas/*.json
│   └── verify_raw.py            Data-quality gate on the raw extract
├── notebooks/                   Databricks notebooks, run in order
│   ├── 00_config.py             Bootstrap: catalog, schemas, volume, constants
│   └── 01_ingest_bronze.py      Ingestion layer
├── schemas/                     Committed Spark schemas (build artefact)
├── data/
│   ├── raw/                     Downloaded CSVs (gitignored)
│   └── manifests/               Row counts, digests, null profiles (committed)
└── tests/
```

`data/raw` is gitignored; `data/manifests` is not. The manifests are how someone
else confirms they are working from the same extract without the 2.50 GB of CSVs
being in the repo — they carry row counts, SHA-256 digests per file, and null
profiles per column.

---

## Runbook

### 1. Local environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
pip install -e .
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
pip install -e .
```

If activation is blocked by execution policy:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

### 2. Kaggle credentials

Get a token from <https://www.kaggle.com/settings> → API. Either form works —
the newer `KGAT_` access token is tried first by the client.

**macOS / Linux**

```bash
mkdir -p ~/.kaggle
printf '%s' 'KGAT_your_token' > ~/.kaggle/access_token   # or: mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/access_token
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null

Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" `
            -Value 'KGAT_your_token' -NoNewline -Encoding ascii

# or, for the older credential form:
# Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

> **Windows gotcha:** do **not** use `echo 'token' > file`. PowerShell 5.1 writes
> redirected output as UTF-16LE with a byte-order mark. The kaggle client strips
> whitespace but not a BOM, so the token arrives corrupted and the failure looks
> like a bad token rather than a bad encoding. `Set-Content -Encoding ascii`
> avoids it. No `chmod` equivalent is needed — permissions are only enforced on
> POSIX.

Then accept the competition rules once while signed in — the API returns a 403
until you do, with an error that does not say so:
<https://www.kaggle.com/c/home-credit-default-risk/rules>

Confirm before downloading:

```
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); print('authenticated ok')"
```

### 3. Download, profile, verify

```bash
python scripts/download_data.py      # ~690 MB download, 2.50 GB extracted
python scripts/generate_schemas.py   # streams every file; writes schemas/ + manifest
python scripts/verify_raw.py         # keys, referential integrity, sentinels
```

On Windows use `.venv\Scripts\python.exe` (or activate the venv first) and
backslashes: `.venv\Scripts\python.exe scripts\download_data.py`.

`download_data.py` prints which credential source it found, so you can tell
immediately whether it picked up the token or the JSON.

`verify_raw.py` exits non-zero on FAIL. Resolve failures before going further —
a truncated `bureau_balance` produces a slightly optimistic bad rate rather than
an error, which is the worst kind of defect.

### 4. Databricks

Sign up for Free Edition (no cloud account needed): <https://www.databricks.com/learn/free-edition>
Then verify identity via LinkedIn — it raises several quota ceilings, and 2.50 GB
needs the headroom.

Clone this repo as a **Git folder** in the workspace. It must be the whole repo,
not two imported notebooks: `00_config.py` walks up to find `src/` and `schemas/`
and raises if they are missing.

Run `notebooks/00_config.py` **first** — it creates the catalog, the four schemas
and the volume, so the upload target exists. Then:

```bash
databricks fs cp --recursive data/raw dbfs:/Volumes/credit_risk/bronze/raw_files/
```

Then run `notebooks/01_ingest_bronze.py`.

Two constraints worth knowing:

- The workspace file uploader will not handle the 358 MB `bureau_balance.csv` —
  use the CLI.
- Free Edition restricts outbound internet to a limited set of trusted domains,
  so you **cannot** download from Kaggle inside a notebook. The data has to go up
  from your machine.

---

## Design notes worth knowing before you edit anything

**Bronze is immutable and complete.** No cleaning, no filtering, no dropping of
columns that look useless. Every such decision belongs in silver where it is
visible and reviewable.

**Schemas are committed artefacts, not runtime inference.** `generate_schemas.py`
streams each full CSV through pyarrow, applies the override rules in
`schemas.py`, and writes `schemas/<table>.json`. Spark reads that file. This is
faster than `inferSchema` (which triggers a second full pass) and, more
importantly, deterministic — inference types nullable integer columns
inconsistently between runs.

**Bronze widens, silver narrows.** `DAYS_*` and `AMT_*` land as double because
they carry nulls. They are cast down in silver, after the `365243` sentinel has
been dealt with explicitly.

**Constants live in `config.py` only.** The observation point, bad definition and
PDO are referenced by notebooks, never redefined in them. A development
scorecard and a production scorecard diverging on a constant nobody noticed is a
well-documented industry failure mode.

**`bureau_balance` has no `SK_ID_CURR`.** It reaches the customer only through
`bureau.SK_ID_BUREAU`. `verify_raw.py` checks that edge explicitly.
