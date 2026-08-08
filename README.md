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

Data downloaded, verified and landed in Delta 2026-08-08: 2.50 GB, all 8
modelling tables match published row counts exactly, 23 checks passed /
9 warnings / 0 failures, 9 bronze tables live in `credit_risk.bronze`.

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & rehearsal | **done** |
| 1 | Ingestion → Bronze | **done** |
| 2 | Cleaning & QA → Silver | **code complete, ready to run** |
| 3 | Population & target definition | **code complete, ready to run** |
| 4 | Feature engineering | **internal done; bureau next** |
| 5 | ABT & splits | not started |
| 6 | EDA | not started |
| 7 | Binning, WOE, feature selection | not started |
| 8 | Champion logistic scorecard | not started |
| 9 | Challenger GBM + SHAP | not started |
| 10 | Validation | not started |
| 11 | Risk buckets & strategy tree | not started |
| 12 | Loss components (LGD/EAD) + portfolio simulation | not started |
| 13 | Monitoring & dashboard | not started |

---

## Layout

```
.
├── plans.md                     Design document — read this first
├── src/credit_risk/
│   ├── config.py                Every constant that governs the model
│   ├── schemas.py               Bronze schema inference, overrides, persistence
│   ├── cleaning.py              Bronze → silver cleaning decisions (pure Python)
│   ├── population.py            Target definition + exclusion policy (pure Python)
│   └── features.py              Feature catalogue + window arithmetic (pure Python)
├── scripts/                     Local, run before touching Databricks
│   ├── download_data.py         Fetch + extract from Kaggle
│   ├── generate_schemas.py      Full-file type inference → schemas/*.json
│   └── verify_raw.py            Data-quality gate on the raw extract
├── notebooks/                   Databricks notebooks, run in order
│   ├── 00_config.py             Bootstrap: catalog, schemas, volume, constants
│   ├── 01_ingest_bronze.py      Ingestion layer
│   ├── 02_clean_silver.py       Sentinels, alignment, pooled panel
│   ├── 03_population_and_target.py  Exclusion waterfall + the label
│   └── 04_features_internal.py  Internal behaviour features
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

### 4. Databricks — full setup

The work is split across two places, and the order matters:

| Step | Where |
|---|---|
| 4.1 Create the workspace | Databricks web UI |
| 4.2 Push the repo, clone as a Git folder | GitHub + web UI |
| 4.3 Run `00_config.py` | **web UI** |
| 4.4 Install + authenticate the CLI | **your terminal** |
| 4.5 Upload the data | **your terminal** |
| 4.6 Run `01_ingest_bronze.py` | **web UI** |
| 4.7 Verify | web UI (SQL) |

Two Free Edition constraints shape how the notebooks are written:

- **Outbound internet is restricted to a limited set of trusted domains**, so you
  **cannot** download from Kaggle inside a notebook. The data has to be pushed up
  from the machine that downloaded it.
- **Serverless compute rejects `.cache()` and `.persist()`**
  (`NOT_SUPPORTED_WITH_SERVERLESS: PERSIST TABLE`). Where a notebook needs many
  counts over one derived frame, compute them in a single `agg()` rather than
  caching and counting repeatedly — that is both compatible and faster.

---

#### 4.1 Create the workspace

Sign up for Free Edition — no cloud account or business email needed:
<https://www.databricks.com/learn/free-edition>

Then **verify your identity via LinkedIn** in account settings. It raises several
quota ceilings, and 2.50 GB needs the headroom. Free Edition is serverless-only,
one workspace, one metastore; exceeding quota shuts down your compute for the
rest of the day.

#### 4.2 Push the repo and clone it as a Git folder

```bash
git init && git add -A && git commit -m "Initial commit"
gh repo create <your-repo-name> --private --source=. --push
```

The push is small (~100 KB) — `data/raw/` is gitignored, while `schemas/*.json`
and `data/manifests/*.json` are tracked deliberately as the reproducibility
record.

In Databricks: **Workspace → Create → Git folder**, point it at the repo URL.

> It must be the **whole repo**, not two imported notebooks. `00_config.py` walks
> up from `notebooks/` to find `src/` and `schemas/` and raises if they are
> missing.

Whenever you push a fix from your machine, come back here and click the branch
name → **Pull** before re-running anything.

#### 4.3 Run `notebooks/00_config.py`

Open it, attach to serverless, **Run All**.

It creates the catalog, the four schemas and the volume — so it has to run
*before* the upload, or there is nowhere to upload to.

Expect: your repo root and `src` path → catalog and schemas created → the time
framework and model constants printed → a final cell reporting the volume is
empty. Empty is correct at this stage.

> If it fails on `CREATE CATALOG` with a permissions error, your Free Edition
> account may not allow new catalogs. Point `CATALOG` in `src/credit_risk/config.py`
> at the built-in `workspace` catalog instead.

#### 4.4 Install and authenticate the CLI

**macOS**

```bash
brew tap databricks/tap
brew trust databricks/tap      # Homebrew blocks third-party taps until trusted
brew install databricks
```

**Windows**

```powershell
winget install Databricks.DatabricksCLI
```

Then authenticate. Your workspace URL is in the browser address bar:

```bash
databricks auth login --host https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
```

> **Two things that trip people up here.** Do not paste a URL that already starts
> with `https://` after the flag — `--host https://https://dbc-...` produces a
> confusing `failed to fetch host metadata for https://https:` warning. And when
> prompted for a profile name, press **Enter** to accept `DEFAULT`; anything else
> has to be passed as `--profile <name>` on every later command, and an `@` in the
> name causes quoting trouble.

A browser opens for OAuth. Confirm it worked:

```bash
databricks auth profiles
databricks fs ls dbfs:/Volumes/credit_risk/bronze/raw_files/
```

Empty output from the second command is correct — the volume exists and is
reachable. An error means 4.3 did not complete.

#### 4.5 Upload the data

```bash
databricks fs cp --recursive data/raw dbfs:/Volumes/credit_risk/bronze/raw_files/
databricks fs ls dbfs:/Volumes/credit_risk/bronze/raw_files/
```

2.50 GB, so allow time. There is a UI path too (**Catalog → volume → Upload to
this volume**), but nine files totalling 2.50 GB through a browser tab has no
resume and no useful progress indicator.

#### 4.6 Run `notebooks/01_ingest_bronze.py`

This notebook is driven by three widgets. **They only appear after the cell that
creates them has run** — scroll to the third code cell (the one starting
`dbutils.widgets.dropdown(...)`), run it with Shift+Enter, and a widget bar
appears at the top of the notebook.

| Widget | First run | Later runs |
|---|---|---|
| **Write mode** | `overwrite` | `skip_existing` to leave landed tables alone |
| **Tables (comma-separated, blank = all)** | blank | a subset to re-do just those |
| **Run OPTIMIZE ZORDER** | **`false`** | `true` once the data is in |

Widget values are read in the same cell that creates them, so after changing them
use **Run All** rather than resuming mid-notebook.

> Leave `run_optimize` off for the first pass. `OPTIMIZE ZORDER` on the 27M-row
> `bureau_balance` against a 2X-Small warehouse is slow enough to eat your daily
> compute quota before the data has even landed. Get the tables in, then optimize.

The notebook **raises** on a row-count mismatch or any unparseable row rather
than letting a bad load flow into silver. If it stops partway, the tables that
already succeeded are committed — fix the cause, pull, and re-run with
`skip_existing`.

#### 4.7 Verify

```sql
SELECT table, status, n_rows, expected_rows, row_count_ok, n_corrupt, duration_sec
FROM credit_risk.reporting.ingestion_audit
ORDER BY run_at DESC, table
```

Phase 1 is done when there are 9 tables, `row_count_ok` is true for the 8
modelling tables, and `n_corrupt` is 0 or null throughout.

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
