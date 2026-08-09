# Running the rehearsal on Kaggle

The same code runs locally and in a Kaggle notebook. `config.py` detects which,
so nothing needs editing when you move between them.

## Why Kaggle for this

| | |
|---|---|
| Dataset already hosted | `wordsforthewise/lending-club` mounts at `/kaggle/input` — no download, nothing on your disk |
| 32 GB RAM, 20 GB `/kaggle/working` | Comfortably more than this needs |
| Fixed scientific stack | No Python 3.14 wheel surprises |
| Public notebook | A shareable link, which a local script is not |

Not suitable for the **Home Credit** build — that needs Spark and belongs on
Databricks. This is the rehearsal only.

## Setup

**1. New notebook** → <https://www.kaggle.com/code> → *New Notebook*.

**2. Attach the data.** *Add Data* → search `wordsforthewise/lending-club` → Add.
It appears at `/kaggle/input/lending-club/`.

**3. Enable Internet.** Notebook settings (right panel) → *Internet* → On.
Requires a phone-verified account. Needed for the `git clone` in step 4.

**4. First cell — pull the code:**

```python
!git clone -q https://github.com/sanglap-02/risk_analysis.git
import sys
sys.path.append('/kaggle/working/risk_analysis/lc/src')

from lending_club import config as cfg, columns as cols
print("on kaggle :", cfg.ON_KAGGLE)
print("raw file  :", cfg.RAW_FILE, cfg.RAW_FILE.exists())
print("artifacts :", cfg.ARTIFACT_DIR)
```

`ON_KAGGLE` should print `True` and `RAW_FILE` should resolve under
`/kaggle/input`. If it points at a local path instead, the dataset was not
attached.

**5. Install what the base image lacks:**

```python
!pip install -q optbinning
```

`pandas`, `numpy`, `scikit-learn`, `statsmodels` and `matplotlib` are already
present.

**6. Run the steps:**

```python
!python /kaggle/working/risk_analysis/lc/steps/02_explore.py
```

Skip `01_download.py` on Kaggle — the data is already mounted.

## Keeping the repo as the source of truth

**Do not paste modules into cells.** That is how the Kaggle copy and the repo
silently diverge, and you end up debugging a difference that exists only in one
of them.

Clone on every run instead. It costs two seconds and guarantees the notebook is
running the committed code. After pushing a change locally:

```python
!cd /kaggle/working/risk_analysis && git pull -q
```

Or restart the session, which re-clones from scratch.

## Session limits worth planning around

- **Idle disconnect** well before the maximum runtime. Anything long must be
  re-runnable from scratch — which the step scripts are, by design.
- **`/kaggle/working` persists** between runs of the same notebook (~20 GB), so
  artifacts survive. `/kaggle/input` is read-only.
- Save a version (*Save & Run All*) to get a clean, reproducible execution
  record — that is the artefact worth sharing.

## What does not transfer

No Spark, no Delta, no Unity Catalog. The medallion structure, the gates and the
audit discipline are all reproduced here in pandas and parquet, but the platform
layer is Databricks-only and stays with the Home Credit build.
