#!/usr/bin/env python3
"""Step 01 — download the Lending Club accepted-loans extract.

    python lc/steps/01_download.py

~600 MB gzipped, ~2.26M loans issued 2007 through 2018 Q4. pandas reads the
.gz directly, so it is never decompressed to disk.

Why this dataset for the rehearsal
----------------------------------
Two things it has that Home Credit does not:

  issue_d       a real origination date, so a genuine out-of-time split is
                possible -- train on 2007-2015, validate on 2016+. Home Credit
                ships no application date at all, which is why our primary
                project has to state "no true OOT" as a limitation.

  recoveries    real post-charge-off recovery amounts, so LGD and EAD can be
                modelled properly rather than proxied from balance reduction.

What it lacks is the monthly panel, so none of the 15 internal behavioural
attributes (utilisation trend, cash advance frequency, overlimit counts) exist
here. That is why this is a rehearsal and not the deliverable.
"""

from __future__ import annotations

import sys
from pathlib import Path

LC_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = LC_ROOT / "data" / "raw"

DATASET = "wordsforthewise/lending-club"
FILENAME = "accepted_2007_to_2018Q4.csv.gz"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / FILENAME

    if target.exists() and target.stat().st_size > 0:
        print(f"Already present: {target} ({target.stat().st_size / 1024**2:,.0f} MB)")
        return 0

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {FILENAME} from {DATASET} ...")
    print("~600 MB. This takes a few minutes.")
    api.dataset_download_file(DATASET, FILENAME, path=str(RAW_DIR))

    # The API sometimes wraps a single file in its own zip.
    if not target.exists():
        import zipfile

        for candidate in RAW_DIR.glob("*.zip"):
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(RAW_DIR)
            candidate.unlink()

    if not target.exists():
        print(f"Expected {target} but it is not there. Files present:", file=sys.stderr)
        for p in sorted(RAW_DIR.iterdir()):
            print(f"  {p.name}", file=sys.stderr)
        return 1

    print(f"\nDownloaded {target} ({target.stat().st_size / 1024**2:,.0f} MB)")
    print("Next: python lc/steps/02_explore.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
