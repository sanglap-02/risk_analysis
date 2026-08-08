#!/usr/bin/env python3
"""Download and extract the Home Credit Default Risk dataset.

Usage
-----
    python scripts/download_data.py            # download + extract, skip if present
    python scripts/download_data.py --force    # re-download even if files exist
    python scripts/download_data.py --keep-zip # keep the ~690 MB archive

Prerequisites
-------------
Kaggle credentials, in any of the forms the client accepts. Newer API tokens
are tried first:

    ~/.kaggle/access_token    a KGAT_... token, one line       chmod 600
    export KAGGLE_API_TOKEN=KGAT_...
    ~/.kaggle/kaggle.json     {"username": "...", "key": "..."}  chmod 600
    export KAGGLE_USERNAME=... KAGGLE_KEY=...

Get either from https://www.kaggle.com/settings -> API.

You must also accept the competition rules once, while signed in, at
https://www.kaggle.com/c/home-credit-default-risk/rules -- the API returns a
403 until you do, and the message it gives is not obvious about the cause.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import (  # noqa: E402
    IGNORED_FILES,
    KAGGLE_COMPETITION,
    RAW_DIR,
    TABLE_SPECS,
)

REQUIRED_FREE_BYTES = 4 * 1024**3  # ~690 MB zip + ~2.7 GB extracted, plus headroom

CREDENTIAL_HELP = """
Kaggle credentials not found. Any one of these works:

Option A -- API token (newer, preferred):
    Sign in at https://www.kaggle.com/settings -> API, create a token, then:
        mkdir -p ~/.kaggle
        printf '%s' 'KGAT_your_token' > ~/.kaggle/access_token
        chmod 600 ~/.kaggle/access_token

Option B -- kaggle.json (older username + key pair):
    Download kaggle.json from the same page, then:
        mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json

Option C -- environment variables:
    export KAGGLE_API_TOKEN=KGAT_your_token
    # or
    export KAGGLE_USERNAME=your_username KAGGLE_KEY=your_api_key

Then accept the competition rules once (required, or the API returns 403):
    https://www.kaggle.com/c/home-credit-default-risk/rules
"""


def credential_source() -> str | None:
    """Return a description of the credentials found, or None.

    The kaggle client accepts two generations of credential. Newer API tokens
    (`KGAT_...`) are tried first by `KaggleApi.authenticate()`; the older
    username+key pair is the fallback. Checking only for kaggle.json would
    reject a perfectly valid token setup.
    """
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))

    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN environment variable"
    for name in ("access_token", "access_token.txt"):
        path = config_dir / name
        if path.exists() and path.read_text().strip():
            return f"{path}"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME / KAGGLE_KEY environment variables"
    if (config_dir / "kaggle.json").exists():
        return f"{config_dir / 'kaggle.json'}"
    return None


def check_disk_space(target: Path) -> None:
    free = shutil.disk_usage(target).free
    if free < REQUIRED_FREE_BYTES:
        print(
            f"WARNING: only {free / 1024**3:.1f} GB free at {target}. "
            f"The dataset needs roughly {REQUIRED_FREE_BYTES / 1024**3:.0f} GB "
            "including the archive. Consider --keep-zip=false (the default) so "
            "the archive is removed after extraction.",
            file=sys.stderr,
        )


def already_extracted(raw_dir: Path) -> bool:
    return all((raw_dir / spec.filename).exists() for spec in TABLE_SPECS)


def download(raw_dir: Path) -> Path:
    """Download the competition archive. Returns the path to the zip."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading '{KAGGLE_COMPETITION}' to {raw_dir} ...")
    print("This is ~690 MB and typically takes a few minutes.")
    api.competition_download_files(KAGGLE_COMPETITION, path=str(raw_dir), quiet=False)

    zip_path = raw_dir / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        candidates = sorted(raw_dir.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError(
                f"Download reported success but no .zip found in {raw_dir}"
            )
        zip_path = candidates[0]
    return zip_path


def extract(zip_path: Path, raw_dir: Path) -> list[Path]:
    """Extract the outer archive, then any nested per-file archives Kaggle nests."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    # Kaggle sometimes ships each CSV as its own nested .zip inside the
    # competition archive. Flatten those too.
    for nested in sorted(raw_dir.glob("*.zip")):
        if nested == zip_path:
            continue
        print(f"  extracting nested {nested.name} ...")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(raw_dir)
        nested.unlink()

    return sorted(raw_dir.glob("*.csv"))


def report(raw_dir: Path) -> int:
    """Print what landed and flag anything missing. Returns count of missing files."""
    print("\nExtracted files")
    print("-" * 78)
    print(f"{'file':<46}{'size':>12}  {'status':<14}")
    print("-" * 78)

    expected = {spec.filename: spec for spec in TABLE_SPECS}
    missing = 0

    for filename, spec in expected.items():
        path = raw_dir / filename
        if path.exists():
            size = f"{path.stat().st_size / 1024**2:,.1f} MB"
            status = "ok"
        else:
            size = "-"
            status = "MISSING" if spec.required else "missing (optional)"
            if spec.required:
                missing += 1
        print(f"{filename:<46}{size:>12}  {status:<14}")

    for path in sorted(raw_dir.glob("*.csv")):
        if path.name not in expected and path.name not in IGNORED_FILES:
            size = f"{path.stat().st_size / 1024**2:,.1f} MB"
            print(f"{path.name:<46}{size:>12}  {'unexpected':<14}")

    print("-" * 78)
    total = sum(p.stat().st_size for p in raw_dir.glob("*.csv"))
    print(f"{'total':<46}{total / 1024**3:>11,.2f} GB")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files are present"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="keep the archive after extraction"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help=f"target directory (default {RAW_DIR})"
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if already_extracted(raw_dir) and not args.force:
        print(f"All expected files already present in {raw_dir}. Use --force to re-download.")
        report(raw_dir)
        return 0

    source = credential_source()
    if source is None:
        print(CREDENTIAL_HELP, file=sys.stderr)
        return 2
    print(f"Using Kaggle credentials from: {source}")

    check_disk_space(raw_dir)

    try:
        zip_path = download(raw_dir)
    except Exception as exc:  # noqa: BLE001 -- surface the cause, whatever it is
        message = str(exc)
        print(f"\nDownload failed: {message}", file=sys.stderr)
        if "403" in message or "Forbidden" in message:
            print(
                "\nA 403 here almost always means the competition rules have not "
                "been accepted on your account. Sign in and accept them at:\n"
                f"  https://www.kaggle.com/c/{KAGGLE_COMPETITION}/rules",
                file=sys.stderr,
            )
        elif "401" in message or "Unauthorized" in message:
            print(CREDENTIAL_HELP, file=sys.stderr)
        return 1

    extract(zip_path, raw_dir)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"Removed {zip_path.name} (use --keep-zip to retain it).")

    missing = report(raw_dir)
    if missing:
        print(f"\n{missing} required file(s) missing -- the download is incomplete.", file=sys.stderr)
        return 1

    print("\nNext: python scripts/generate_schemas.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
