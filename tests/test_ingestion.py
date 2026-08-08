#!/usr/bin/env python3
"""Tests for the ingestion layer's local half.

Run with:  python tests/test_ingestion.py

Deliberately dependency-free (no pytest) so it runs in the same bare venv the
download scripts use. The Spark half of the ingestion layer is exercised on
Databricks; everything testable off-cluster is tested here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from credit_risk import config as cfg
from credit_risk import schemas as sch

FAILURES: list[str] = []

# Mirrors the status vocabulary in scripts/verify_raw.py.
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 78}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_cohort_windows() -> None:
    section("Cohort time framework")

    dev = cfg.COHORT_DEV
    check(dev.obs_window == (-24, -13), f"dev feature window is [-24, -13], got {dev.obs_window}")
    check(dev.perf_window == (-12, -1), f"dev label window is [-12, -1], got {dev.perf_window}")

    oot = cfg.COHORT_OOT
    check(oot.obs_window == (-30, -19), f"oot feature window is [-30, -19], got {oot.obs_window}")
    check(oot.perf_window == (-18, -7), f"oot label window is [-18, -7], got {oot.perf_window}")

    # The leakage guard the whole design rests on: no cohort may look at a month
    # at or after its own observation point when building features.
    for cohort in cfg.COHORTS:
        check(
            cohort.obs_window[1] <= cohort.obs_point,
            f"{cohort.name}: feature window ends at or before the observation point",
        )
        check(
            cohort.perf_window[0] > cohort.obs_point,
            f"{cohort.name}: label window starts strictly after the observation point",
        )
        check(
            cohort.obs_window[1] < cohort.perf_window[0],
            f"{cohort.name}: feature and label windows do not overlap",
        )

    # The OOT cohort must genuinely precede dev, or it is not out of time.
    check(
        cfg.COHORT_OOT.obs_point < cfg.COHORT_DEV.obs_point,
        "oot observation point is earlier than dev",
    )


def test_target_thresholds() -> None:
    section("Target definition")
    check(
        cfg.INDETERMINATE_DPD_LOW < cfg.BAD_DPD_THRESHOLD,
        "indeterminate band sits below the bad threshold",
    )
    # Measured on the full extract: SK_DPD_DEF nets off tolerance-level
    # delinquency so aggressively that it leaves 39 ever-90+ accounts out of
    # 104,307, and 14 bads in the dev cohort. Pinned so nobody "tidies" this
    # back to the stricter-looking column without re-measuring.
    check(cfg.DPD_COLUMN == "SK_DPD", "uses raw SK_DPD, not the over-netted SK_DPD_DEF")


def test_model_specs() -> None:
    section("Model specifications")
    keys = [m.key for m in cfg.MODELS]
    check(len(keys) == len(set(keys)), "model keys are unique")
    check(
        cfg.MODEL_BEHAVIOUR.has_true_oot and not cfg.MODEL_APPLICATION.has_true_oot,
        "only the behaviour model claims a genuine out-of-time split",
    )
    check(
        all(m.limitation for m in cfg.MODELS),
        "every model records its limitation, so the model doc cannot omit it",
    )
    check(
        cfg.MODEL_APPLICATION.expected_bads > cfg.MODEL_BEHAVIOUR.expected_bads,
        "application model is the better-powered of the two",
    )
    check(
        len(cfg.BEHAVIOUR_PANELS) == 2 and "credit_card_balance" in cfg.BEHAVIOUR_PANELS,
        "behaviour population pools both panel products",
    )
    check(
        all(p in cfg.TABLES for p in cfg.BEHAVIOUR_PANELS),
        "every pooled panel is a registered source table",
    )


def test_table_registry() -> None:
    section("Table registry")
    names = [s.name for s in cfg.TABLE_SPECS]
    check(len(names) == len(set(names)), "table names are unique")
    check(len(cfg.TABLES) == len(cfg.TABLE_SPECS), "TABLES dict covers every spec")

    bb = cfg.TABLES["bureau_balance"]
    check(
        "SK_ID_CURR" not in bb.zorder_by,
        "bureau_balance is not clustered by SK_ID_CURR (it has no such column)",
    )
    check(
        cfg.TABLES["installments_payments"].natural_key == (),
        "installments_payments claims no natural key (multiple payments per instalment)",
    )
    check(
        cfg.TABLES["application_test"].role == "holdout",
        "application_test is tagged holdout so it can never be trained on",
    )


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #


def test_type_resolution() -> None:
    section("Type override precedence")

    # Exact overrides beat everything, including a contradictory inference.
    check(sch.resolve_type("SK_ID_CURR", "double")[0] == sch.LONG, "SK_ID_CURR forced to long")
    check(sch.resolve_type("MONTHS_BALANCE", "double")[0] == sch.INTEGER, "MONTHS_BALANCE forced to integer")
    check(sch.resolve_type("SK_DPD_DEF", "double")[0] == sch.INTEGER, "SK_DPD_DEF forced to integer")

    # Exact beats prefix: CNT_CHILDREN is integer despite the CNT_ -> double rule.
    typ, reason = sch.resolve_type("CNT_CHILDREN", "double")
    check(typ == sch.INTEGER and reason == "exact-override", "exact override beats prefix override")

    # Prefix overrides widen the null-bearing numerics.
    check(sch.resolve_type("AMT_BALANCE", "long")[0] == sch.DOUBLE, "AMT_ widened to double")
    check(sch.resolve_type("DAYS_ENTRY_PAYMENT", "long")[0] == sch.DOUBLE, "DAYS_ widened to double")
    check(sch.resolve_type("CNT_INSTALMENT", "long")[0] == sch.DOUBLE, "CNT_ widened to double")

    # FLAG_ is deliberately NOT overridden: the prefix covers both 'Y'/'N'
    # strings and 0/1 integers, so any blanket rule corrupts one of them.
    check(
        sch.resolve_type("FLAG_OWN_CAR", sch.STRING) == (sch.STRING, "inferred"),
        "FLAG_OWN_CAR left as inferred string",
    )
    check(
        sch.resolve_type("FLAG_DOCUMENT_2", sch.INTEGER) == (sch.INTEGER, "inferred"),
        "FLAG_DOCUMENT_2 left as inferred integer",
    )


def test_arrow_mapping() -> None:
    section("Arrow to Spark type mapping")
    cases = {
        "int64": sch.LONG,
        "int32": sch.INTEGER,
        "double": sch.DOUBLE,
        "float": sch.DOUBLE,
        "string": sch.STRING,
        "bool": sch.BOOLEAN,
        "null": sch.STRING,
        "something_unknown": sch.STRING,
    }
    for arrow, expected in cases.items():
        got = sch.arrow_type_to_spark(arrow)
        check(got == expected, f"{arrow} -> {expected} (got {got})")


def test_struct_schema_is_spark_shaped() -> None:
    section("StructType JSON output")
    schema = sch.build_struct_schema(
        {"SK_ID_CURR": "int64", "AMT_BALANCE": "int64", "FLAG_OWN_CAR": "string"}
    )

    check(schema["type"] == "struct", "top level is a struct")
    check(len(schema["fields"]) == 3, "one field per column")

    by_name = {f["name"]: f for f in schema["fields"]}
    check(by_name["SK_ID_CURR"]["type"] == "long", "SK_ID_CURR typed long")
    check(by_name["AMT_BALANCE"]["type"] == "double", "AMT_BALANCE widened to double")
    check(by_name["FLAG_OWN_CAR"]["type"] == "string", "FLAG_OWN_CAR left string")

    # The bug this guards against: arrow type names leaking through as Spark
    # types when no override matched. 'int64' is not a valid Spark type name.
    valid = {"long", "integer", "double", "string", "boolean", "date", "timestamp"}
    check(
        all(f["type"] in valid for f in schema["fields"]),
        "every emitted type is a valid Spark type name",
    )

    check(
        all("arrow_type" in f["metadata"] and "type_source" in f["metadata"] for f in schema["fields"]),
        "provenance recorded in field metadata",
    )


def test_column_name_sanitisation() -> None:
    """Unity Catalog rejects empty names and several characters CSVs allow.

    HomeCredit_columns_description.csv ships an unnamed index column. UC refuses
    it with: At columns.0: name "" is not a valid name.
    """
    section("Column name sanitisation")

    check(sch.sanitise_column_name("", 0) == "col_0", "empty header becomes col_<position>")
    check(sch.sanitise_column_name("   ", 3) == "col_3", "whitespace-only header becomes col_<position>")
    check(sch.sanitise_column_name("My Column", 1) == "My_Column", "spaces replaced")
    check(sch.sanitise_column_name("a.b", 1) == "a_b", "periods replaced")
    check(sch.sanitise_column_name("a/b", 1) == "a_b", "forward slashes replaced")
    check(sch.sanitise_column_name("SK_ID_CURR", 1) == "SK_ID_CURR", "valid names untouched")
    check(sch.sanitise_column_name("_ingested_at", 1) == "_ingested_at", "leading underscore preserved")

    # The rename must be traceable back to the source file.
    field = sch.build_struct_field("", "int64", position=0)
    check(field["name"] == "col_0", "renamed field carries the safe name")
    check(field["metadata"].get("source_column") == "", "original header recorded in metadata")
    check("renamed_reason" in field["metadata"], "reason for the rename recorded")

    untouched = sch.build_struct_field("SK_ID_CURR", "int64", position=0)
    check("source_column" not in untouched["metadata"], "unrenamed fields carry no rename metadata")

    # Type overrides key off the ORIGINAL header, not the sanitised one.
    renamed = sch.build_struct_field("AMT BALANCE", "int64", position=2)
    check(renamed["name"] == "AMT_BALANCE", "sanitised to a valid name")
    check(renamed["type"] == sch.DOUBLE, "prefix override still applied via the original header")


def test_committed_schemas_are_uc_safe() -> None:
    """Every committed schema must be a valid Unity Catalog target schema."""
    section("Committed schemas are Unity Catalog safe")

    from credit_risk.config import SCHEMA_DIR

    files = sorted(Path(SCHEMA_DIR).glob("*.json"))
    if not files:
        check(True, "no committed schemas present; skipped")
        return

    bad: list[str] = []
    for path in files:
        for field in json.loads(path.read_text())["fields"]:
            name = field["name"]
            if not name.strip() or sch.sanitise_column_name(name, 0) != name:
                bad.append(f"{path.stem}.{name!r}")

    check(not bad, f"all {len(files)} committed schemas have UC-safe column names"
                   + (f" -- offending: {bad[:5]}" if bad else ""))


def test_schema_round_trip() -> None:
    section("Schema persistence round trip")
    with tempfile.TemporaryDirectory() as tmp:
        original = sch.build_struct_schema({"SK_ID_CURR": "int64", "AMT_CREDIT": "double"})
        sch.write_schema("demo", original, schema_dir=tmp)
        restored = sch.read_schema("demo", schema_dir=tmp)
        check(restored == original, "schema survives write/read unchanged")
        check(sch.schema_columns("demo", schema_dir=tmp) == ["SK_ID_CURR", "AMT_CREDIT"], "column order preserved")

        try:
            sch.read_schema("does_not_exist", schema_dir=tmp)
            check(False, "missing schema raises FileNotFoundError")
        except FileNotFoundError as exc:
            check("generate_schemas" in str(exc), "missing-schema error names the fix")


# --------------------------------------------------------------------------- #
# end-to-end on synthetic files
# --------------------------------------------------------------------------- #

SENTINEL = cfg.DAYS_SENTINEL


def write_fixture(raw: Path) -> None:
    """A miniature Home Credit extract that reproduces the real join graph.

    Deliberately includes: a DAYS_ sentinel, an orphan customer on bureau (which
    should WARN, not FAIL, because application_train holds labelled customers
    only), both flavours of FLAG_ column, and nulls in EXT_SOURCE_1.
    """
    (raw / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET,AMT_INCOME_TOTAL,DAYS_BIRTH,DAYS_EMPLOYED,EXT_SOURCE_1,FLAG_OWN_CAR,FLAG_DOCUMENT_2\n"
        + "".join(
            f"{i},{i % 2},{100000 + i * 10},{-10000 - i},"
            f"{SENTINEL if i % 5 == 0 else -2000 - i},"
            f"{'' if i % 3 == 0 else round(0.1 + i / 100, 3)},"
            f"{'Y' if i % 2 else 'N'},{i % 2}\n"
            for i in range(1, 21)
        )
    )
    (raw / "application_test.csv").write_text(
        "SK_ID_CURR,AMT_INCOME_TOTAL,DAYS_BIRTH\n"
        + "".join(f"{i},{200000 + i},{-11000 - i}\n" for i in range(21, 26))
    )
    (raw / "bureau.csv").write_text(
        "SK_ID_CURR,SK_ID_BUREAU,CREDIT_ACTIVE,DAYS_CREDIT,AMT_CREDIT_SUM,AMT_CREDIT_SUM_DEBT\n"
        + "".join(
            f"{(i % 20) + 1},{5000 + i},{'Active' if i % 3 else 'Closed'},"
            f"{-500 - i},{20000 + i * 100},{i * 50}\n"
            for i in range(40)
        )
        # An orphan customer: belongs to application_test, not train. Expected WARN.
        + "9999,5999,Active,-300,15000,900\n"
    )
    (raw / "bureau_balance.csv").write_text(
        "SK_ID_BUREAU,MONTHS_BALANCE,STATUS\n"
        + "".join(
            f"{5000 + i},{-m},{'C' if m % 4 else str(m % 6)}\n"
            for i in range(40)
            for m in range(1, 7)
        )
    )
    (raw / "previous_application.csv").write_text(
        "SK_ID_PREV,SK_ID_CURR,NAME_CONTRACT_TYPE,AMT_CREDIT,DAYS_DECISION\n"
        + "".join(
            f"{9000 + i},{(i % 20) + 1},"
            f"{'Consumer loans' if i % 2 else 'Cash loans'},{30000 + i * 100},{-800 - i}\n"
            for i in range(30)
        )
    )
    (raw / "credit_card_balance.csv").write_text(
        "SK_ID_PREV,SK_ID_CURR,MONTHS_BALANCE,AMT_BALANCE,AMT_CREDIT_LIMIT_ACTUAL,"
        "AMT_DRAWINGS_ATM_CURRENT,SK_DPD,SK_DPD_DEF\n"
        + "".join(
            f"{9000 + i},{(i % 20) + 1},{-m},{1000 + i * 10 + m},{5000},"
            f"{'' if m % 3 else 200},{m if m > 4 else 0},{m if m > 4 else 0}\n"
            for i in range(30)
            for m in range(1, 7)
        )
    )
    (raw / "POS_CASH_balance.csv").write_text(
        "SK_ID_PREV,SK_ID_CURR,MONTHS_BALANCE,CNT_INSTALMENT,SK_DPD,SK_DPD_DEF\n"
        + "".join(
            f"{9000 + i},{(i % 20) + 1},{-m},{'' if m == 2 else 12},0,0\n"
            for i in range(30)
            for m in range(1, 7)
        )
    )
    (raw / "installments_payments.csv").write_text(
        "SK_ID_PREV,SK_ID_CURR,NUM_INSTALMENT_NUMBER,DAYS_INSTALMENT,DAYS_ENTRY_PAYMENT,"
        "AMT_INSTALMENT,AMT_PAYMENT\n"
        + "".join(
            f"{9000 + i},{(i % 20) + 1},{n},{-100 - n * 30},{-100 - n * 30 + (n % 3)},"
            f"{500.0},{500.0 if n % 4 else 250.0}\n"
            for i in range(30)
            for n in range(1, 5)
        )
    )
    (raw / "HomeCredit_columns_description.csv").write_text(
        'Table,Row,Description,Special\n'
        'application_train,SK_ID_CURR,"ID of loan, unique",\n'
        'bureau,SK_ID_BUREAU,"Recoded ID of previous Credit Bureau credit",hashed\n'
    )


def _env_with(schema_dir: Path, manifest_dir: Path) -> dict[str, str]:
    """Subprocess env pointing the scripts at throwaway artefact directories.

    Without this the suite would overwrite the committed schemas/ with schemas
    derived from 20-row fixtures.
    """
    import os

    return os.environ | {
        "CREDIT_RISK_SCHEMA_DIR": str(schema_dir),
        "CREDIT_RISK_MANIFEST_DIR": str(manifest_dir),
    }


def run_script(
    name: str, raw: Path, env: dict[str, str], extra: list[str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), "--raw-dir", str(raw), *(extra or [])],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_end_to_end() -> None:
    section("End-to-end: generate_schemas + verify_raw on a synthetic extract")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        raw.mkdir()
        write_fixture(raw)

        # Redirect the committed-artefact locations so the test never writes
        # into the real schemas/ or data/manifests/.
        schema_dir = tmp_path / "schemas"
        manifest_dir = tmp_path / "manifests"
        env = _env_with(schema_dir, manifest_dir)

        gen_proc = run_script("generate_schemas.py", raw, env=env)
        check(gen_proc.returncode == 0, f"generate_schemas exits 0 (stderr: {gen_proc.stderr[-300:]})")

        for table in ("application_train", "bureau_balance", "credit_card_balance"):
            check((schema_dir / f"{table}.json").exists(), f"schemas/{table}.json written")

        def fields_of(table: str) -> dict[str, str]:
            path = schema_dir / f"{table}.json"
            if not path.exists():
                return {}
            return {f["name"]: f["type"] for f in json.loads(path.read_text())["fields"]}

        cc = fields_of("credit_card_balance")
        check(cc.get("SK_ID_PREV") == "long", "SK_ID_PREV typed long")
        check(cc.get("MONTHS_BALANCE") == "integer", "MONTHS_BALANCE typed integer")
        check(cc.get("SK_DPD_DEF") == "integer", "SK_DPD_DEF typed integer")
        check(
            cc.get("AMT_DRAWINGS_ATM_CURRENT") == "double",
            "null-bearing AMT_DRAWINGS_ATM_CURRENT widened to double",
        )

        app_fields = fields_of("application_train")
        check(app_fields.get("FLAG_OWN_CAR") == "string", "FLAG_OWN_CAR inferred as string")
        check(
            app_fields.get("FLAG_DOCUMENT_2") in {"integer", "long"},
            "FLAG_DOCUMENT_2 inferred as integer",
        )

        profile_path = manifest_dir / "raw_profile.json"
        check(profile_path.exists(), "raw_profile.json written")
        if profile_path.exists():
            tables = json.loads(profile_path.read_text())["tables"]
            check(tables["application_train"]["n_rows"] == 20, "row count captured from the streaming pass")
            check(tables["application_train"]["null_counts"]["EXT_SOURCE_1"] > 0, "null counts captured")
            check(
                tables["application_train"]["row_count_matches_reference"] is False,
                "row-count mismatch against the reference is detected, not ignored",
            )
            check("sha256" in tables["bureau"], "file digest recorded for reproducibility")

        ver_proc = run_script("verify_raw.py", raw, env=env)
        report_path = manifest_dir / "raw_verification.json"
        check(report_path.exists(), "raw_verification.json written")
        if not report_path.exists():
            return

        by_check: dict[str, list[dict]] = {}
        for finding in json.loads(report_path.read_text())["findings"]:
            by_check.setdefault(finding["check"], []).append(finding)

        check(
            all(f["status"] != "FAIL" for f in by_check.get("key-uniqueness", [])),
            "no spurious key-uniqueness failures on clean fixture data",
        )

        refs = {f["target"]: f for f in by_check.get("referential", [])}
        bb_edge = refs.get("bureau_balance.SK_ID_BUREAU -> bureau.SK_ID_BUREAU")
        check(bb_edge is not None and bb_edge["status"] == PASS, "bureau_balance -> bureau edge resolves")

        orphan_edge = refs.get("bureau.SK_ID_CURR -> application_train.SK_ID_CURR")
        check(
            orphan_edge is not None and orphan_edge["status"] == WARN,
            "orphan against the partial parent WARNs rather than FAILs",
        )

        panel = by_check.get("panel-index", [])
        check(
            len(panel) == 3 and all(f["status"] == PASS for f in panel),
            "MONTHS_BALANCE confirmed negative on all three panels",
        )

        sentinel = {f["target"]: f for f in by_check.get("days-sentinel", [])}
        app_sentinel = sentinel.get("application_train")
        check(
            app_sentinel is not None
            and app_sentinel["status"] == WARN
            and "DAYS_EMPLOYED" in app_sentinel.get("columns", {}),
            f"{SENTINEL} sentinel found and attributed to DAYS_EMPLOYED",
        )
        check(
            sentinel.get("bureau", {}).get("status") == PASS,
            "clean DAYS_ columns report no sentinel",
        )

        # The fixture has 20 rows, not 307,511. The mismatch MUST surface as a
        # FAIL and MUST drive a non-zero exit -- that is the gate working.
        check(
            any(f["status"] == FAIL for f in by_check.get("row-count", [])),
            "reference row-count mismatch raises FAIL",
        )
        check(ver_proc.returncode != 0, "verify_raw exits non-zero when a FAIL is present")


def test_negative_paths() -> None:
    """The gates must fire on bad data, not just stay quiet on good data.

    A check that has only ever been observed passing is not a check.
    """
    section("Negative paths: corrupted fixtures must FAIL")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        raw.mkdir()
        write_fixture(raw)

        # 1. Duplicate the declared natural key on bureau_balance.
        bb = raw / "bureau_balance.csv"
        lines = bb.read_text().splitlines()
        bb.write_text("\n".join([*lines, lines[1]]) + "\n")

        # 2. Break referential integrity past the materiality threshold.
        #    previous_application is a partial parent, so a handful of orphans
        #    is tolerated by design; the gate only fires when the orphan share
        #    of DISTINCT child ids exceeds MAX_ORPHAN_RATE_DISTINCT. The fixture
        #    has 30 real SK_ID_PREV, so 20 fabricated ones puts it at 40%.
        cc = raw / "credit_card_balance.csv"
        cc.write_text(
            cc.read_text()
            + "".join(f"{88000 + i},1,-3,1500,5000,0,0,0\n" for i in range(20))
        )

        # 3. Flip a MONTHS_BALANCE positive, breaking the cohort design.
        pos = raw / "POS_CASH_balance.csv"
        pos.write_text(pos.read_text() + "9000,1,4,12,0,0\n")

        env = _env_with(tmp_path / "schemas", tmp_path / "manifests")
        run_script("generate_schemas.py", raw, env=env)
        run_script("verify_raw.py", raw, env=env)

        report_path = tmp_path / "manifests" / "raw_verification.json"
        check(report_path.exists(), "verification report written for corrupted fixture")
        if not report_path.exists():
            return

        findings = json.loads(report_path.read_text())["findings"]
        failed = {(f["check"], f["target"]) for f in findings if f["status"] == FAIL}

        check(
            any(c == "key-uniqueness" and "bureau_balance" in t for c, t in failed),
            "duplicate natural key detected on bureau_balance",
        )
        check(
            any(c == "referential" and t.startswith("credit_card_balance") for c, t in failed),
            "orphan rate above the materiality threshold FAILs even for a partial parent",
        )
        check(
            any(c == "panel-index" and "pos_cash_balance" in t for c, t in failed),
            "non-negative MONTHS_BALANCE detected on pos_cash_balance",
        )


# --------------------------------------------------------------------------- #

def main() -> int:
    test_cohort_windows()
    test_target_thresholds()
    test_model_specs()
    test_table_registry()
    test_type_resolution()
    test_arrow_mapping()
    test_struct_schema_is_spark_shaped()
    test_column_name_sanitisation()
    test_committed_schemas_are_uc_safe()
    test_schema_round_trip()
    test_end_to_end()
    test_negative_paths()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
