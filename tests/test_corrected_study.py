"""Portable checks for the corrected release; no full-grid methods are rerun."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "CVI_Full_Grid_Study"


def checked(command, directory):
    with tempfile.TemporaryDirectory(prefix="cvi-ci-cache-") as cache:
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=cache)
        result = subprocess.run([sys.executable, *command], cwd=directory,
                                env=environment, capture_output=True, text=True,
                                timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_corrected_release_scope_and_frozen_source_hashes():
    plan = json.loads((STUDY / "published/provenance/combined_plan.json").read_text())
    expected = {("single", n, 1) for n in (10, 50, 100, 200, 500, 1000)}
    expected |= {("multi", 100, m) for m in (10, 50, 100, 200, 500)}
    assert {(s["study"], s["n"], s["m"]) for s in plan["settings"]} == expected
    assert plan["seeds"] == list(range(10))
    assert plan["methods"] == ["proposed", "oe_true", "oe_low", "oe_high", "golden_ratio"]
    assert plan["seconds"] == 1000 and plan["diagnostic_seconds"] == 300
    for filename, expected_hash in plan["source_hashes"].items():
        assert hashlib.sha256((STUDY / "code" / filename).read_bytes()).hexdigest() == expected_hash


def test_published_mc_csv_retains_all_methods_and_measured_attainment():
    with (STUDY / "published/multi/per_run_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 250
    assert {int(r["n"]) for r in rows} == {100}
    assert {int(r["m"]) for r in rows} == {10, 50, 100, 200, 500}
    assert len({(r["m"], r["seed"], r["method"]) for r in rows}) == 250
    inventory = json.loads((STUDY / "published/multi/mc_diagnostic_inventory.json").read_text())
    assert inventory["checkpoints"] == 6327
    assert not inventory["algorithm_failures"]
    assert not inventory["unresolved_joint_diagnostics"]
    assert (ROOT / "JoC_submission/results/mc_table.tex").read_bytes() == (
        STUDY / "published/multi/mc_table.tex").read_bytes()


def test_corrected_reporting_regressions():
    checked(["-m", "unittest", "test_revised_contract", "test_sensitivity_two_instances",
             "test_priority_execution_context", "test_portable_package",
             "test_mc_insertion_bundle", "test_dual_adoption"], STUDY / "reporting")


def test_corrected_worker_numerics_and_resume_guards():
    for script in ("test_checks.py", "tests/test_robustness.py", "tests/test_large_worker.py"):
        checked([script], STUDY / "code")
