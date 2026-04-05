from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_warm_benchmark_script_writes_json_and_markdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "benchmark_warm_inprocess.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--n",
                "80",
                "--runs",
                "2",
                "--output-dir",
                str(tmp_path),
                "--cases",
                "glmboost_gaussian_bols",
                "gamboost_gaussian_btree",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if "llvmlite" in exc.stderr or "Could not find/load shared object file" in exc.stderr:
            pytest.skip("warm benchmark subprocess could not load llvmlite in this test environment")
        raise

    payload = json.loads(result.stdout.split("\n\n", 1)[0])
    assert payload["config"]["n"] == 80
    assert payload["cases"] == ["glmboost_gaussian_bols", "gamboost_gaussian_btree"]
    assert set(payload["python"]) == set(payload["r"])
    assert (tmp_path / "benchmark_warm_inprocess.json").exists()
    assert (tmp_path / "benchmark_warm_inprocess.md").exists()
