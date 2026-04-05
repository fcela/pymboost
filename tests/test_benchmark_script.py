from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_script_writes_json_and_markdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "benchmark_r_vs_python.py"

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
            "cvrisk_gaussian_bols",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout.split("\n\n", 1)[0])
    assert payload["config"]["n"] == 80
    assert payload["cases"] == ["glmboost_gaussian_bols", "cvrisk_gaussian_bols"]
    assert payload["runner"] == "hyperfine"
    assert "glmboost_gaussian_bols" in payload["results"]
    assert (tmp_path / "benchmark_r_vs_python.json").exists()
    assert (tmp_path / "benchmark_r_vs_python.md").exists()
