from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_docs_example_scripts_run() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = [
        root / "examples" / "getting_started.py",
        root / "examples" / "cars_gamboost.py",
        root / "examples" / "cars_cvrisk.py",
        root / "examples" / "blackboost_demo.py",
        root / "examples" / "additive_models.py",
        root / "examples" / "bodyfat_examples.py",
        root / "examples" / "chart_gallery.py",
        root / "examples" / "monotone_splines.py",
    ]

    for script in scripts:
        env = os.environ.copy()
        env["NUMBA_DISABLE_JIT"] = "1"
        env["PYTHONPATH"] = str(root)
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            if "Could not find/load shared object file" in exc.stderr:
                pytest.skip("llvmlite is not loadable in subprocesses in this environment")
            raise
        assert result.stdout.strip()
