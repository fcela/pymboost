from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_CASE = ROOT / "benchmarks" / "benchmark_python_case.py"
R_CASE = ROOT / "benchmarks" / "benchmark_r_case.R"

CASES = [
    "glmboost_gaussian_bols",
    "gamboost_gaussian_bbs_bols",
    "gamboost_gaussian_bmono",
    "gamboost_gaussian_btree",
    "glmboost_binomial_bols",
    "glmboost_poisson_bols",
    "cvrisk_gaussian_bols",
    "cvrisk_gaussian_bmono",
    "cvrisk_gaussian_btree",
]


def _run_hyperfine(case: str, n: int, runs: int, warmup: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        export_path = Path(tmp) / f"{case}.json"
        py_cmd = f"{sys.executable} {PY_CASE} --case {case} --n {n} > /dev/null 2>&1"
        r_cmd = f"Rscript {R_CASE} --case {case} --n {n} > /dev/null 2>&1"
        cmd = [
            "hyperfine",
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
            "--export-json",
            str(export_path),
            "-n",
            "python",
            py_cmd,
            "-n",
            "r",
            r_cmd,
        ]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        payload = json.loads(export_path.read_text())
        results = {item.get("name", item["command"]): item for item in payload["results"]}
        return {
            "python": results["python"],
            "r": results["r"],
        }


def _run_fallback(case: str, n: int, runs: int) -> dict[str, object]:
    def time_command(command: list[str]) -> dict[str, object]:
        raw = []
        for _ in range(runs):
            proc = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            raw.append(float(proc.stderr.strip() or 0.0))
        raise NotImplementedError("fallback mode is not implemented without hyperfine")

    return {
        "python": time_command([sys.executable, str(PY_CASE), "--case", case, "--n", str(n)]),
        "r": time_command(["Rscript", str(R_CASE), "--case", case, "--n", str(n)]),
    }


def _format_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Benchmark Results",
        "",
        "| Case | Python mean (s) | R mean (s) | Python/R speedup |",
        "|---|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        py = payload["results"][case]["python"]
        r = payload["results"][case]["r"]
        ratio = r["mean"] / py["mean"] if py["mean"] > 0.0 else float("inf")
        lines.append(f"| `{case}` | {py['mean']:.6f} | {r['mean']:.6f} | {ratio:.2f}x |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Python pymboost against R mboost with hyperfine.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()

    use_hyperfine = shutil.which("hyperfine") is not None
    if not use_hyperfine:
        raise RuntimeError("hyperfine is required for this benchmark harness")

    unknown = [case for case in args.cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {unknown}")
    results = {case: _run_hyperfine(case, args.n, args.runs, args.warmup) for case in args.cases}
    payload = {
        "config": {"n": args.n, "runs": args.runs, "warmup": args.warmup},
        "cases": list(args.cases),
        "runner": "hyperfine",
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_r_vs_python.json"
    md_path = args.output_dir / "benchmark_r_vs_python.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_format_markdown(payload))

    print(json.dumps(payload, indent=2))
    print()
    try:
        print(md_path.relative_to(ROOT))
    except ValueError:
        print(md_path)


if __name__ == "__main__":
    main()
