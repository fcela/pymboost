"""Smoke-test runner for book/parity-gallery.md.

Extracts every ``{code-cell} ipython3`` block from the chapter and runs them
sequentially in a single Python process, mimicking how Jupyter Book would
execute the notebook. Prints per-cell status and first-line error diagnostics
so we can iterate quickly without spinning up jupyter-book's full build.

Usage::

    python scripts/smoke_parity_gallery.py
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
DEFAULT_CHAPTER = BOOK / "parity-gallery.md"

CELL_RE = re.compile(
    r"```\{code-cell\} ipython3\s*(?::tags:[^\n]*\n)?\n(.*?)```",
    re.DOTALL,
)


def extract_cells(md: str) -> list[str]:
    cells: list[str] = []
    pos = 0
    while True:
        m = re.search(r"```\{code-cell\} ipython3[^\n]*\n", md[pos:])
        if not m:
            break
        start = pos + m.end()
        # Skip optional :tags: line
        after_tag_m = re.match(r":tags:[^\n]*\n", md[start:])
        if after_tag_m:
            start += after_tag_m.end()
        end_m = re.search(r"\n```\s*$|\n```\n", md[start:], flags=re.MULTILINE)
        if not end_m:
            end_m = re.search(r"\n```", md[start:])
        if not end_m:
            break
        cells.append(md[start : start + end_m.start()])
        pos = start + end_m.end()
    return cells


def main() -> int:
    # Accept one or more chapter paths; default to the parity-gallery chapter.
    if len(sys.argv) > 1:
        chapters = [Path(p).resolve() for p in sys.argv[1:]]
    else:
        chapters = [DEFAULT_CHAPTER]

    failed_total = 0
    cells_total = 0
    for chapter in chapters:
        print(f"\n============ {chapter} ============")
        md = chapter.read_text()
        cells = extract_cells(md)
        print(f"Extracted {len(cells)} code cells")
        failed_total += _run_cells(cells)
        cells_total += len(cells)

    print(f"\nTOTAL: {cells_total - failed_total}/{cells_total} cells ran without error")
    return 0 if failed_total == 0 else 1


def _run_cells(cells: list[str]) -> int:
    ns: dict[str, object] = {"__name__": "__main__"}
    failed = 0
    for i, src in enumerate(cells, start=1):
        head = src.strip().splitlines()[0] if src.strip() else "<empty>"
        print(f"\n── Cell {i:02d} ── {head[:80]}")
        try:
            exec(compile(src, f"<cell-{i}>", "exec"), ns, ns)
        except Exception:
            failed += 1
            traceback.print_exc(limit=4)
    return failed


if __name__ == "__main__":
    sys.exit(main())
