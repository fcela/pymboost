from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_jupyter_book_tutorial_exists_and_is_explicit():
    chapter = ROOT / "book" / "hands-on-tutorial.md"
    text = chapter.read_text()

    assert chapter.exists()
    assert "Hands-on Tutorial" in text
    assert "glmboost(" in text
    assert "gamboost(" in text
    assert "cvrisk(" in text
    assert "class OurQuantReg(Family)" in text
    assert "run_example(" not in text
    assert "from examples." not in text
