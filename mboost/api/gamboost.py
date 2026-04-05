from __future__ import annotations

from .glmboost import GLMBoostModel, _split_formula, glmboost


def _rewrite_formula_with_dfbase(formula: str, data, dfbase: int) -> str:
    response, terms = _split_formula(formula, data=data)
    rewritten = []
    for term in terms:
        if "(" in term or ")" in term:
            rewritten.append(term)
        else:
            rewritten.append(f"bbs({term}, df={dfbase}, knots=20)")
    return f"{response} ~ {' + '.join(rewritten)}"


def gamboost(
    formula: str,
    *,
    dfbase: int = 4,
    **kwargs,
) -> GLMBoostModel:
    data = kwargs.get("data")
    return glmboost(_rewrite_formula_with_dfbase(formula, data=data, dfbase=dfbase), **kwargs)
