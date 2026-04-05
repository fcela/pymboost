from __future__ import annotations

from dataclasses import dataclass

from .glmboost import GLMBoostModel, _parse_call, _split_formula, glmboost


@dataclass(frozen=True)
class TreeControls:
    minsplit: int = 10
    minbucket: int = 4
    maxdepth: int = 2


def _rewrite_formula_as_tree(formula: str, *, data, tree_controls: TreeControls) -> str:
    response, terms = _split_formula(formula, data=data)
    feature_terms: list[str] = []
    by_term: str | None = None
    for term in terms:
        if term.endswith(")"):
            fn_name, args, kwargs = _parse_call(term)
            if fn_name != "btree":
                raise NotImplementedError("blackboost currently supports only plain feature terms or explicit btree(...) terms")
            if by_term is not None and kwargs.get("by") is not None:
                raise NotImplementedError("blackboost currently supports at most one by= modifier")
            feature_terms.extend(args)
            if kwargs.get("by") is not None:
                by_term = str(kwargs["by"])
            continue
        feature_terms.append(term)

    if not feature_terms:
        raise ValueError("blackboost requires at least one feature term")

    deduped: list[str] = []
    for term in feature_terms:
        if term not in deduped:
            deduped.append(term)

    tree_term = ", ".join(deduped)
    by_clause = f", by={by_term}" if by_term is not None else ""
    return (
        f"{response} ~ btree({tree_term}{by_clause}, "
        f"maxdepth={tree_controls.maxdepth}, minsplit={tree_controls.minsplit}, minbucket={tree_controls.minbucket})"
    )


def blackboost(
    formula: str,
    *,
    tree_controls: TreeControls | None = None,
    **kwargs,
) -> GLMBoostModel:
    controls = TreeControls() if tree_controls is None else tree_controls
    rewritten = _rewrite_formula_as_tree(formula, data=kwargs.get("data"), tree_controls=controls)
    return glmboost(rewritten, **kwargs)
