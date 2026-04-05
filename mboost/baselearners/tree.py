from __future__ import annotations

from .base import BaseLearnerSpec


def btree(
    *names: str,
    by: str | None = None,
    maxdepth: int = 1,
    minsplit: int = 10,
    minbucket: int = 4,
) -> BaseLearnerSpec:
    if len(names) == 0:
        raise ValueError("btree requires at least one feature")
    if maxdepth < 1:
        raise ValueError("maxdepth must be at least 1")
    if minsplit < 2:
        raise ValueError("minsplit must be at least 2")
    if minbucket < 1:
        raise ValueError("minbucket must be at least 1")
    return BaseLearnerSpec(
        name=", ".join(names),
        kind="tree",
        center=False,
        intercept=False,
        feature_names=tuple(names),
        by=by,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
    )
