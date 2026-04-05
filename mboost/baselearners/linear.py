from __future__ import annotations

import numpy as np

from .base import BaseLearnerSpec


def bols(
    name: str,
    *,
    df=None,
    lambda_: float = 0.0,
    intercept: bool = True,
    center: bool = False,
    by: str | None = None,
) -> BaseLearnerSpec:
    if df is not None and df <= 0:
        raise ValueError("df must be positive")
    if lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative")
    if df is not None and lambda_ != 0.0:
        raise ValueError("specify either df or lambda_, not both")
    return BaseLearnerSpec(
        name=name,
        kind="linear",
        penalty=None if df is not None else lambda_,
        center=center,
        intercept=intercept,
        by=by,
        df=df,
    )


def factor_dummy(
    name: str,
    *,
    target_level,
    lambda_: float = 0.0,
    center: bool = True,
    by: str | None = None,
) -> BaseLearnerSpec:
    if lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative")
    return BaseLearnerSpec(
        name=name,
        kind="factor_dummy",
        penalty=lambda_,
        center=center,
        intercept=False,
        by=by,
        target_level=np.asarray(target_level).item(),
    )


def brandom(
    name: str,
    *,
    df: float = 4,
    lambda_: float | None = None,
    by: str | None = None,
) -> BaseLearnerSpec:
    if lambda_ is not None and lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative")
    if df <= 0:
        raise ValueError("df must be positive")
    return BaseLearnerSpec(
        name=name,
        kind="random",
        penalty=lambda_,
        center=False,
        intercept=False,
        by=by,
        df=int(df) if float(df).is_integer() else df,
    )
