from __future__ import annotations

from .base import BaseLearnerSpec


def bbs(
    name: str,
    *,
    df: int = 4,
    knots: int | None = 20,
    lambda_: float | None = None,
    degree: int = 3,
    differences: int = 2,
    center: bool = False,
    by: str | None = None,
) -> BaseLearnerSpec:
    if df <= 0:
        raise ValueError("df must be positive")
    if lambda_ is not None and lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if differences < 1:
        raise ValueError("differences must be positive")
    return BaseLearnerSpec(
        name=name,
        kind="spline",
        penalty=lambda_,
        center=center,
        by=by,
        df=df,
        knots=knots,
        degree=degree,
        differences=differences,
    )


def bmono(
    name: str,
    *,
    constraint: str = "increasing",
    type: str = "quad.prog",
    by: str | None = None,
    knots: int = 20,
    degree: int = 3,
    differences: int = 2,
    df: int = 4,
    lambda_: float | None = None,
    niter: int = 10,
    intercept: bool = True,
    boundary_constraints: bool = False,
) -> BaseLearnerSpec:
    if df <= 0:
        raise ValueError("df must be positive")
    if lambda_ is not None and lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if differences < 1:
        raise ValueError("differences must be positive")
    if constraint not in {
        "increasing",
        "decreasing",
        "convex",
        "concave",
        "positive",
        "negative",
    }:
        raise ValueError(f"unsupported constraint: {constraint}")
    if type not in {"quad.prog", "iterative"}:
        raise ValueError(f"unsupported type: {type}")
    if niter <= 0:
        raise ValueError("niter must be positive")
    if boundary_constraints and type != "quad.prog":
        raise NotImplementedError("boundary_constraints is only supported for type='quad.prog' for now")
    return BaseLearnerSpec(
        name=name,
        kind="mono_spline",
        penalty=lambda_,
        center=False,
        intercept=intercept,
        by=by,
        target_level=constraint,
        df=df,
        knots=knots,
        degree=degree,
        differences=differences,
        solver_type=type,
        niter=niter,
    )
