from __future__ import annotations

from mboost.api.glmboost import GLMBoostModel


def coef(model: GLMBoostModel):
    return model.coefficients_


def risk(model: GLMBoostModel):
    return model.risk_.copy()


def fitted(model: GLMBoostModel):
    return model.fitted_.copy()


def selected(model: GLMBoostModel):
    return list(model.selected)
