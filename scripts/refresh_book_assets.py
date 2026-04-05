"""Refresh cached R reference outputs used by the pymboost book.

The book chapters compare Python results against R `mboost` side-by-side.
Running rpy2 at book-build time is slow and fragile. Instead, this script
runs every required R call once, locally, and caches the numeric outputs
as JSON under ``book/_static/r_cache/``. Chapters then load these JSON
blobs via ``book_utils.load_cached_r_json``.

Usage::

    python scripts/refresh_book_assets.py              # refresh everything
    python scripts/refresh_book_assets.py --only glmboost_bodyfat  # one target
    python scripts/refresh_book_assets.py --list       # list targets

Each cached payload is a plain dict with JSON-serialisable values (lists of
floats or strings). Chapters should treat the cache as read-only.

This file is intentionally a skeleton: new case studies add themselves by
appending a ``@register("name")`` function that returns a dict. Keep each
function small, deterministic, and independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

# Make the book_utils module importable without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "book"))

import book_utils  # noqa: E402


Target = Callable[[], dict]
_REGISTRY: dict[str, Target] = {}


def register(name: str) -> Callable[[Target], Target]:
    def decorator(fn: Target) -> Target:
        if name in _REGISTRY:
            raise ValueError(f"duplicate refresh target: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


@register("bodyfat_glmboost")
def _bodyfat_glmboost() -> dict:
    """Linear bodyfat model, matching `glmboost.md`."""
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    bodyfat = pl.read_csv(book_utils.data_dir() / "bodyfat.csv")
    book_utils.r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
    ro.r("bodyfat <- bodyfat_py")
    ro.r(
        """
        glm1_r <- glmboost(
          DEXfat ~ hipcirc + kneebreadth + anthro3a,
          data = bodyfat
        )
        glm2_r <- glmboost(DEXfat ~ ., data = bodyfat)
        """
    )
    glm1 = book_utils.r_named_vector("coef(glm1_r, off2int = TRUE)", value_name="coefficient")
    glm2 = book_utils.r_named_vector('coef(glm2_r, which = "")', value_name="coefficient")
    fitted_glm1 = book_utils.r_numeric("as.numeric(predict(glm1_r, type = 'link'))")
    return {
        "glm1_coef": glm1.to_dict(orient="list"),
        "glm2_coef": glm2.to_dict(orient="list"),
        "glm1_fitted": fitted_glm1.tolist(),
    }


@register("baselearners_bols")
def _baselearners_bols() -> dict:
    """Bodyfat linear model with explicit bols terms, for book/baselearners/bols.md."""
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    bodyfat = pl.read_csv(book_utils.data_dir() / "bodyfat.csv")
    book_utils.r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
    ro.r("bodyfat <- bodyfat_py")
    ro.r(
        """
        bols_r <- mboost(
          DEXfat ~ bols(hipcirc) + bols(kneebreadth) + bols(anthro3a),
          data = bodyfat,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        """
    )
    # coef(m) on an mboost() fit with explicit bols(...) terms returns a list of
    # 2-vectors named (Intercept, slope) per learner. Flatten to long form.
    term_labels = list(ro.r("names(coef(bols_r))"))
    coef_rows: list[dict] = []
    for label in term_labels:
        vec = ro.r(f"coef(bols_r)[['{label}']]")
        for component, value in zip(list(vec.names), [float(v) for v in vec]):
            coef_rows.append({"term": label, "component": component, "value": value})
    offset = float(ro.r("as.numeric(bols_r$offset)")[0])
    fitted = book_utils.r_numeric("as.numeric(predict(bols_r))").tolist()
    return {
        "coefficients": coef_rows,
        "offset": offset,
        "fitted": fitted,
    }


@register("baselearners_bbs")
def _baselearners_bbs() -> dict:
    """Bodyfat additive spline model, for book/baselearners/bbs.md."""
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    bodyfat = pl.read_csv(book_utils.data_dir() / "bodyfat.csv")
    book_utils.r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
    ro.r("bodyfat <- bodyfat_py")
    ro.r(
        """
        bbs_r <- mboost(
          DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a),
          data = bodyfat,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        """
    )
    fitted = book_utils.r_numeric("as.numeric(predict(bbs_r))").tolist()

    partial_curves: dict[str, dict] = {}
    median_hipcirc = float(np.median(bodyfat["hipcirc"].to_numpy()))
    median_knee = float(np.median(bodyfat["kneebreadth"].to_numpy()))
    median_anthro = float(np.median(bodyfat["anthro3a"].to_numpy()))
    for index, feature in enumerate(["hipcirc", "kneebreadth", "anthro3a"], start=1):
        lo = float(bodyfat[feature].min())
        hi = float(bodyfat[feature].max())
        grid = np.linspace(lo, hi, 120)
        overlay = bodyfat.to_pandas().iloc[[0] * grid.size].copy()
        overlay["hipcirc"] = median_hipcirc
        overlay["kneebreadth"] = median_knee
        overlay["anthro3a"] = median_anthro
        overlay[feature] = grid
        book_utils.r_assign_dataframe("overlay_df", overlay)
        curve = book_utils.r_numeric(
            f"as.numeric(predict(bbs_r, newdata = overlay_df, which = {index}))"
        ).tolist()
        partial_curves[feature] = {"x": grid.tolist(), "effect": curve}
    return {"fitted": fitted, "partial_curves": partial_curves}


@register("baselearners_bmono")
def _baselearners_bmono() -> dict:
    """Bodyfat monotone spline model, for book/baselearners/bmono.md."""
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    bodyfat = pl.read_csv(book_utils.data_dir() / "bodyfat.csv")
    book_utils.r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
    ro.r("bodyfat <- bodyfat_py")
    ro.r(
        """
        bmono_r <- mboost(
          DEXfat ~ bmono(hipcirc, constraint='increasing') + bbs(kneebreadth) + bbs(anthro3a),
          data = bodyfat,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        """
    )
    fitted = book_utils.r_numeric("as.numeric(predict(bmono_r))").tolist()

    partial_curves: dict[str, dict] = {}
    median_hipcirc = float(np.median(bodyfat["hipcirc"].to_numpy()))
    median_knee = float(np.median(bodyfat["kneebreadth"].to_numpy()))
    median_anthro = float(np.median(bodyfat["anthro3a"].to_numpy()))
    lo = float(bodyfat["hipcirc"].min())
    hi = float(bodyfat["hipcirc"].max())
    grid = np.linspace(lo, hi, 120)
    overlay = bodyfat.to_pandas().iloc[[0] * grid.size].copy()
    overlay["hipcirc"] = grid
    overlay["kneebreadth"] = median_knee
    overlay["anthro3a"] = median_anthro
    book_utils.r_assign_dataframe("overlay_df", overlay)
    curve = book_utils.r_numeric(
        "as.numeric(predict(bmono_r, newdata = overlay_df, which = 1))"
    ).tolist()
    partial_curves["hipcirc"] = {"x": grid.tolist(), "effect": curve}
    return {"fitted": fitted, "partial_curves": partial_curves}


@register("baselearners_brandom")
def _baselearners_brandom() -> dict:
    """Fixed+random worked example, for book/baselearners/brandom.md.

    Generates the panel in R (to match R's RNG) and also returns the
    generated panel so the Python chapter can fit on the same data.
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    ro.r(
        """
        set.seed(11)
        n_groups <- 15; n_per <- 30
        group_labels <- factor(rep(sprintf("%02d", 0:(n_groups-1)), each = n_per))
        true_u <- rnorm(n_groups, sd = 0.6)
        z <- rnorm(n_groups * n_per)
        y <- 0.4 * z + true_u[as.integer(group_labels)] + rnorm(n_groups * n_per, sd = 0.25)
        dat <- data.frame(group = group_labels, z = z, y = y)
        brandom_r <- mboost(
          y ~ bols(z) + brandom(group),
          data = dat,
          control = boost_control(mstop = 300, nu = 0.1)
        )
        """
    )
    group_vec = list(ro.r("as.character(dat$group)"))
    z_vec = [float(v) for v in ro.r("as.numeric(dat$z)")]
    y_vec = [float(v) for v in ro.r("as.numeric(dat$y)")]
    group_effects = [float(v) for v in ro.r("coef(brandom_r)[['brandom(group)']]")]
    fitted = book_utils.r_numeric("as.numeric(predict(brandom_r))").tolist()
    bols_z_coef = [float(v) for v in ro.r("coef(brandom_r)[['bols(z)']]")]
    return {
        "panel": {"group": group_vec, "z": z_vec, "y": y_vec},
        "group_effects": group_effects,
        "bols_z_coef": bols_z_coef,  # [intercept, slope]
        "fitted": fitted,
    }


@register("gallery_glmboost_paths")
def _gallery_glmboost_paths() -> dict:
    """bodyfat glmboost: full coefficient path + AIC curve + varimp.

    Backs the Gaussian-canon figure in ``book/parity-gallery.md``. Uses
    ``glmboost(DEXfat ~ ., data = bodyfat)`` with ``mstop = 120`` to give
    the AIC curve room to turn over; the Hofner tutorial figure uses the
    same setup. Both the per-iteration coefficient path and the AIC path
    are returned in long form so the polars consumer in the gallery can
    feed them directly to the Altair factories.
    """
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    book_utils.r_load_library("TH.data")
    ro.r("data(bodyfat)")
    ro.r(
        """
        glm_gallery_r <- glmboost(
          DEXfat ~ .,
          data = bodyfat,
          control = boost_control(mstop = 120, nu = 0.1)
        )
        """
    )
    # Coefficient path: coef(m, which='', aggregate='cumsum') is a list with
    # one 1 x mstop matrix per term. Flatten to long form.
    ro.r('cp <- coef(glm_gallery_r, which = "", aggregate = "cumsum")')
    term_names = [str(x) for x in ro.r("names(cp)")]
    path_rows: list[dict] = []
    for index, term in enumerate(term_names, start=1):
        series = np.asarray(ro.r(f"as.numeric(cp[[{index}]])"), dtype=np.float64)
        for it, value in enumerate(series.tolist(), start=1):
            path_rows.append({"term": term, "iteration": it, "coefficient": value})

    # AIC curve. The gbAIC object is a scalar (value at the optimal mstop)
    # with the full curve stashed in ``attr(a, "AIC")`` — length equal to
    # mstop. Pulling the scalar would give us just one number, so we reach
    # for the attribute explicitly.
    ro.r('aic_gallery_r <- AIC(glm_gallery_r, method = "corrected")')
    aic_values = np.asarray(
        ro.r('as.numeric(attr(aic_gallery_r, "AIC"))'), dtype=np.float64
    ).tolist()
    aic_df = np.asarray(
        ro.r('as.numeric(attr(aic_gallery_r, "df"))'), dtype=np.float64
    ).tolist()
    aic_selected = int(ro.r("mstop(aic_gallery_r)")[0])

    # Variable importance — a named numeric vector
    vi_values = np.asarray(ro.r("as.numeric(varimp(glm_gallery_r))"), dtype=np.float64).tolist()
    vi_terms = [str(x) for x in ro.r("names(varimp(glm_gallery_r))")]

    # Final risk path
    risk_path = np.asarray(ro.r("as.numeric(risk(glm_gallery_r))"), dtype=np.float64).tolist()

    return {
        "mstop": 120,
        "coef_path": path_rows,
        "aic": aic_values,
        "aic_df": aic_df,
        "aic_selected": aic_selected,
        "varimp": [
            {"term": t, "reduction": v} for t, v in zip(vi_terms, vi_values)
        ],
        "risk": risk_path,
    }


@register("gallery_gamboost_cvrisk")
def _gallery_gamboost_cvrisk() -> dict:
    """bodyfat gamboost: cvrisk fold matrix + risk path.

    The seed and fold definition are pinned so the Python chapter can feed
    the *same* fold matrix into ``mboost.cvrisk`` and get a point-for-point
    comparable fold overlay. The resulting cached payload carries:

    * ``mstop``: the grid length
    * ``fold_weights``: a ``(B, n)`` matrix of fold indicator weights so
      Python can reconstruct identical folds
    * ``fold_risks``: a ``(B, mstop)`` matrix of held-out risks per fold
    * ``mean_risk``: length-``mstop`` across-fold mean
    * ``selected_mstop``: R's argmin of the mean curve
    """
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    book_utils.r_load_library("TH.data")
    ro.r("data(bodyfat)")
    ro.r(
        """
        set.seed(2026)
        gam_cv_r <- gamboost(
          DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a),
          data = bodyfat,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        fold_w <- cv(model.weights(gam_cv_r), type = "kfold", B = 10)
        cv_mat <- cvrisk(gam_cv_r, folds = fold_w, grid = 1:100)
        """
    )
    mstop = 100
    fold_weights = np.asarray(ro.r("as.matrix(fold_w)"), dtype=np.float64)
    fold_risks = np.asarray(ro.r("as.matrix(cv_mat)"), dtype=np.float64)  # (B, mstop)
    mean_risk = fold_risks.mean(axis=0)
    selected_mstop = int(ro.r("mstop(cv_mat)")[0])
    risk_path = np.asarray(ro.r("as.numeric(risk(gam_cv_r))"), dtype=np.float64).tolist()
    return {
        "mstop": mstop,
        "fold_weights": fold_weights.tolist(),
        "fold_risks": fold_risks.tolist(),
        "mean_risk": mean_risk.tolist(),
        "selected_mstop": selected_mstop,
        "risk": risk_path,
    }


@register("gallery_synthetic_glm")
def _gallery_synthetic_glm() -> dict:
    """Clean synthetic parity reference for Binomial and Poisson families.

    The parity test suite already demonstrates machine-precision parity for
    single-term Binomial and Poisson fits; this target caches the R side of
    a slightly larger multi-term version (still deterministic, still
    numerically stable) so the parity gallery can lead with a
    machine-precision figure before moving on to the applied wpbc / birds
    examples which accumulate small-scale drift.
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    ro.r(
        """
        set.seed(42)
        n <- 200
        x1 <- runif(n, -2, 2)
        x2 <- runif(n, -2, 2)
        x3 <- runif(n, -2, 2)
        # Binomial: inverse-logit link. R's mboost Binomial demands a
        # factor response (the 2nd level being the "success"), whereas
        # Python's Binomial is happy with numeric 0/1. We fit R on the
        # factor and store the integer y_binomial for Python.
        eta_b <- 0.3 + 0.9 * x1 - 0.5 * x2
        p <- 1 / (1 + exp(-eta_b))
        y_b <- rbinom(n, 1, p)
        bin_df <- data.frame(y = factor(y_b, levels = c(0, 1)),
                             x1 = x1, x2 = x2, x3 = x3)
        bin_r <- glmboost(
          y ~ x1 + x2 + x3,
          data = bin_df,
          family = Binomial(),
          control = boost_control(mstop = 100, nu = 0.1)
        )

        # Poisson: log link
        eta_p <- 0.5 + 0.4 * x1 - 0.25 * x2
        lam <- exp(eta_p)
        y_p <- rpois(n, lam)
        pois_df <- data.frame(y = y_p, x1 = x1, x2 = x2, x3 = x3)
        pois_r <- glmboost(
          y ~ x1 + x2 + x3,
          data = pois_df,
          family = Poisson(),
          control = boost_control(mstop = 100, nu = 0.1)
        )
        """
    )
    bin_coef_vec = ro.r('coef(bin_r, off2int = TRUE, which = "")')
    pois_coef_vec = ro.r('coef(pois_r, off2int = TRUE, which = "")')
    return {
        "panel": {
            "x1": [float(v) for v in ro.r("as.numeric(bin_df$x1)")],
            "x2": [float(v) for v in ro.r("as.numeric(bin_df$x2)")],
            "x3": [float(v) for v in ro.r("as.numeric(bin_df$x3)")],
            # factor(y, levels=c(0,1)) — as.integer returns 1/2; pull raw numeric.
            "y_binomial": [int(v) for v in ro.r("as.integer(as.character(bin_df$y))")],
            "y_poisson": [int(v) for v in ro.r("as.integer(pois_df$y)")],
        },
        "binomial": {
            "coefficients": [
                {"term": str(t), "coefficient": float(v)}
                for t, v in zip(list(bin_coef_vec.names), [float(x) for x in bin_coef_vec])
            ],
            "fitted": np.asarray(
                ro.r("as.numeric(fitted(bin_r))"), dtype=np.float64
            ).tolist(),
            "link": np.asarray(
                ro.r('as.numeric(predict(bin_r, type = "link"))'), dtype=np.float64
            ).tolist(),
            "prob": np.asarray(
                ro.r('as.numeric(predict(bin_r, type = "response"))'), dtype=np.float64
            ).tolist(),
            "risk": np.asarray(
                ro.r("as.numeric(risk(bin_r))"), dtype=np.float64
            ).tolist(),
        },
        "poisson": {
            "coefficients": [
                {"term": str(t), "coefficient": float(v)}
                for t, v in zip(list(pois_coef_vec.names), [float(x) for x in pois_coef_vec])
            ],
            "fitted": np.asarray(
                ro.r("as.numeric(fitted(pois_r))"), dtype=np.float64
            ).tolist(),
            "link": np.asarray(
                ro.r('as.numeric(predict(pois_r, type = "link"))'), dtype=np.float64
            ).tolist(),
            "rate": np.asarray(
                ro.r('as.numeric(predict(pois_r, type = "response"))'), dtype=np.float64
            ).tolist(),
            "risk": np.asarray(
                ro.r("as.numeric(risk(pois_r))"), dtype=np.float64
            ).tolist(),
        },
    }


@register("gallery_wpbc_binomial")
def _gallery_wpbc_binomial() -> dict:
    """wpbc Wisconsin breast cancer: Binomial glmboost parity.

    The tutorial figure compares class-probability predictions and the
    coefficient path on the recurrence indicator. We pick a stable subset
    of predictors rather than the full 32-column design so the figure is
    legible; the parity claim is about numerical agreement, not about
    clinical interpretability.
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    book_utils.r_load_library("TH.data")
    ro.r("data(wpbc)")
    ro.r(
        """
        wpbc_clean <- na.omit(wpbc[, c('status', 'mean_radius', 'mean_texture',
                                       'mean_smoothness', 'mean_concavity',
                                       'worst_radius', 'worst_texture', 'tsize')])
        wpbc_clean$status <- factor(wpbc_clean$status, levels = c('N','R'))
        glm_bin_r <- glmboost(
          status ~ .,
          data = wpbc_clean,
          family = Binomial(),
          control = boost_control(mstop = 200, nu = 0.1)
        )
        """
    )
    coef_vec = ro.r('coef(glm_bin_r, off2int = TRUE, which = "")')
    coef_rows = [
        {"term": str(t), "coefficient": float(v)}
        for t, v in zip(list(coef_vec.names), [float(x) for x in coef_vec])
    ]
    # Class-probability predictions on the training frame
    prob = np.asarray(ro.r('as.numeric(predict(glm_bin_r, type = "response"))'), dtype=np.float64).tolist()
    link = np.asarray(ro.r('as.numeric(predict(glm_bin_r, type = "link"))'), dtype=np.float64).tolist()
    status = [str(x) for x in ro.r("as.character(wpbc_clean$status)")]
    risk_path = np.asarray(ro.r("as.numeric(risk(glm_bin_r))"), dtype=np.float64).tolist()

    # Coefficient path
    ro.r('cp_bin <- coef(glm_bin_r, which = "", aggregate = "cumsum")')
    term_names = [str(x) for x in ro.r("names(cp_bin)")]
    path_rows: list[dict] = []
    for index, term in enumerate(term_names, start=1):
        series = np.asarray(ro.r(f"as.numeric(cp_bin[[{index}]])"), dtype=np.float64)
        for it, value in enumerate(series.tolist(), start=1):
            path_rows.append({"term": term, "iteration": it, "coefficient": value})

    # Save the cleaned panel so Python fits the same frame. The panel is
    # assigned to R's globalenv first so we can access its columns via
    # ``ro.r("design$column")`` in the dict comprehension below.
    ro.r(
        """
        design <- data.frame(
          status = as.character(wpbc_clean$status),
          wpbc_clean[, setdiff(names(wpbc_clean), 'status')]
        )
        """
    )
    column_names = [str(x) for x in ro.r("names(design)")]
    panel: dict[str, list] = {}
    for name in column_names:
        if name == "status":
            panel[name] = [str(x) for x in ro.r('as.character(design$status)')]
        else:
            panel[name] = [float(x) for x in ro.r(f"as.numeric(design${name})")]
    return {
        "panel": panel,
        "coefficients": coef_rows,
        "coef_path": path_rows,
        "prob_response": prob,
        "link_response": link,
        "status": status,
        "risk": risk_path,
    }


@register("gallery_birds_poisson")
def _gallery_birds_poisson() -> dict:
    """birds count data: Poisson glmboost parity.

    ``SG5`` in the ``birds`` data is an integer count used throughout
    the mboost illustrations vignette. We fit a log-link Poisson glmboost
    against a handful of habitat predictors.
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    book_utils.r_load_library("TH.data")
    ro.r("data(birds)")
    ro.r(
        """
        birds_use <- birds[, c('SG5','GST','DBH','AOT','DWC','LOG')]
        glm_pois_r <- glmboost(
          SG5 ~ .,
          data = birds_use,
          family = Poisson(),
          control = boost_control(mstop = 300, nu = 0.1)
        )
        """
    )
    coef_vec = ro.r('coef(glm_pois_r, off2int = TRUE, which = "")')
    coef_rows = [
        {"term": str(t), "coefficient": float(v)}
        for t, v in zip(list(coef_vec.names), [float(x) for x in coef_vec])
    ]
    link = np.asarray(ro.r('as.numeric(predict(glm_pois_r, type = "link"))'), dtype=np.float64).tolist()
    rate = np.asarray(ro.r('as.numeric(predict(glm_pois_r, type = "response"))'), dtype=np.float64).tolist()
    risk_path = np.asarray(ro.r("as.numeric(risk(glm_pois_r))"), dtype=np.float64).tolist()

    panel_cols = ["SG5", "GST", "DBH", "AOT", "DWC", "LOG"]
    panel = {
        name: [float(x) for x in ro.r(f"as.numeric(birds_use${name})")] for name in panel_cols
    }
    return {
        "panel": panel,
        "coefficients": coef_rows,
        "link_response": link,
        "rate_response": rate,
        "risk": risk_path,
    }


@register("gallery_bodyfat_quantile")
def _gallery_bodyfat_quantile() -> dict:
    """bodyfat quantile regression at several tau levels."""
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    book_utils.r_load_library("TH.data")
    ro.r("data(bodyfat)")
    tau_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    all_predictions: dict[str, list[float]] = {}
    for tau in tau_values:
        ro.r(
            f"""
            glm_q_r <- glmboost(
              DEXfat ~ hipcirc + kneebreadth + anthro3a,
              data = bodyfat,
              family = QuantReg(tau = {tau}),
              control = boost_control(mstop = 200, nu = 0.1)
            )
            """
        )
        all_predictions[f"tau_{tau}"] = np.asarray(
            ro.r('as.numeric(predict(glm_q_r, type = "link"))'), dtype=np.float64
        ).tolist()
    return {
        "tau_values": tau_values,
        "predictions": all_predictions,
    }


@register("gallery_btree_grid")
def _gallery_btree_grid() -> dict:
    """btree prediction surface on a fixed x-grid.

    btree's R reference is an ``rpart`` / CART stump under ``blackboost``.
    We fit the same synthetic step-function target R and Python both see,
    emit the reference prediction grid, and let Python compare shape on a
    shared grid rather than coefficient-level parity (which is impossible
    for tree ensembles with different backends).
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    ro.r(
        """
        set.seed(20260406)
        n <- 400
        x <- sort(runif(n, -1, 1))
        y <- ifelse(x < -0.3, -1,
                    ifelse(x < 0.4, 0.5, -0.5)) + rnorm(n, sd = 0.15)
        tree_df <- data.frame(x = x, y = y)
        tree_r <- mboost(
          y ~ btree(x),
          data = tree_df,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        grid_x <- seq(-1, 1, length.out = 400)
        grid_pred <- as.numeric(predict(tree_r, newdata = data.frame(x = grid_x)))
        """
    )
    return {
        "panel": {
            "x": [float(v) for v in ro.r("as.numeric(tree_df$x)")],
            "y": [float(v) for v in ro.r("as.numeric(tree_df$y)")],
        },
        "grid": {
            "x": [float(v) for v in ro.r("as.numeric(grid_x)")],
            "prediction": [float(v) for v in ro.r("as.numeric(grid_pred)")],
        },
    }


@register("gallery_blackboost_grid")
def _gallery_blackboost_grid() -> dict:
    """blackboost joint surface on a 2-D grid.

    Two-feature interaction target ``y = sin(pi x1) * x2 + noise``. We fit
    ``blackboost`` in R and evaluate on a fixed 25x25 grid so the Python
    chapter can compare its CART-wrapper output against the R `partykit`
    ensemble on the same points.
    """
    import numpy as np
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    ro.r(
        """
        set.seed(20260406)
        n <- 500
        x1 <- runif(n, -1, 1)
        x2 <- runif(n, -1, 1)
        y <- sin(pi * x1) * x2 + rnorm(n, sd = 0.2)
        black_df <- data.frame(y = y, x1 = x1, x2 = x2)
        black_r <- blackboost(
          y ~ x1 + x2,
          data = black_df,
          control = boost_control(mstop = 100, nu = 0.1)
        )
        g1 <- seq(-1, 1, length.out = 25)
        g2 <- seq(-1, 1, length.out = 25)
        grid_df <- expand.grid(x1 = g1, x2 = g2)
        grid_pred <- as.numeric(predict(black_r, newdata = grid_df))
        """
    )
    return {
        "panel": {
            "x1": [float(v) for v in ro.r("as.numeric(black_df$x1)")],
            "x2": [float(v) for v in ro.r("as.numeric(black_df$x2)")],
            "y": [float(v) for v in ro.r("as.numeric(black_df$y)")],
        },
        "grid": {
            "x1": [float(v) for v in ro.r("as.numeric(grid_df$x1)")],
            "x2": [float(v) for v in ro.r("as.numeric(grid_df$x2)")],
            "prediction": [float(v) for v in ro.r("as.numeric(grid_pred)")],
        },
    }


@register("bodyfat_gamboost")
def _bodyfat_gamboost() -> dict:
    """Additive bodyfat model, matching `gamboost.md`."""
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    book_utils.r_load_library("mboost")
    bodyfat = pl.read_csv(book_utils.data_dir() / "bodyfat.csv")
    book_utils.r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
    ro.r("bodyfat <- bodyfat_py")
    ro.r(
        """
        gam1_r <- gamboost(
          DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a),
          data = bodyfat
        )
        """
    )
    fitted = book_utils.r_numeric("as.numeric(predict(gam1_r, type = 'link'))").tolist()

    partial_curves: dict[str, dict] = {}
    for index, feature in enumerate(["hipcirc", "kneebreadth", "anthro3a"], start=1):
        lo = float(bodyfat[feature].min())
        hi = float(bodyfat[feature].max())
        grid = np.linspace(lo, hi, 120)
        base_df = bodyfat.to_pandas().iloc[[0] * grid.size].copy()
        base_df[feature] = grid
        book_utils.r_assign_dataframe("overlay_df", base_df)
        curve = book_utils.r_numeric(
            f"as.numeric(predict(gam1_r, newdata = overlay_df, which = {index}))"
        ).tolist()
        partial_curves[feature] = {"x": grid.tolist(), "effect": curve}

    return {"gam1_fitted": fitted, "partial_curves": partial_curves}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", help="only refresh the named target (repeatable)")
    parser.add_argument("--list", action="store_true", help="list all known targets and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(_REGISTRY):
            print(name)
        return 0

    targets = list(_REGISTRY.items())
    if args.only:
        unknown = sorted(set(args.only) - set(_REGISTRY))
        if unknown:
            parser.error(f"unknown target(s): {', '.join(unknown)}")
        targets = [(name, fn) for name, fn in targets if name in set(args.only)]

    cache_dir = book_utils.r_cache_dir()
    print(f"Writing cached R outputs to {cache_dir}")
    for name, fn in targets:
        print(f"  - {name} ... ", end="", flush=True)
        payload = fn()
        path = book_utils.save_cached_r_json(name, payload)
        print(f"wrote {path.relative_to(book_utils.project_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
