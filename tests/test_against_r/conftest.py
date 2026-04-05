from __future__ import annotations

import numpy as np
import pytest
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import get_conversion, localconverter
from rpy2.robjects.packages import importr


@pytest.fixture(scope="session")
def r_mboost():
    return importr("mboost")


def _as_r_vector(values: np.ndarray):
    arr = np.asarray(values)
    with localconverter(default_converter + numpy2ri.converter):
        return get_conversion().py2rpy(arr)


def _as_r_matrix(values: np.ndarray):
    arr = np.asarray(values)
    with localconverter(default_converter + numpy2ri.converter):
        return get_conversion().py2rpy(arr)


def _as_r_array(values: np.ndarray):
    arr = np.asarray(values)
    with localconverter(default_converter + numpy2ri.converter):
        return get_conversion().py2rpy(arr)


@pytest.fixture(scope="session")
def r_reference_runner(r_mboost):
    r_runner = robjects.r(
        """
        function(x, y, family_name, mstop, nu, tau) {
          family <- switch(
            family_name,
            gaussian = mboost::Gaussian(),
            binomial = mboost::Binomial(),
            expectile = mboost::ExpectReg(tau = tau),
            gamma = mboost::GammaReg(),
            huber = mboost::Huber(),
            laplace = mboost::Laplace(),
            poisson = mboost::Poisson(),
            quantile = mboost::QuantReg(tau = tau),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") {
            factor(y, levels = c(0, 1))
          } else {
            y
          }
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = x, y = response),
            family = family,
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        family: str,
        mstop: int = 5,
        nu: float = 0.1,
        tau: float = 0.5,
    ):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be one-dimensional arrays of the same length")

        fit = r_runner(
            _as_r_vector(x),
            _as_r_vector(y),
            family,
            mstop,
            nu,
            tau,
        )

        fitted = np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1)
        risk = np.asarray(fit.rx2("risk"), dtype=np.float64)
        return {
            "fitted": fitted,
            "risk": risk,
        }

    return run_reference


@pytest.fixture(scope="session")
def r_reference_predict_runner(r_mboost):
    r_runner = robjects.r(
        """
        function(x, y, x_new, family_name, mstop, nu, tau, pred_type) {
          family <- switch(
            family_name,
            gaussian = mboost::Gaussian(),
            binomial = mboost::Binomial(),
            expectile = mboost::ExpectReg(tau = tau),
            gamma = mboost::GammaReg(),
            huber = mboost::Huber(),
            laplace = mboost::Laplace(),
            poisson = mboost::Poisson(),
            quantile = mboost::QuantReg(tau = tau),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") {
            factor(y, levels = c(0, 1))
          } else {
            y
          }
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = x, y = response),
            family = family,
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = pred_type)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        family: str,
        mstop: int = 5,
        nu: float = 0.1,
        tau: float = 0.5,
        pred_type: str = "link",
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            family,
            mstop,
            nu,
            tau,
            pred_type,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_interaction_reference_runner():
    r_runner = robjects.r(
        """
        function(x1, x2, y, x1_new, x2_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x1:x2,
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x1 = x1_new, x2 = x2_new), type = "link"),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray,
        *,
        x1_new: np.ndarray,
        x2_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x1, dtype=np.float64)),
            _as_r_vector(np.asarray(x2, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x1_new, dtype=np.float64)),
            _as_r_vector(np.asarray(x2_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_transform_reference_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ log(x),
            data = data.frame(x = x, y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = "link"),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_factor_interaction_reference_runner():
    r_runner = robjects.r(
        """
        function(x, g, y, x_new, g_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x:g,
            data = data.frame(x = x, g = factor(g), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new, g = factor(g_new, levels = levels(factor(g)))),
              type = "link"
            ),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        g: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        g_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(g)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            _as_r_vector(np.asarray(g_new)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_star_interaction_reference_runner():
    r_runner = robjects.r(
        """
        function(x1, x2, y, x1_new, x2_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x1 * x2,
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x1 = x1_new, x2 = x2_new), type = "link"),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray,
        *,
        x1_new: np.ndarray,
        x2_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x1, dtype=np.float64)),
            _as_r_vector(np.asarray(x2, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x1_new, dtype=np.float64)),
            _as_r_vector(np.asarray(x2_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_factor_star_reference_runner():
    r_runner = robjects.r(
        """
        function(x, g, y, x_new, g_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x * g,
            data = data.frame(x = x, g = factor(g), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new, g = factor(g_new, levels = levels(factor(g)))),
              type = "link"
            ),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        g: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        g_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(g, dtype=object)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            _as_r_vector(np.asarray(g_new, dtype=object)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_transform_factor_star_reference_runner():
    r_runner = robjects.r(
        """
        function(x, g, y, x_new, g_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ log(x) * g,
            data = data.frame(x = x, g = factor(g), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new, g = factor(g_new, levels = levels(factor(g)))),
              type = "link"
            ),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        g: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        g_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(g, dtype=object)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            _as_r_vector(np.asarray(g_new, dtype=object)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_dot_mixed_reference_runner():
    r_runner = robjects.r(
        """
        function(x, z, g, y, x_new, z_new, g_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ .,
            data = data.frame(x = x, z = z, g = factor(g), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new, z = z_new, g = factor(g_new, levels = levels(factor(g)))),
              type = "link"
            ),
            risk = mboost::risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        z: np.ndarray,
        g: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        z_new: np.ndarray,
        g_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(g, dtype=object)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            _as_r_vector(np.asarray(z_new, dtype=np.float64)),
            _as_r_vector(np.asarray(g_new, dtype=object)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_three_bbs_partial_runner():
    r_runner = robjects.r(
        """
        function(x1, x2, x3, y, grid, which_name, mstop, nu) {
          library(mboost)
          fit <- mboost::gamboost(
            y ~ bbs(x1) + bbs(x2) + bbs(x3),
            data = data.frame(x1 = x1, x2 = x2, x3 = x3, y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          overlay_df <- data.frame(
            x1 = rep(x1[1], length(grid)),
            x2 = rep(x2[1], length(grid)),
            x3 = rep(x3[1], length(grid)),
            y = rep(y[1], length(grid))
          )
          which_idx <- switch(
            which_name,
            x1 = 1L,
            x2 = 2L,
            x3 = 3L,
            stop("unsupported term")
          )
          if (which_name == "x1") overlay_df$x1 <- grid
          if (which_name == "x2") overlay_df$x2 <- grid
          if (which_name == "x3") overlay_df$x3 <- grid
          list(
            pred = as.numeric(stats::predict(fit, newdata = overlay_df, which = which_idx))
          )
        }
        """
    )

    def run_reference(
        x1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        y: np.ndarray,
        *,
        grid: np.ndarray,
        which_name: str,
        mstop: int = 100,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x1, dtype=np.float64)),
            _as_r_vector(np.asarray(x2, dtype=np.float64)),
            _as_r_vector(np.asarray(x3, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(grid, dtype=np.float64)),
            which_name,
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, family_name, folds, mstop, nu, tau) {
          family <- switch(
            family_name,
            gaussian = mboost::Gaussian(),
            binomial = mboost::Binomial(),
            expectile = mboost::ExpectReg(tau = tau),
            huber = mboost::Huber(),
            laplace = mboost::Laplace(),
            poisson = mboost::Poisson(),
            quantile = mboost::QuantReg(tau = tau),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") {
            factor(y, levels = c(0, 1))
          } else {
            y
          }
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = x, y = response),
            family = family,
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- mboost::cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_cvrisk(
        x: np.ndarray,
        y: np.ndarray,
        *,
        family: str,
        folds: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
        tau: float = 0.5,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            family,
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            mstop,
            nu,
            tau,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_cvrisk


@pytest.fixture(scope="session")
def r_factor_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, folds, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = factor(x), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- mboost::cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        folds: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_factor_reference_runner():
    r_runner = robjects.r(
        """
        function(x, y, mstop, nu) {
          library(mboost)
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = factor(x), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = mboost::risk(fit),
            coef = cf[[1]],
            selected = selected(fit),
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "selected": np.asarray(fit.rx2("selected"), dtype=np.int64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_factor_predict_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, mstop, nu) {
          fit <- mboost::glmboost(
            y ~ x,
            data = data.frame(x = factor(x), y = y),
            family = mboost::Gaussian(),
            control = mboost::boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(
              fit,
              newdata = data.frame(x = factor(x_new, levels = levels(factor(x)))),
              type = "link"
            )
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        mstop: int = 5,
        nu: float = 0.1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new)),
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_reference




@pytest.fixture(scope="session")
def r_gamboost_bbs_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, knots, lambda_value, df_value, degree, differences, center_value, mstop, nu) {
          library(mboost)
          bl <- if (is.na(knots)) {
            if (is.na(lambda_value)) {
              bbs(x, df = df_value,
                  degree = degree, differences = differences, center = center_value)
            } else {
              bbs(x, lambda = lambda_value,
                  degree = degree, differences = differences, center = center_value)
            }
          } else if (is.na(lambda_value)) {
            bbs(x, knots = knots, df = df_value,
                degree = degree, differences = differences, center = center_value)
          } else {
            bbs(x, knots = knots, lambda = lambda_value,
                degree = degree, differences = differences, center = center_value)
          }
          fit <- gamboost(
            y ~ bl,
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = "link"),
            risk = risk(fit),
            coef = cf[[1]]
          )
        }
        """
    )

    def run_bbs_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray | None = None,
        knots: int,
        lambda_value: float | None,
        df: int | None = None,
        degree: int,
        differences: int,
        center: bool = False,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x if x_new is None else x_new, dtype=np.float64)),
            np.nan if knots is None else knots,
            np.nan if lambda_value is None else lambda_value,
            -1 if df is None else df,
            degree,
            differences,
            center,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
        }

    return run_bbs_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, constraint, knots, lambda_value, degree, differences, boundary_constraints_value, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences,
              boundary.constraints = boundary_constraints_value
            ),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = "link"),
            risk = risk(fit),
            coef = cf[[1]]
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray | None = None,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        boundary_constraints: bool = False,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x if x_new is None else x_new, dtype=np.float64)),
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            boundary_constraints,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_family_runner():
    r_runner = robjects.r(
        """
        function(x, y, family_name, constraint, knots, lambda_value, degree, differences, mstop, nu) {
          library(mboost)
          family <- switch(
            family_name,
            binomial = Binomial(),
            poisson = Poisson(),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") factor(y, levels = c(0, 1)) else y
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences
            ),
            data = data.frame(x = x, y = response),
            family = family,
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]]
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        family: str,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            family,
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_family_predict_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, family_name, constraint, knots, lambda_value, degree, differences, mstop, nu) {
          library(mboost)
          family <- switch(
            family_name,
            binomial = Binomial(),
            poisson = Poisson(),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") factor(y, levels = c(0, 1)) else y
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences
            ),
            data = data.frame(x = x, y = response),
            family = family,
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = "link")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        family: str,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            family,
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_by_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, constraint, knots, lambda_value, degree, differences, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              by = z,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences
            ),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_factor_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, constraint, lambda_value, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(x, constraint = constraint, lambda = lambda_value),
            data = data.frame(x = ordered(x, levels = sort(unique(x))), y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          x_new <- ordered(x_new, levels = sort(unique(x)))
          list(
            fitted = stats::predict(fit, type = "link"),
            pred = stats::predict(fit, newdata = data.frame(x = x_new), type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray | None = None,
        constraint: str,
        lambda_value: float,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x if x_new is None else x_new)),
            constraint,
            float(lambda_value),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_factor_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, folds, constraint, lambda_value, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(x, constraint = constraint, lambda = lambda_value),
            data = data.frame(x = ordered(x, levels = sort(unique(x))), y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        folds: np.ndarray,
        constraint: str,
        lambda_value: float,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            constraint,
            float(lambda_value),
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bbs_by_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, knots, df_value, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x, knots = knots, df = df_value, degree = 3, differences = 2, by = z),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        knots: int,
        df_value: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            knots,
            df_value,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_gaussian_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x, knots = 5, df = 4, degree = 3, differences = 2) + bols(z),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            selected = selected(fit)
          )
        }
        """
    )

    def run_gamboost_gaussian(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "selected": np.asarray(fit.rx2("selected"), dtype=np.int64).reshape(-1),
        }

    return run_gamboost_gaussian


@pytest.fixture(scope="session")
def r_gamboost_gaussian_predict_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, x_new, z_new, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x, knots = 5, df = 4, degree = 3, differences = 2) + bols(z),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new, z = z_new),
              type = "link"
            )
          )
        }
        """
    )

    def run_predict(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        z_new: np.ndarray,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            _as_r_vector(np.asarray(z_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_predict


@pytest.fixture(scope="session")
def r_gamboost_bbs_multiterm_predict_runner():
    r_runner = robjects.r(
        """
        function(x1, x2, y, x1_new, x2_new, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x1) + bbs(x2),
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(
              fit,
              newdata = data.frame(x1 = x1_new, x2 = x2_new),
              type = "link"
            )
          )
        }
        """
    )

    def run_predict(
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray,
        *,
        x1_new: np.ndarray,
        x2_new: np.ndarray,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x1, dtype=np.float64)),
            _as_r_vector(np.asarray(x2, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x1_new, dtype=np.float64)),
            _as_r_vector(np.asarray(x2_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_predict


@pytest.fixture(scope="session")
def r_gamboost_dfbase_runner():
    r_runner = robjects.r(
        """
        function(x, y, dfbase, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ x,
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            dfbase = dfbase,
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        dfbase: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            dfbase,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, folds, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x, knots = 5, df = 4, degree = 3, differences = 2),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_gamboost_cvrisk(
        x: np.ndarray,
        y: np.ndarray,
        *,
        folds: np.ndarray,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_gamboost_cvrisk


@pytest.fixture(scope="session")
def r_cvrisk_bootstrap_runner():
    r_runner = robjects.r(
        """
        function(x, y, B, mstop, nu, seed) {
          library(mboost)
          set.seed(seed)
          folds <- cv(rep(1, length(x)), type = "bootstrap", B = B)
          fit <- glmboost(
            y ~ x,
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            folds = folds,
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        B: int,
        mstop: int,
        nu: float,
        seed: int = 1,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            B,
            mstop,
            nu,
            seed,
        )
        return {
            "folds": np.asarray(fit.rx2("folds"), dtype=np.float64),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, folds, constraint, knots, lambda_value, degree, differences, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences
            ),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        folds: np.ndarray,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bmono_family_cvrisk_runner():
    r_runner = robjects.r(
        """
        function(x, y, folds, family_name, constraint, knots, lambda_value, degree, differences, mstop, nu) {
          library(mboost)
          family <- switch(
            family_name,
            binomial = Binomial(),
            poisson = Poisson(),
            stop("unsupported family")
          )
          response <- if (family_name == "binomial") factor(y, levels = c(0, 1)) else y
          fit <- gamboost(
            y ~ bmono(
              x,
              constraint = constraint,
              knots = knots,
              lambda = lambda_value,
              degree = degree,
              differences = differences
            ),
            data = data.frame(x = x, y = response),
            family = family,
            control = boost_control(mstop = mstop, nu = nu)
          )
          cv_fit <- cvrisk(fit, folds = folds, grid = 0:mstop)
          list(
            risk = colMeans(cv_fit),
            fold_risk = cv_fit
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        folds: np.ndarray,
        family: str,
        constraint: str,
        knots: int,
        lambda_value: float,
        degree: int,
        differences: int,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(folds, dtype=np.float64)),
            family,
            constraint,
            knots,
            float(lambda_value),
            degree,
            differences,
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64).reshape(-1),
            "fold_risk": np.asarray(fit.rx2("fold_risk"), dtype=np.float64),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_varimp_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bmono(x, constraint = "increasing", knots = 5, lambda = 1, degree = 3, differences = 2) + bols(z),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          vi <- mboost:::varimp.mboost(fit)
          list(
            baselearner = unname(vi),
            baselearner_names = names(vi),
            variable = tapply(unname(vi), attr(vi, "variable_names"), sum),
            variable_names = names(tapply(unname(vi), attr(vi, "variable_names"), sum))
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "baselearner_names": [str(value) for value in fit.rx2("baselearner_names")],
            "baselearner": np.asarray(fit.rx2("baselearner"), dtype=np.float64).reshape(-1),
            "variable_names": [str(value) for value in fit.rx2("variable_names")],
            "variable": np.asarray(fit.rx2("variable"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bols_runner():
    r_runner = robjects.r(
        """
        function(z, y, mstop, nu, df_value, lambda_value) {
          library(mboost)
          learner <- if (!is.na(df_value)) {
            bols(z, df = df_value)
          } else if (!is.na(lambda_value)) {
            bols(z, lambda = lambda_value)
          } else {
            bols(z)
          }
          fit <- gamboost(
            y ~ learner,
            data = data.frame(z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_gamboost_bols(
        z: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
        df_value: float | None = None,
        lambda_value: float | None = None,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
            robjects.NA_Real if df_value is None else float(df_value),
            robjects.NA_Real if lambda_value is None else float(lambda_value),
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_gamboost_bols


@pytest.fixture(scope="session")
def r_gamboost_bols_factor_runner():
    r_runner = robjects.r(
        """
        function(x, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bols(x),
            data = data.frame(x = factor(x), y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_gamboost_bols_factor(
        x: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_gamboost_bols_factor


@pytest.fixture(scope="session")
def r_gamboost_brandom_runner():
    r_runner = robjects.r(
        """
        function(x, y, df_value, lambda_value, mstop, nu) {
          library(mboost)
          fit <- if (is.na(lambda_value)) {
            gamboost(
              y ~ brandom(x, df = df_value),
              data = data.frame(x = factor(x), y = y),
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
          } else {
            gamboost(
              y ~ brandom(x, lambda = lambda_value),
              data = data.frame(x = factor(x), y = y),
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
          }
          cf <- coef(fit)
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            coef = cf[[1]],
            offset = attr(cf, "offset")
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        df_value: float | None,
        lambda_value: float | None,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            -1 if df_value is None else df_value,
            np.nan if lambda_value is None else lambda_value,
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "coef": np.asarray(fit.rx2("coef"), dtype=np.float64).reshape(-1),
            "offset": float(fit.rx2("offset")[0]),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_btree_runner():
    r_runner = robjects.r(
        """
        function(x, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ btree(x),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            selected = selected(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "selected": np.asarray(fit.rx2("selected"), dtype=np.int64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_btree_by_runner():
    r_runner = robjects.r(
        """
        function(x, by, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ btree(x, by = by),
            data = data.frame(x = x, by = by, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            fitted = stats::predict(fit, type = "link"),
            risk = risk(fit),
            selected = selected(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        by: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(by, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "fitted": np.asarray(fit.rx2("fitted"), dtype=np.float64).reshape(-1),
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "selected": np.asarray(fit.rx2("selected"), dtype=np.int64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_btree_predict_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ btree(x),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(
              fit,
              newdata = data.frame(x = x_new),
              type = "link"
            )
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_btree_mixed_runner():
    r_runner = robjects.r(
        """
        function(x, z, y, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ btree(x) + bols(z),
            data = data.frame(x = x, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            risk = risk(fit),
            selected = selected(fit)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        *,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(z, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mstop,
            nu,
        )
        return {
            "risk": np.asarray(fit.rx2("risk"), dtype=np.float64),
            "selected": np.asarray(fit.rx2("selected"), dtype=np.int64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_bols_factor_predict_runner():
    r_runner = robjects.r(
        """
        function(x, y, x_new, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bols(x),
            data = data.frame(x = factor(x), y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          list(
            pred = stats::predict(
              fit,
              newdata = data.frame(x = factor(x_new, levels = levels(factor(x)))),
              type = "link"
            )
          )
        }
        """
    )

    def run_predict(
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_new: np.ndarray,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new)),
            mstop,
            nu,
        )
        return {
            "pred": np.asarray(fit.rx2("pred"), dtype=np.float64).reshape(-1),
        }

    return run_predict


@pytest.fixture(scope="session")
def r_confint_bootstrap_runner():
    r_runner = robjects.r(
        """
        function(x, y, boot_weights, x_new, level, mode, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bols(x),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          pred0 <- if (mode == "fitted") {
            as.numeric(predict(fit, type = "link"))
          } else {
            as.numeric(predict(fit, newdata = data.frame(x = x_new), which = 1))
          }
          preds <- sapply(seq_len(ncol(boot_weights)), function(i) {
            mod <- gamboost(
              y ~ bols(x),
              data = data.frame(x = x, y = y),
              weights = boot_weights[, i],
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
            if (mode == "fitted") {
              as.numeric(predict(mod, type = "link"))
            } else {
              as.numeric(predict(mod, newdata = data.frame(x = x_new), which = 1))
            }
          })
          alpha <- 1 - level
          list(
            estimate = pred0,
            lower = apply(preds, 1, stats::quantile, probs = alpha / 2),
            upper = apply(preds, 1, stats::quantile, probs = 1 - alpha / 2),
            std_error = apply(preds, 1, stats::sd)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        boot_weights: np.ndarray,
        x_new: np.ndarray | None = None,
        level: float,
        mode: str,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(boot_weights, dtype=np.float64)),
            _as_r_vector(np.asarray(x if x_new is None else x_new, dtype=np.float64)),
            float(level),
            mode,
            mstop,
            nu,
        )
        return {
            "estimate": np.asarray(fit.rx2("estimate"), dtype=np.float64).reshape(-1),
            "lower": np.asarray(fit.rx2("lower"), dtype=np.float64).reshape(-1),
            "upper": np.asarray(fit.rx2("upper"), dtype=np.float64).reshape(-1),
            "std_error": np.asarray(fit.rx2("std_error"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_confint_bootstrap_bbs_runner():
    r_runner = robjects.r(
        """
        function(x, y, boot_weights, x_new, level, mode, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bbs(x, knots = 5, lambda = 1.0, degree = 3, differences = 2),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          pred0 <- if (mode == "fitted") {
            as.numeric(predict(fit, type = "link"))
          } else {
            as.numeric(predict(fit, newdata = data.frame(x = x_new), which = 1))
          }
          preds <- sapply(seq_len(ncol(boot_weights)), function(i) {
            mod <- gamboost(
              y ~ bbs(x, knots = 5, lambda = 1.0, degree = 3, differences = 2),
              data = data.frame(x = x, y = y),
              weights = boot_weights[, i],
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
            if (mode == "fitted") {
              as.numeric(predict(mod, type = "link"))
            } else {
              as.numeric(predict(mod, newdata = data.frame(x = x_new), which = 1))
            }
          })
          alpha <- 1 - level
          list(
            estimate = pred0,
            lower = apply(preds, 1, stats::quantile, probs = alpha / 2),
            upper = apply(preds, 1, stats::quantile, probs = 1 - alpha / 2),
            std_error = apply(preds, 1, stats::sd)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        boot_weights: np.ndarray,
        x_new: np.ndarray | None = None,
        level: float,
        mode: str,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(boot_weights, dtype=np.float64)),
            _as_r_vector(np.asarray(x if x_new is None else x_new, dtype=np.float64)),
            float(level),
            mode,
            mstop,
            nu,
        )
        return {
            "estimate": np.asarray(fit.rx2("estimate"), dtype=np.float64).reshape(-1),
            "lower": np.asarray(fit.rx2("lower"), dtype=np.float64).reshape(-1),
            "upper": np.asarray(fit.rx2("upper"), dtype=np.float64).reshape(-1),
            "std_error": np.asarray(fit.rx2("std_error"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_confint_bootstrap_bmstop_runner():
    r_runner = robjects.r(
        """
        function(x, y, boot_weights, inner_boot, x_new, level, mstop, nu) {
          library(mboost)
          fit <- gamboost(
            y ~ bols(x),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          preds <- sapply(seq_len(ncol(boot_weights)), function(i) {
            mod <- gamboost(
              y ~ bols(x),
              data = data.frame(x = x, y = y),
              weights = boot_weights[, i],
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
            cv_fit <- cvrisk(mod, folds = inner_boot[, , i], grid = 0:mstop)
            mod <- mod[mstop(cv_fit)]
            as.numeric(predict(mod, newdata = data.frame(x = x_new), which = 1))
          })
          base_fit <- gamboost(
            y ~ bols(x),
            data = data.frame(x = x, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
          )
          alpha <- 1 - level
          list(
            lower = apply(preds, 1, stats::quantile, probs = alpha / 2),
            upper = apply(preds, 1, stats::quantile, probs = 1 - alpha / 2),
            std_error = apply(preds, 1, stats::sd)
          )
        }
        """
    )

    def run_reference(
        x: np.ndarray,
        y: np.ndarray,
        *,
        boot_weights: np.ndarray,
        inner_boot: np.ndarray,
        x_new: np.ndarray,
        level: float,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            _as_r_matrix(np.asarray(boot_weights, dtype=np.float64)),
            _as_r_array(np.asarray(inner_boot, dtype=np.float64)),
            _as_r_vector(np.asarray(x_new, dtype=np.float64)),
            float(level),
            mstop,
            nu,
        )
        return {
            "lower": np.asarray(fit.rx2("lower"), dtype=np.float64).reshape(-1),
            "upper": np.asarray(fit.rx2("upper"), dtype=np.float64).reshape(-1),
            "std_error": np.asarray(fit.rx2("std_error"), dtype=np.float64).reshape(-1),
        }

    return run_reference


@pytest.fixture(scope="session")
def r_gamboost_aic_runner():
    r_runner = robjects.r(
        """
        function(x, y, mode, mstop, nu) {
          library(mboost)
          fit <- if (mode == "bbs") {
            gamboost(
              y ~ bbs(x, knots = 5, df = 4, degree = 3, differences = 2),
              data = data.frame(x = x, y = y),
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
          } else if (mode == "bols") {
            gamboost(
              y ~ bols(x),
              data = data.frame(x = x, y = y),
              family = Gaussian(),
              control = boost_control(mstop = mstop, nu = nu)
            )
          } else {
            stop("unsupported mode")
          }
          a <- AIC(fit, method = "corrected")
          list(
            value = as.numeric(a),
            mstop = attr(a, "mstop"),
            df = attr(a, "df")[attr(a, "mstop")],
            aic_path = attr(a, "AIC")
          )
        }
        """
    )

    def run_gamboost_aic(
        x: np.ndarray,
        y: np.ndarray,
        *,
        mode: str,
        mstop: int,
        nu: float,
    ):
        fit = r_runner(
            _as_r_vector(np.asarray(x, dtype=np.float64)),
            _as_r_vector(np.asarray(y, dtype=np.float64)),
            mode,
            mstop,
            nu,
        )
        return {
            "value": float(fit.rx2("value")[0]),
            "mstop": int(fit.rx2("mstop")[0]),
            "df": float(fit.rx2("df")[0]),
            "aic_path": np.asarray(fit.rx2("aic_path"), dtype=np.float64).reshape(-1),
        }

    return run_gamboost_aic
