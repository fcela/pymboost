args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

case_name <- parse_arg("--case")
n <- as.integer(parse_arg("--n", "2000"))

library(mboost)

x_linear <- seq(-1, 1, length.out = n)
z_linear <- cos(3 * x_linear)
y_linear <- 1.5 * x_linear - 0.25 * z_linear
linear <- data.frame(x = x_linear, z = z_linear, y = y_linear)

x_spline <- seq(0, 1, length.out = n)
z_spline <- seq(-1, 1, length.out = n)
y_spline <- sin(2 * pi * x_spline) + 0.5 * z_spline
spline <- data.frame(x = x_spline, z = z_spline, y = y_spline)

x_mono <- seq(0, 1, length.out = n)
y_mono <- log1p(5 * x_mono) + 0.1 * x_mono
mono <- data.frame(x = x_mono, y = y_mono)

x_tree <- seq(-1, 1, length.out = n)
z_tree <- sin(4 * pi * x_tree)
y_tree <- ifelse(x_tree < -0.2, -1.0, ifelse(x_tree < 0.4, 0.5, 1.25)) + 0.25 * as.numeric(z_tree > 0)
tree <- data.frame(x = x_tree, z = z_tree, y = y_tree)

x_bin <- seq(-1, 1, length.out = n)
z_bin <- cos(2.5 * x_bin)
score_bin <- 1.4 * x_bin - 0.8 * z_bin + 0.15 * sin(5 * x_bin)
y_bin <- as.numeric(score_bin > 0.1)
binomial_df <- data.frame(x = x_bin, z = z_bin, y = y_bin)

x_pois <- seq(-1, 1, length.out = n)
z_pois <- sin(2 * x_pois)
mu_pois <- exp(0.25 + 0.45 * x_pois - 0.25 * z_pois)
y_pois <- round(mu_pois)
poisson_df <- data.frame(x = x_pois, z = z_pois, y = y_pois)

fold_ids <- rep(0:4, length.out = nrow(linear))
fold_matrix <- matrix(1, nrow = nrow(linear), ncol = length(unique(fold_ids)))
for (i in seq_len(ncol(fold_matrix))) {
  fold_matrix[fold_ids == (i - 1), i] <- 0
}

if (case_name == "glmboost_gaussian_bols") {
  glmboost(
    y ~ x + z,
    data = linear,
    family = Gaussian(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "gamboost_gaussian_bbs_bols") {
  gamboost(
    y ~ bbs(x, knots = 20, df = 4, degree = 3, differences = 2) + bols(z),
    data = spline,
    family = Gaussian(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "cvrisk_gaussian_bols") {
  cvrisk(
    gamboost(
      y ~ x + z,
      data = linear,
      family = Gaussian(),
      control = boost_control(mstop = 50, nu = 0.1)
    ),
    folds = fold_matrix
  )
} else if (case_name == "cvrisk_gaussian_btree") {
  cvrisk(
    gamboost(
      y ~ btree(x),
      data = linear[, c("x", "y")],
      family = Gaussian(),
      control = boost_control(mstop = 50, nu = 0.1)
    ),
    folds = fold_matrix
  )
} else if (case_name == "gamboost_gaussian_bmono") {
  gamboost(
    y ~ bmono(x, constraint = "increasing", knots = 20, df = 4, degree = 3, differences = 2),
    data = mono,
    family = Gaussian(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "gamboost_gaussian_btree") {
  gamboost(
    y ~ btree(x, z),
    data = tree,
    family = Gaussian(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "glmboost_binomial_bols") {
  glmboost(
    factor(y) ~ x + z,
    data = binomial_df,
    family = Binomial(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "glmboost_poisson_bols") {
  glmboost(
    y ~ x + z,
    data = poisson_df,
    family = Poisson(),
    control = boost_control(mstop = 100, nu = 0.1)
  )
} else if (case_name == "cvrisk_gaussian_bmono") {
  mono_fold_ids <- rep(0:4, length.out = nrow(mono))
  mono_fold_matrix <- matrix(1, nrow = nrow(mono), ncol = length(unique(mono_fold_ids)))
  for (i in seq_len(ncol(mono_fold_matrix))) {
    mono_fold_matrix[mono_fold_ids == (i - 1), i] <- 0
  }
  cvrisk(
    gamboost(
      y ~ bmono(x, constraint = "increasing", knots = 20, df = 4, degree = 3, differences = 2),
      data = mono,
      family = Gaussian(),
      control = boost_control(mstop = 50, nu = 0.1)
    ),
    folds = mono_fold_matrix
  )
} else {
  stop(paste("unknown case:", case_name))
}
