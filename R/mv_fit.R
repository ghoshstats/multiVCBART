#' Fit a Multivariate BART Model
#'
#' Multivariate flexBART with Omega-coupled residuals and group-HS leaf shrinkage.
#'
#' @param Y_train n x q matrix of training responses.
#' @param cov_ensm p x R integer matrix mapping covariates to ensembles.
#' @param Z_train n x R matrix (modifiers / ensemble weights).
#' @param X_cont_train n x p_cont numeric matrix of continuous predictors (or NULL).
#' @param X_cat_train n x p_cat integer matrix of categorical predictors (or NULL).
#' @param Y_test Optional n_test x q matrix of test responses.
#' @param Z_test Optional n_test x R matrix of test modifiers.
#' @param X_cont_test Optional n_test x p_cont numeric matrix of test continuous predictors.
#' @param X_cat_test Optional n_test x p_cat integer matrix of test categorical predictors.
#' @param cutpoints_list List of available cutpoints for continuous variables.
#' @param cat_levels_list List of available levels for categorical variables.
#' @param edge_mat_list List of edge matrices for network-structured categorical variables.
#' @param nest_list List defining nesting structure between variables.
#' @param graph_cut_type Integer specifying the graph partitioning algorithm.
#' @param sparse Boolean indicating if variable selection should be performed.
#' @param a_u Dirichlet concentration parameter for sparse variable selection (e.g., 0.5).
#' @param nest_v Boolean indicating if nesting influences variable selection.
#' @param nest_v_option Integer specifying how nesting influences variable selection.
#' @param nest_c Boolean indicating if nesting influences cutset selection.
#' @param M_vec Integer vector of length R specifying number of trees per ensemble.
#' @param alpha_vec Numeric vector of length R for tree prior depth penalty.
#' @param beta_vec Numeric vector of length R for tree prior depth penalty.
#' @param mu0_vec Numeric vector of prior means for leaf nodes.
#' @param sigma_B_n Hyperparameter for global variance scale.
#' @param omega_method String specifying Omega update method ("auto", "gaussian", or "stable_hs").
#' @param q_small Integer threshold for choosing Omega update method.
#' @param v0_gauss Hyperparameter for Gaussian Omega prior.
#' @param omega_a0 Hyperparameter for Omega diagonal prior.
#' @param omega_b0 Hyperparameter for Omega diagonal prior.
#' @param omega_b_tau Penalty term (1/tau_0^2) for Graphical Horseshoe sparsity. Larger values force a sparser network.
#' @param lam2_min Minimum allowed value for local shrinkage scales.
#' @param tau2_min Minimum allowed value for global shrinkage scale.
#' @param jitter Small value added to diagonal for Cholesky stability.
#' @param max_tries Maximum attempts for Cholesky decomposition.
#' @param nd Number of posterior draws to save.
#' @param burn Number of burn-in iterations.
#' @param thin Thinning parameter.
#' @param save_samples Boolean to save full posterior samples.
#' @param save_trees Boolean to save tree structures.
#' @param verbose Boolean to print MCMC progress.
#' @param print_every Integer specifying frequency of progress updates.
#' 
#' @useDynLib mvBART, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @export
mv_flexbart <- function(
    Y_train,
    cov_ensm,
    Z_train,
    X_cont_train = NULL,
    X_cat_train = NULL,
    Y_test = NULL,
    Z_test = NULL,
    X_cont_test = NULL,
    X_cat_test = NULL,
    cutpoints_list = NULL,
    cat_levels_list = NULL,
    edge_mat_list = NULL,
    nest_list = NULL,
    graph_cut_type = 0,
    sparse = FALSE, a_u = 0.5,
    nest_v = FALSE, nest_v_option = 0, nest_c = FALSE,
    M_vec,
    alpha_vec,
    beta_vec,
    mu0_vec = rep(0, length(M_vec)),
    sigma_B_n = 1.0,
    omega_method = "auto",
    q_small = 3,
    v0_gauss = 1e6,
    omega_a0 = 1, omega_b0 = 1,
    omega_b_tau = 1.0,
    lam2_min = 1e-12,
    tau2_min = 1e-12,
    jitter = 1e-10,
    max_tries = 10,
    nd = 1000, burn = 500, thin = 1,
    save_samples = TRUE,
    save_trees = FALSE,
    verbose = TRUE,
    print_every = 50
) {
  Y_train <- as.matrix(Y_train)
  Z_train <- as.matrix(Z_train)
  n <- nrow(Y_train); q <- ncol(Y_train)
  stopifnot(nrow(Z_train) == n)
  
  tY_train <- t(Y_train)    # q x n
  tZ_train <- t(Z_train)    # R x n
  
  if (is.null(X_cont_train)) tXc_train <- matrix(numeric(0), 0, n) else tXc_train <- t(as.matrix(X_cont_train))
  if (is.null(X_cat_train)) tXa_train <- matrix(integer(0), 0, n) else tXa_train <- t(as.matrix(X_cat_train))
  
  if (is.null(Y_test) || is.null(Z_test)) {
    tY_test <- matrix(numeric(0), 0, 0)
    tZ_test <- matrix(numeric(0), 0, 0)
    tXc_test <- matrix(numeric(0), 0, 0)
    tXa_test <- matrix(integer(0), 0, 0)
  } else {
    Y_test <- as.matrix(Y_test)
    Z_test <- as.matrix(Z_test)
    stopifnot(ncol(Y_test) == q)
    stopifnot(nrow(Z_test) == nrow(Y_test))
    tY_test <- t(Y_test)
    tZ_test <- t(Z_test)
    if (is.null(X_cont_test)) tXc_test <- matrix(numeric(0), 0, nrow(Y_test)) else tXc_test <- t(as.matrix(X_cont_test))
    if (is.null(X_cat_test)) tXa_test <- matrix(integer(0), 0, nrow(Y_test)) else tXa_test <- t(as.matrix(X_cat_test))
  }
  
  ._mv_fit(
    tY_train, cov_ensm, tZ_train, tXc_train, tXa_train,
    tY_test, tZ_test, tXc_test, tXa_test,
    cutpoints_list, cat_levels_list, edge_mat_list, nest_list,
    graph_cut_type,
    sparse, a_u,
    nest_v, nest_v_option, nest_c,
    M_vec, alpha_vec, beta_vec, mu0_vec,
    sigma_B_n,
    omega_method, q_small, v0_gauss,
    omega_a0, omega_b0, omega_b_tau,
    lam2_min, tau2_min, jitter, max_tries,
    nd, burn, thin,
    save_samples, save_trees, verbose, print_every
  )
}
