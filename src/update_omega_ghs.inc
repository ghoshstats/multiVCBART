#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <algorithm>  // std::max
#include <cmath>      // std::pow
#include <string>

#include "update_omega_ghs.h"

// -----------------------------
// RNG helpers
// -----------------------------
static inline double rgamma_rate(double shape, double rate) {
  return R::rgamma(shape, 1.0 / rate);
}
static inline double rinvgamma1(double shape, double rate) {
  return 1.0 / rgamma_rate(shape, rate);
}

// -----------------------------
// Cholesky with jitter
// -----------------------------
static bool chol_spd_arma(const arma::mat& A_in,
                          arma::mat& R,
                          double jitter,
                          int max_tries) {
  arma::mat A = 0.5 * (A_in + A_in.t());
  double base = arma::mean(A.diag());
  if (!arma::is_finite(base) || base <= 0.0) base = 1.0;

  for (int k = 0; k <= max_tries; ++k) {
    double eps = jitter * std::pow(10.0, k) * base;
    arma::mat Aj = A;
    Aj.diag() += eps;
    if (arma::chol(R, Aj)) return true;
  }
  return false;
}

static inline arma::vec chol_solve_upper(const arma::mat& R, const arma::vec& b) {
  arma::vec y = arma::solve(arma::trimatl(R.t()), b);
  arma::vec x = arma::solve(arma::trimatu(R), y);
  return x;
}

static inline arma::vec chol_norm_sample(const arma::mat& R, const arma::vec& mu) {
  arma::vec z(mu.n_elem);
  for (arma::uword i = 0; i < z.n_elem; ++i) z[i] = R::rnorm(0.0, 1.0);
  arma::vec eps = arma::solve(arma::trimatu(R), z);
  return mu + eps;
}

static inline arma::mat chol_inv_from_upper(const arma::mat& R) {
  arma::mat I = arma::eye(R.n_rows, R.n_cols);
  arma::mat Rin = arma::solve(arma::trimatu(R), I);
  return Rin * Rin.t();
}

// -----------------------------
// Gaussian prior Omega sweep
// -----------------------------
// [[Rcpp::export]]
Rcpp::List omega_gaussian_sweep_cpp(arma::mat Omega,
                                   const arma::mat& S,
                                   int n,
                                   double v0,
                                   double a0,
                                   double b0,
                                   double jitter,
                                   int max_tries) {
  Rcpp::RNGScope scope;
  int q = (int)Omega.n_cols;

  for (int p = 0; p < q; ++p) {
    arma::uvec idx(q - 1);
    for (int k = 0, t = 0; k < q; ++k) if (k != p) idx[t++] = (arma::uword)k;

    arma::mat Omega11 = Omega.submat(idx, idx);

    arma::mat R11;
    if (!chol_spd_arma(Omega11, R11, jitter, max_tries))
      Rcpp::stop("omega_gaussian_sweep_cpp: chol(Omega11) failed");

    arma::mat Sigma11 = chol_inv_from_upper(R11);

    arma::vec s12(q - 1);
    for (int t = 0; t < q - 1; ++t) s12[t] = S((int)idx[t], p);

    double spp = S(p, p);

    arma::mat A = spp * Sigma11;
    A.diag() += (1.0 / v0);

    arma::mat RA;
    if (!chol_spd_arma(A, RA, jitter, max_tries))
      Rcpp::stop("omega_gaussian_sweep_cpp: chol(A) failed");

    arma::vec mu   = -chol_solve_upper(RA, s12);
    arma::vec beta = chol_norm_sample(RA, mu);

    double gamma = rgamma_rate(n / 2.0 + a0, (spp + 2.0 * b0) / 2.0);

    for (int t = 0; t < q - 1; ++t) {
      int k = (int)idx[t];
      Omega(k, p) = beta[t];
      Omega(p, k) = beta[t];
    }

    double quad = arma::as_scalar(beta.t() * Sigma11 * beta);
    Omega(p, p) = gamma + quad;
  }

  Omega = 0.5 * (Omega + Omega.t());
  return Rcpp::List::create(Rcpp::_["Omega"] = Omega);
}

// -----------------------------
//  Graphical horseshoe sweep
// -----------------------------
// [[Rcpp::export]]
Rcpp::List ghs_one_sweep_stable_cpp(arma::mat Omega,
                                   const arma::mat& S,
                                   int n,
                                   arma::mat lambda2,
                                   arma::mat nu,
                                   double tau2,
                                   double xi_tau,
                                   double b_tau, // Added for sparsity control
                                   double a0,
                                   double b0,
                                   double lam2_min,
                                   double tau2_min,
                                   double jitter,
                                   int max_tries) {
  Rcpp::RNGScope scope;
  int q = (int)Omega.n_cols;

  tau2 = std::max(tau2, tau2_min);

  // ---- column updates ----
  for (int p = 0; p < q; ++p) {
    arma::uvec idx(q - 1);
    for (int k = 0, t = 0; k < q; ++k) if (k != p) idx[t++] = (arma::uword)k;

    arma::mat Omega11 = Omega.submat(idx, idx);

    arma::mat R11;
    if (!chol_spd_arma(Omega11, R11, jitter, max_tries))
      Rcpp::stop("ghs_one_sweep_stable_cpp: chol(Omega11) failed");

    arma::mat Sigma11 = chol_inv_from_upper(R11);

    arma::vec s12(q - 1), lam2v(q - 1);
    for (int t = 0; t < q - 1; ++t) {
      int k = (int)idx[t];
      s12[t]  = S(k, p);
      lam2v[t] = std::max(lambda2(k, p), lam2_min);
    }

    double spp = S(p, p);
    arma::vec dvec = 1.0 / (tau2 * lam2v);

    arma::mat A = spp * Sigma11;
    A.diag() += dvec;

    arma::mat RA;
    if (!chol_spd_arma(A, RA, jitter, max_tries))
      Rcpp::stop("ghs_one_sweep_stable_cpp: chol(A) failed");

    arma::vec mu   = -chol_solve_upper(RA, s12);
    arma::vec beta = chol_norm_sample(RA, mu);

    double gamma = rgamma_rate(n / 2.0 + a0, (spp + 2.0 * b0) / 2.0);

    for (int t = 0; t < q - 1; ++t) {
      int k = (int)idx[t];
      Omega(k, p) = beta[t];
      Omega(p, k) = beta[t];
    }

    double quad = arma::as_scalar(beta.t() * Sigma11 * beta);
    Omega(p, p) = gamma + quad;
  }

  // ---- local HS updates ----
  for (int k = 0; k < q - 1; ++k) {
    for (int kp = k + 1; kp < q; ++kp) {
      double w = Omega(k, kp);

      double rate_lam = (w * w) / (2.0 * tau2) + 1.0 / nu(k, kp);
      double lam2 = rinvgamma1(1.0, rate_lam);
      lam2 = std::max(lam2, lam2_min);
      lambda2(k, kp) = lambda2(kp, k) = lam2;

      double rate_nu = 1.0 + 1.0 / lam2;
      double nu_kkp = rinvgamma1(1.0, rate_nu); // Mathematically exact shape = 1.0
      nu(k, kp) = nu(kp, k) = nu_kkp;
    }
  }

  // ---- global tau update (Makalic & Schmidt) ----
  double w2_over_lam = 0.0;
  for (int k = 0; k < q - 1; ++k) {
    for (int kp = k + 1; kp < q; ++kp) {
      double lam2 = std::max(lambda2(k, kp), lam2_min);
      double w = Omega(k, kp);
      w2_over_lam += (w * w) / lam2;
    }
  }

  double m = (double)q * (q - 1) / 2.0;
  tau2 = rinvgamma1(0.5 + m / 2.0, 1.0 / xi_tau + 0.5 * w2_over_lam);
  tau2 = std::max(tau2, tau2_min);

  // Manual Sparsity Knob: Uses b_tau (from R's omega_b_tau)
  xi_tau = rinvgamma1(1.0, b_tau + 1.0 / tau2);

  Omega = 0.5 * (Omega + Omega.t());

  return Rcpp::List::create(
    Rcpp::_["Omega"]   = Omega,
    Rcpp::_["lambda2"] = lambda2,
    Rcpp::_["nu"]      = nu,
    Rcpp::_["tau2"]    = tau2,
    Rcpp::_["xi_tau"]  = xi_tau
  );
}

// -----------------------------
// Wrapper: auto / fallback
// -----------------------------
// [[Rcpp::export]]
Rcpp::List update_Omega_cpp(const arma::mat& R,
                            Rcpp::List state,
                            std::string method,
                            int q_small,
                            double v0_gauss,
                            double a0, double b0,
                            double a_tau, double b_tau,
                            double lam2_min,
                            double tau2_min,
                            double jitter,
                            int max_tries) {
  Rcpp::RNGScope scope;

  int n = (int)R.n_rows;
  int q = (int)R.n_cols;

  arma::mat S = R.t() * R;

  if (!state.containsElementNamed("Omega")) {
    state["Omega"] = arma::eye(q, q);
  }
  arma::mat Omega = Rcpp::as<arma::mat>(state["Omega"]);

  if (method == "auto") {
    method = (q <= q_small) ? "gaussian" : "stable_hs";
  }

  if (method == "gaussian") {
    Rcpp::List out = omega_gaussian_sweep_cpp(Omega, S, n, v0_gauss, a0, b0, jitter, max_tries);
    state["Omega"] = out["Omega"];
    return state;
  }

  if (!state.containsElementNamed("lambda2")) {
    arma::mat lam = arma::ones(q, q);
    lam.diag().ones();
    state["lambda2"] = lam;
  }
  if (!state.containsElementNamed("nu")) {
    arma::mat nuM = arma::ones(q, q);
    nuM.diag().ones();
    state["nu"] = nuM;
  }
  if (!state.containsElementNamed("tau2")) {
    state["tau2"] = 1.0;
  }
  if (!state.containsElementNamed("xi_tau")) {
    state["xi_tau"] = 1.0;
  }

  arma::mat lambda2 = Rcpp::as<arma::mat>(state["lambda2"]);
  arma::mat nu      = Rcpp::as<arma::mat>(state["nu"]);
  double tau2       = Rcpp::as<double>(state["tau2"]);
  double xi_tau     = Rcpp::as<double>(state["xi_tau"]);

  try {
    // b_tau correctly passed into the sweep
    Rcpp::List out = ghs_one_sweep_stable_cpp(Omega, S, n, lambda2, nu, tau2, xi_tau, b_tau,
                                             a0, b0, lam2_min, tau2_min, jitter, max_tries);
    state["Omega"]   = out["Omega"];
    state["lambda2"] = out["lambda2"];
    state["nu"]      = out["nu"];
    state["tau2"]    = out["tau2"];
    state["xi_tau"]  = out["xi_tau"];
    return state;
  } catch (...) {
    Rcpp::warning("stable_hs Omega update failed; falling back to Gaussian Omega sweep this iteration.");
    Rcpp::List out2 = omega_gaussian_sweep_cpp(Omega, S, n, v0_gauss, a0, b0, jitter, max_tries);
    state["Omega"] = out2["Omega"];
    return state;
  }
}
