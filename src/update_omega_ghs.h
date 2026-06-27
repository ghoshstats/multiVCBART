#ifndef UPDATE_OMEGA_GHS_H
#define UPDATE_OMEGA_GHS_H

#include <RcppArmadillo.h>
#include <string>

Rcpp::List omega_gaussian_sweep_cpp(arma::mat Omega,
                                    const arma::mat& S,
                                    int n,
                                    double v0 = 1e6,
                                    double a0 = 1.0,
                                    double b0 = 1.0,
                                    double jitter = 1e-10,
                                    int max_tries = 10);

// Passed b_tau down into this function to act as the sparsity override
Rcpp::List ghs_one_sweep_stable_cpp(arma::mat Omega,
                                    const arma::mat& S,
                                    int n,
                                    arma::mat lambda2,
                                    arma::mat nu,
                                    double tau2,
                                    double xi_tau,
                                    double b_tau,
                                    double a0 = 1.0,
                                    double b0 = 1.0,
                                    double lam2_min = 1e-12,
                                    double tau2_min = 1e-12,
                                    double jitter = 1e-10,
                                    int max_tries = 10);

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
                            int max_tries);

#endif
