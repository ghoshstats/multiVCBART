#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <string>
#include <algorithm>
#include <cmath>

#include "update_tree.h"
#include "data_parsing_funs.h"
#include "funs.h"
#include "update_omega_ghs.h"

using namespace Rcpp;

static constexpr double EPS = 1e-16;

// -----------------------------
// HS helpers (Makalic–Schmidt)
// (namespaced to avoid collisions with update_omega_ghs.cpp)
// -----------------------------
namespace leaf_hs {
  static inline double rgamma_rate(double shape, double rate) {
    return R::rgamma(shape, 1.0 / rate);
  }
  static inline double rinvgamma_rate(double shape, double rate) {
    return 1.0 / rgamma_rate(shape, rate);
  }
}

// Sum of squares and count of leaf mus in a (j,r) group across all M_r trees
static inline void leaf_sumsq_count_one_group(
    std::vector<tree>& t_vec_flat,
    std::vector<suff_stat>& ss_train_vec_flat,
    int start_idx,
    int M,
    double& ssq,
    int& K)
{
  ssq = 0.0;
  K   = 0;

  for (int m = 0; m < M; ++m) {
    int idx = start_idx + m;
    for (suff_stat_it it = ss_train_vec_flat[idx].begin(); it != ss_train_vec_flat[idx].end(); ++it) {
      int nid = it->first;
      tree::tree_p np = t_vec_flat[idx].get_ptr(nid);
      if (!np) continue;
      double mu = np->get_mu();
      ssq += mu * mu;
      ++K;
    }
  }
}

// [[Rcpp::export(name="._mv_fit")]]
Rcpp::List mv_fit(Rcpp::NumericMatrix tY_train,
                  Rcpp::IntegerMatrix cov_ensm,
                  Rcpp::NumericMatrix tZ_train,
                  Rcpp::NumericMatrix tX_cont_train,
                  Rcpp::IntegerMatrix tX_cat_train,
                  Rcpp::NumericMatrix tY_test,
                  Rcpp::NumericMatrix tZ_test,
                  Rcpp::NumericMatrix tX_cont_test,
                  Rcpp::IntegerMatrix tX_cat_test,
                  Rcpp::Nullable<Rcpp::List> cutpoints_list,
                  Rcpp::Nullable<Rcpp::List> cat_levels_list,
                  Rcpp::Nullable<Rcpp::List> edge_mat_list,
                  Rcpp::Nullable<Rcpp::List> nest_list,
                  int graph_cut_type,
                  bool sparse,
                  double a_u, // Kept: Dirichlet concentration
                  bool nest_v,
                  int nest_v_option,
                  bool nest_c,
                  Rcpp::IntegerVector M_vec,
                  Rcpp::NumericVector alpha_vec,
                  Rcpp::NumericVector beta_vec,
                  Rcpp::NumericVector mu0_vec,
                  double sigma_B_n,
                  std::string omega_method,
                  int q_small,
                  double v0_gauss,
                  double omega_a0,
                  double omega_b0,
                  double omega_b_tau, // Kept: GHS Sparsity Knob (1 / tau_0^2)
                  double lam2_min,
                  double tau2_min,
                  double jitter,
                  int max_tries,
                  int nd,
                  int burn,
                  int thin,
                  bool save_samples,
                  bool save_trees,
                  bool verbose,
                  int print_every)
{
  RNGScope scope;
  RNG gen;

  // -----------------------------
  // Dimensions
  // -----------------------------
  int q = tY_train.nrow();
  int n_train = tY_train.ncol();

  int Rens = tZ_train.nrow();
  if (tZ_train.ncol() != n_train) stop("tZ_train must be R x n_train.");
  if ((int)M_vec.size() != Rens) stop("M_vec length must equal R.");
  if ((int)alpha_vec.size() != Rens || (int)beta_vec.size() != Rens || (int)mu0_vec.size() != Rens)
    stop("alpha_vec, beta_vec, mu0_vec must all be length R.");

  int p_cont = (tX_cont_train.size() > 0) ? tX_cont_train.nrow() : 0;
  int p_cat  = (tX_cat_train.size()  > 0) ? tX_cat_train.nrow()  : 0;
  int p      = p_cont + p_cat;

  int n_test = (tY_test.size() > 0) ? tY_test.ncol() : 0;
  if (n_test > 0) {
    if (tY_test.nrow() != q) stop("tY_test must be q x n_test.");
    if (tZ_test.nrow() != Rens || tZ_test.ncol() != n_test) stop("tZ_test must be R x n_test.");
  }

  // -----------------------------
  // Parse cutpoints, cat levels, graphs, nesting
  // -----------------------------
  std::vector<std::set<double>> cutpoints;
  if (p_cont > 0) parse_cutpoints(cutpoints, p_cont, cutpoints_list);

  std::vector<std::set<int>> cat_levels;
  std::vector<std::vector<edge>> edges;
  if (p_cat > 0) {
    parse_cat_levels(cat_levels, p_cat, cat_levels_list);
    parse_graphs(edges, p_cat, edge_mat_list);
  }

  std::vector<hi_lo_map> nesting;
  std::vector<edge_map> nest_graph_in;
  std::vector<edge_map> nest_graph_out;
  std::vector<std::map<int, std::set<int>>> nest_graph_components;
  parse_nesting(nesting, nest_graph_in, nest_graph_out, nest_graph_components,
                p_cont, cov_ensm, cat_levels, nest_list);

  // -----------------------------
  // data_info
  // -----------------------------
  data_info di_train;
  di_train.n = n_train;
  di_train.R = Rens;
  di_train.p_cont = p_cont;
  di_train.p_cat  = p_cat;
  di_train.p      = p;
  di_train.z = tZ_train.begin();
  if (p_cont > 0) di_train.x_cont = tX_cont_train.begin();
  if (p_cat  > 0) di_train.x_cat  = tX_cat_train.begin();

  data_info di_test;
  di_test.n = 0;
  if (n_test > 0) {
    di_test.n = n_test;
    di_test.R = Rens;
    di_test.p_cont = p_cont;
    di_test.p_cat  = p_cat;
    di_test.p      = p;
    di_test.z = tZ_test.begin();
    if (p_cont > 0) di_test.x_cont = tX_cont_test.begin();
    if (p_cat  > 0) di_test.x_cat  = tX_cat_test.begin();
  }

  // -----------------------------
  // Splitting probabilities per (j,r)
  // -----------------------------
  std::vector<std::vector<std::vector<int>>> var_count(q,
    std::vector<std::vector<int>>(Rens, std::vector<int>(p, 0)));
  std::vector<std::vector<int>> rule_count(q, std::vector<int>(Rens, 0));
  std::vector<std::vector<std::vector<double>>> theta(q,
    std::vector<std::vector<double>>(Rens, std::vector<double>(p, 0.0)));
  std::vector<std::vector<double>> u(q, std::vector<double>(Rens, 0.0));

  for (int j = 0; j < q; ++j) {
    for (int r = 0; r < Rens; ++r) {
      int n_avail = 0;
      for (int k = 0; k < p; ++k) {
        if (cov_ensm(k, r) == 1) { theta[j][r][k] = 1.0; ++n_avail; }
      }
      if (n_avail == 0) stop("Each ensemble must have >= 1 covariate.");
      for (int k = 0; k < p; ++k) theta[j][r][k] /= (double)n_avail;
      u[j][r] = 1.0 / (1.0 + (double)n_avail);
    }
  }

  std::vector<std::vector<tree_prior_info>> tree_pi_vec(q,
    std::vector<tree_prior_info>(Rens));

  for (int j = 0; j < q; ++j) {
    for (int r = 0; r < Rens; ++r) {
      if (p_cont > 0) tree_pi_vec[j][r].cutpoints = &cutpoints;
      if (p_cat > 0) {
        tree_pi_vec[j][r].cat_levels = &cat_levels;
        tree_pi_vec[j][r].edges = &edges;
        tree_pi_vec[j][r].graph_cut_type = graph_cut_type;
        tree_pi_vec[j][r].nesting = &nesting;
        tree_pi_vec[j][r].nest_in = &(nest_graph_in[r]);
        tree_pi_vec[j][r].nest_out = &(nest_graph_out[r]);
        tree_pi_vec[j][r].nest_components = &(nest_graph_components[r]);
      }
      tree_pi_vec[j][r].nest_v = nest_v;
      tree_pi_vec[j][r].nest_v_option = nest_v_option;
      tree_pi_vec[j][r].nest_c = nest_c;

      tree_pi_vec[j][r].theta = &(theta[j][r]);
      tree_pi_vec[j][r].var_count = &(var_count[j][r]);
      tree_pi_vec[j][r].rule_count = &(rule_count[j][r]);

      tree_pi_vec[j][r].alpha = alpha_vec[r];
      tree_pi_vec[j][r].beta  = beta_vec[r];
      tree_pi_vec[j][r].mu0   = mu0_vec[r];

      tree_pi_vec[j][r].tau = 1.0; // overwritten each iter by HS
    }
  }

  // -----------------------------
  // Flat Arrays and Tree offsets calculation
  // -----------------------------
  int total_trees_per_outcome = 0;
  std::vector<int> ensm_offsets(Rens, 0);
  for (int r = 0; r < Rens; ++r) {
    ensm_offsets[r] = total_trees_per_outcome;
    total_trees_per_outcome += M_vec[r];
  }
  
  int total_trees = q * total_trees_per_outcome;

  std::vector<tree> t_vec(total_trees);
  std::vector<suff_stat> ss_train_vec(total_trees);
  std::vector<suff_stat> ss_test_vec(total_trees);

  for (int j = 0; j < q; ++j) {
    for (int r = 0; r < Rens; ++r) {
      for (int m = 0; m < M_vec[r]; ++m) {
        int idx = j * total_trees_per_outcome + ensm_offsets[r] + m;
        tree_traversal(ss_train_vec[idx], t_vec[idx], di_train);
        if (n_test > 0) tree_traversal(ss_test_vec[idx], t_vec[idx], di_test);
      }
    }
  }

  // -----------------------------
  // Armadillo views: Y (n x q)
  // -----------------------------
  arma::mat Yt(n_train, q);
  {
    arma::mat Yqxn(tY_train.begin(), q, n_train, false);
    Yt = Yqxn.t();
  }

  arma::mat F = arma::zeros<arma::mat>(n_train, q);
  arma::mat E = Yt - F;

  // -----------------------------
  // Omega state init
  // -----------------------------
  Rcpp::List Omega_state;
  Omega_state["Omega"] = arma::eye(q, q);

  // -----------------------------
  // HS leaf scales init
  // -----------------------------
  arma::mat lambda2_leaf = arma::ones<arma::mat>(q, Rens);
  arma::mat nu_leaf      = arma::ones<arma::mat>(q, Rens);

  double tauB2 = std::max(EPS, sigma_B_n * sigma_B_n);
  double xi    = 1.0;

  // -----------------------------
  // Outputs
  // -----------------------------
  int total_draws = burn + (nd * thin);
  arma::mat fit_train_mean = arma::zeros<arma::mat>(n_train, q);
  arma::cube beta_train_mean(n_train, Rens, q, arma::fill::zeros);

  arma::mat fit_test_mean(1,1);
  arma::cube beta_test_mean(1,1,1);
  if (n_test > 0) {
    fit_test_mean = arma::zeros<arma::mat>(n_test, q);
    beta_test_mean = arma::cube(n_test, Rens, q, arma::fill::zeros);
  }

  arma::cube fit_train_samp(1,1,1);
  arma::cube fit_test_samp(1,1,1);
  arma::cube Omega_samp(1,1,1);
  if (save_samples) {
    fit_train_samp = arma::cube(nd, n_train, q, arma::fill::zeros);
    if (n_test > 0) fit_test_samp = arma::cube(nd, n_test, q, arma::fill::zeros);
    Omega_samp = arma::cube(nd, q, q, arma::fill::zeros);
  }

  arma::vec tauB2_samples(total_draws, arma::fill::zeros);
  Rcpp::List tree_draws(nd);
  set_str_conversion set_str;

  std::vector<double> rwork(n_train);
  int accept = 0;
  int sample_index = 0;

  // -----------------------------
  // Main MCMC
  // -----------------------------
  for (int iter = 0; iter < total_draws; ++iter) {

    if (iter % print_every == 0) {
      Rcpp::checkUserInterrupt();
      if (verbose) {
        std::string phase = (iter < burn) ? "Warmup" : "Sampling";
        Rcpp::Rcout << "  MCMC Iteration: " << (iter+1)
                    << " of " << total_draws << "; " << phase << "\n";
      }
    }

    // ---- set tau(j,r) from HS scales ----
    for (int j = 0; j < q; ++j) {
      for (int r = 0; r < Rens; ++r) {
        double Mr = std::max(1.0, (double)M_vec[r]);
        double tau_jr = std::sqrt(std::max(EPS, tauB2 * lambda2_leaf(j,r) / Mr));
        tree_pi_vec[j][r].tau = tau_jr;
      }
    }

    // ---- update outcomes j ----
    for (int j = 0; j < q; ++j) {

      arma::mat Omega = as<arma::mat>(Omega_state["Omega"]);
      double omjj = Omega(j,j);
      if (!arma::is_finite(omjj) || omjj <= 0.0) stop("Omega(j,j) must be positive.");

      arma::vec tmp = E * Omega.row(j).t();
      arma::vec offset = (tmp - E.col(j) * omjj) / omjj;

      for (int i = 0; i < n_train; ++i) rwork[i] = E(i,j) + offset[i];
      di_train.rp = rwork.data();

      double sigma_j = std::sqrt(1.0 / omjj);

      for (int r = 0; r < Rens; ++r) {

        for (int m = 0; m < M_vec[r]; ++m) {
          int idx = j * total_trees_per_outcome + ensm_offsets[r] + m;

          // remove old tree
          for (suff_stat_it l_it = ss_train_vec[idx].begin();
               l_it != ss_train_vec[idx].end(); ++l_it) {

            int nid = l_it->first;
            double mu_old = t_vec[idx].get_ptr(nid)->get_mu();

            for (int_it it2 = l_it->second.begin(); it2 != l_it->second.end(); ++it2) {
              int ii = *it2;
              double z = di_train.z[r + ii * Rens];
              rwork[ii] += z * mu_old;
              F(ii, j)  -= z * mu_old;
            }
          }

          update_tree_multi(t_vec[idx],
                            ss_train_vec[idx],
                            ss_test_vec[idx],
                            accept, r, sigma_j,
                            di_train, di_test,
                            tree_pi_vec[j][r],
                            gen);

          // add new tree
          for (suff_stat_it l_it = ss_train_vec[idx].begin();
               l_it != ss_train_vec[idx].end(); ++l_it) {

            int nid = l_it->first;
            double mu_new = t_vec[idx].get_ptr(nid)->get_mu();

            for (int_it it2 = l_it->second.begin(); it2 != l_it->second.end(); ++it2) {
              int ii = *it2;
              double z = di_train.z[r + ii * Rens];
              rwork[ii] -= z * mu_new;
              F(ii, j)  += z * mu_new;
            }
          }
        }

        if (sparse) {
          // Hard-coded 1.0 for the removed b_u ghost argument
          update_theta_u_subset(theta[j][r], u[j][r], var_count[j][r], a_u, 1.0, gen);
        }
      }

      E.col(j) = Yt.col(j) - F.col(j);
    }

    // ---- update Omega given residuals E ----
    // Hard-coded 1.0 for the removed omega_a_tau ghost argument
    Omega_state = update_Omega_cpp(E, Omega_state,
                                  omega_method, q_small, v0_gauss,
                                  omega_a0, omega_b0,
                                  1.0, omega_b_tau,
                                  lam2_min, tau2_min,
                                  jitter, max_tries);

    // ---- horseshoe update on leaf scales ----
    arma::mat ssq_mat(q, Rens, arma::fill::zeros);
    arma::imat K_mat(q, Rens, arma::fill::zeros);

    for (int j = 0; j < q; ++j) {
      for (int r = 0; r < Rens; ++r) {
        double ssq = 0.0;
        int K = 0;
        int start_idx = j * total_trees_per_outcome + ensm_offsets[r];
        
        leaf_sumsq_count_one_group(t_vec, ss_train_vec, start_idx, M_vec[r], ssq, K);
        ssq_mat(j,r) = ssq;
        K_mat(j,r)   = K;

        double shape = 0.5 * ((double)K + 1.0);
        double rate  = (1.0 / std::max(EPS, nu_leaf(j,r)))
                     + 0.5 * ((double)M_vec[r]) * ssq / std::max(EPS, tauB2);

        double lam2 = leaf_hs::rinvgamma_rate(shape, rate);
        lam2 = std::max(EPS, lam2);
        lambda2_leaf(j,r) = lam2;

        nu_leaf(j,r) = leaf_hs::rinvgamma_rate(1.0, 1.0 + 1.0/lam2);
      }
    }

    double sum_term = 0.0;
    double Ktot = 0.0;
    for (int j = 0; j < q; ++j) {
      for (int r = 0; r < Rens; ++r) {
        sum_term += 0.5 * ((double)M_vec[r]) * ssq_mat(j,r) / std::max(EPS, (double)lambda2_leaf(j,r));
        Ktot += (double)K_mat(j,r);
      }
    }

    double shape_tau = 0.5 * (Ktot + 1.0);
    double rate_tau  = (1.0 / std::max(EPS, xi)) + sum_term;

    tauB2 = leaf_hs::rinvgamma_rate(shape_tau, rate_tau);
    tauB2 = std::max(EPS, tauB2);

    xi = leaf_hs::rinvgamma_rate(1.0,
                        (1.0 / std::max(EPS, sigma_B_n * sigma_B_n)) + (1.0 / tauB2));

    tauB2_samples(iter) = tauB2;

    // ---- save draws ----
    if (iter >= burn && ((iter - burn) % thin == 0)) {
      sample_index = (int)((iter - burn) / thin);

      fit_train_mean += F;

      for (int j = 0; j < q; ++j) {
        for (int r = 0; r < Rens; ++r) {
          for (int m = 0; m < M_vec[r]; ++m) {
            int idx = j * total_trees_per_outcome + ensm_offsets[r] + m;
            
            for (suff_stat_it l_it = ss_train_vec[idx].begin();
                 l_it != ss_train_vec[idx].end(); ++l_it) {
              int nid = l_it->first;
              double mu = t_vec[idx].get_ptr(nid)->get_mu();
              for (int_it it2 = l_it->second.begin(); it2 != l_it->second.end(); ++it2) {
                int ii = *it2;
                beta_train_mean(ii, r, j) += mu;
              }
            }
          }
        }
      }

      if (n_test > 0) {
        arma::mat Ftest = arma::zeros<arma::mat>(n_test, q);
        for (int j = 0; j < q; ++j) {
          for (int r = 0; r < Rens; ++r) {
            for (int m = 0; m < M_vec[r]; ++m) {
              int idx = j * total_trees_per_outcome + ensm_offsets[r] + m;
              
              for (suff_stat_it l_it = ss_test_vec[idx].begin();
                   l_it != ss_test_vec[idx].end(); ++l_it) {
                int nid = l_it->first;
                double mu = t_vec[idx].get_ptr(nid)->get_mu();
                for (int_it it2 = l_it->second.begin(); it2 != l_it->second.end(); ++it2) {
                  int ii = *it2;
                  double z = di_test.z[r + ii * Rens];
                  Ftest(ii, j) += z * mu;
                  beta_test_mean(ii, r, j) += mu;
                }
              }
            }
          }
        }
        fit_test_mean += Ftest;

        if (save_samples) {
          for (int j = 0; j < q; ++j)
            for (int ii = 0; ii < n_test; ++ii)
              fit_test_samp(sample_index, ii, j) = Ftest(ii, j);
        }
      }

      if (save_samples) {
        for (int j = 0; j < q; ++j)
          for (int ii = 0; ii < n_train; ++ii)
            fit_train_samp(sample_index, ii, j) = F(ii, j);

        arma::mat Om = as<arma::mat>(Omega_state["Omega"]);
        for (int a = 0; a < q; ++a)
          for (int b = 0; b < q; ++b)
            Omega_samp(sample_index, a, b) = Om(a,b);
      }

      if (save_trees) {
        Rcpp::List tmp_tree_draws(q);
        for (int j = 0; j < q; ++j) {
          Rcpp::List per_outcome(Rens);
          for (int r = 0; r < Rens; ++r) {
            Rcpp::CharacterVector tree_string_vec(M_vec[r]);
            for (int m = 0; m < M_vec[r]; ++m) {
              int idx = j * total_trees_per_outcome + ensm_offsets[r] + m;
              tree_string_vec[m] = write_tree(t_vec[idx], tree_pi_vec[j][r], set_str);
            }
            per_outcome[r] = tree_string_vec;
          }
          tmp_tree_draws[j] = per_outcome;
        }
        tree_draws[sample_index] = tmp_tree_draws;
      }
    }
  }

  fit_train_mean /= (double)nd;
  beta_train_mean /= (double)nd;
  if (n_test > 0) {
    fit_test_mean /= (double)nd;
    beta_test_mean /= (double)nd;
  }

  Rcpp::List out;
  out["fit_train_mean"]  = fit_train_mean;
  out["beta_train_mean"] = beta_train_mean;

  if (n_test > 0) {
    out["fit_test_mean"]  = fit_test_mean;
    out["beta_test_mean"] = beta_test_mean;
  }

  out["Omega_state"] = Omega_state;

  out["tauB2"]        = tauB2_samples;
  out["lambda2_leaf"] = lambda2_leaf;
  out["nu_leaf"]      = nu_leaf;

  if (save_samples) {
    out["fit_train"] = fit_train_samp;
    if (n_test > 0) out["fit_test"] = fit_test_samp;
    out["Omega"] = Omega_samp;
  }
  if (save_trees) out["trees"] = tree_draws;

  return out;
}

// Unity Build Includes
#include "data_parsing_funs.inc"
#include "graph_funs.inc"
#include "rng.inc"
#include "rule_funs.inc"
#include "tree.inc"
#include "update_omega_ghs.inc"
#include "update_tree_multi.inc"
#include "funs.inc"
