#include "update_tree.h"
#include <cmath> 

// --------------------------------------------
//  Multivariate (Z-weighted) leaf-jump updates
// --------------------------------------------

void compute_jump_posterior_multi(std::map<int, jump_post> &jp_map,
                                 suff_stat &ss,
                                 int &r,
                                 double &sigma,
                                 data_info &di,
                                 tree_prior_info &tree_pi)
{
  // reminder posterior of jump is N(P^-1 Theta, P^-1)
  int i = 0;
  double z = 0.0;
  jp_map.clear();

  for(suff_stat_it l_it = ss.begin(); l_it != ss.end(); ++l_it){
    jp_map.insert(std::pair<int, jump_post>(l_it->first, jump_post()));
    std::map<int, jump_post>::iterator jp_it = jp_map.find(l_it->first);

    jp_it->second.P     = 1.0/pow(tree_pi.tau, 2.0);      // prior leaf precision
    jp_it->second.Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0); // contribution from prior leaf

    if(l_it->second.size() > 0){
      for(int_it it = l_it->second.begin(); it != l_it->second.end(); ++it){
        i = *it;
        z = di.z[r + i*di.R];
        jp_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
        jp_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
      }
    }
  }
}

double compute_lil(int &nid, std::map<int, jump_post> &jp_map)
{
  if(jp_map.count(nid) == 0) Rcpp::stop("[compute_lil]: Did not find element for node in jp_map");
  std::map<int, jump_post>::iterator jp_it = jp_map.find(nid);
  return(-0.5 * log(jp_it->second.P) + 0.5 * pow(jp_it->second.Theta, 2.0)/jp_it->second.P);
}

void draw_mu(tree &t, std::map<int, jump_post> &jp_map, RNG &gen)
{
  double post_mean;
  double post_sd;
  for(std::map<int,jump_post>::iterator l_it = jp_map.begin(); l_it != jp_map.end(); ++l_it){
    post_mean = l_it->second.Theta/l_it->second.P;
    post_sd   = sqrt(1.0/l_it->second.P);
    t.get_ptr(l_it->first)->set_mu(gen.normal(post_mean, post_sd));
  }
}


// ---------------------------------------------------
// grow: in-place ss update + jp for proposed children
// ---------------------------------------------------
void compute_ss_grow_multi(suff_stat &ss,
                          std::map<int, jump_post> &jp_map,
                          int &nx_nid,
                          rule_t &rule,
                          int &r,
                          double &sigma,
                          data_info &di,
                          tree_prior_info &tree_pi)
{
  int i = 0;
  double z = 0.0;
  int nxl_nid = 2*nx_nid;
  int nxr_nid = 2*nx_nid+1;

  // check that our data structures have element for nx but not for nxl or nxr (the proposed children)
  if(ss.count(nx_nid) == 0 || ss.count(nxl_nid) == 1 || ss.count(nxr_nid) == 1)
    Rcpp::stop("[compute_ss_grow_multi]: something is wrong with ss_train");
  if(jp_map.count(nx_nid) == 0 || jp_map.count(nxl_nid) == 1 || jp_map.count(nxr_nid) == 1)
    Rcpp::stop("[compute_ss_grow_multi]: something is wrong with jp_map!");

  suff_stat_it nx_it = ss.find(nx_nid); // points to nx's element in ss

  // add elements to ss and jp_map for nxl
  ss.insert(std::pair<int, std::vector<int>>(nxl_nid, std::vector<int>()));
  suff_stat_it nxl_it = ss.find(nxl_nid);

  jp_map.insert(std::pair<int, jump_post>(nxl_nid, jump_post()));
  std::map<int, jump_post>::iterator jpl_it = jp_map.find(nxl_nid);
  jpl_it->second.P     = 1.0/pow(tree_pi.tau, 2.0);
  jpl_it->second.Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);

  // add elements to ss and jp_map for nxr
  ss.insert(std::pair<int, std::vector<int>>(nxr_nid, std::vector<int>()));
  suff_stat_it nxr_it = ss.find(nxr_nid);

  jp_map.insert(std::pair<int, jump_post>(nxr_nid, jump_post()));
  std::map<int, jump_post>::iterator jpr_it = jp_map.find(nxr_nid);
  jpr_it->second.P     = 1.0/pow(tree_pi.tau, 2.0);
  jpr_it->second.Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);

  if(nx_it->second.size() > 0){
    if(!rule.is_cat){
      double xx_cont = 0.0;
      for(std::vector<int>::iterator it = nx_it->second.begin(); it != nx_it->second.end(); ++it){
        i = *it;
        z = di.z[r + i*di.R];
        xx_cont = *(di.x_cont + i*di.p_cont + rule.v_aa);

        if(xx_cont < rule.c){
          nxl_it->second.push_back(i);
          jpl_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
          jpl_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
        } else if(xx_cont >= rule.c){
          nxr_it->second.push_back(i);
          jpr_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
          jpr_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
        } else{
          Rcpp::Rcout << "  i = " << i << " v = " << rule.v_aa+1
                      << "  value = " << xx_cont << " cutpoint = " << rule.c << std::endl;
          Rcpp::stop("[compute_ss_grow_multi]: could not assign obs to left/right child!");
        }
      }
    } else{
      int xx_cat = 0;
      for(std::vector<int>::iterator it = nx_it->second.begin(); it != nx_it->second.end(); ++it){
        i = *it;
        z = di.z[r + i*di.R];
        xx_cat = *(di.x_cat + i*di.p_cat + rule.v_cat);

        int l_count = rule.l_vals.count(xx_cat);
        int r_count = rule.r_vals.count(xx_cat);

        if(l_count != r_count){
          if(l_count == 1){
            nxl_it->second.push_back(i);
            jpl_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
            jpl_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
          } else{
            nxr_it->second.push_back(i);
            jpr_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
            jpr_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
          }
        } else{
          Rcpp::stop("[compute_ss_grow_multi]: could not assign obs in categorical split!");
        }
      }
    }
  }
}


// ---------------------------------------------------
// helper: test-set ss grow/prune (no jp bookkeeping)
// ---------------------------------------------------
void compute_ss_grow(suff_stat &ss, int &nx_nid, rule_t &rule, data_info &di)
{
  int i ;
  int nxl_nid = 2*nx_nid;
  int nxr_nid = 2*nx_nid+1;

  if(ss.count(nx_nid) == 0 || ss.count(nxl_nid) == 1 || ss.count(nxr_nid) == 1)
    Rcpp::stop("[compute_ss_grow]: something is wrong with ss_test");

  suff_stat_it nx_it = ss.find(nx_nid);

  ss.insert(std::pair<int, std::vector<int>>(nxl_nid, std::vector<int>()));
  suff_stat_it nxl_it = ss.find(nxl_nid);

  ss.insert(std::pair<int, std::vector<int>>(nxr_nid, std::vector<int>()));
  suff_stat_it nxr_it = ss.find(nxr_nid);

  if(nx_it->second.size() > 0){
    if(!rule.is_cat){
      double xx_cont = 0.0;
      for(std::vector<int>::iterator it = nx_it->second.begin(); it != nx_it->second.end(); ++it){
        i = *it;
        xx_cont = *(di.x_cont + i*di.p_cont + rule.v_aa);

        if(xx_cont < rule.c) nxl_it->second.push_back(i);
        else if(xx_cont >= rule.c) nxr_it->second.push_back(i);
        else Rcpp::stop("[compute_ss_grow]: could not assign obs in axis-aligned split!");
      }
    } else{
      int xx_cat = 0;
      for(std::vector<int>::iterator it = nx_it->second.begin(); it != nx_it->second.end(); ++it){
        i = *it;
        xx_cat = *(di.x_cat + i*di.p_cat + rule.v_cat);

        int l_count = rule.l_vals.count(xx_cat);
        int r_count = rule.r_vals.count(xx_cat);

        if(l_count != r_count){
          if(l_count == 1) nxl_it->second.push_back(i);
          else nxr_it->second.push_back(i);
        } else{
          Rcpp::stop("[compute_ss_grow]: could not assign obs in categorical split!");
        }
      }
    }
  }
}

void compute_ss_prune(suff_stat &ss, int &nxl_nid, int &nxr_nid, int &nx_nid, data_info &di)
{
  if(ss.count(nxl_nid) == 0 || ss.count(nxr_nid) == 0 || ss.count(nx_nid) == 1)
    Rcpp::stop("[compute_ss_prune]: something is wrong with ss_test");

  suff_stat_it nxl_it = ss.find(nxl_nid);
  suff_stat_it nxr_it = ss.find(nxr_nid);

  ss.insert(std::pair<int, std::vector<int>>(nx_nid, std::vector<int>()));
  suff_stat_it nx_it = ss.find(nx_nid);

  if(nxl_it->second.size() > 0){
    for(std::vector<int>::iterator it = nxl_it->second.begin(); it != nxl_it->second.end(); ++it)
      nx_it->second.push_back(*it);
  }
  if(nxr_it->second.size() > 0){
    for(std::vector<int>::iterator it = nxr_it->second.begin(); it != nxr_it->second.end(); ++it)
      nx_it->second.push_back(*it);
  }
}


// ---------------------------------------------------
// prune: in-place ss update + jp for merged parent
// ---------------------------------------------------
void compute_ss_prune_multi(suff_stat &ss,
                           std::map<int, jump_post> &jp_map,
                           int &nxl_nid,
                           int &nxr_nid,
                           int &nx_nid,
                           int &r,
                           double &sigma,
                           data_info &di,
                           tree_prior_info &tree_pi)
{
  int i = 0;
  double z = 0.0;

  if(ss.count(nxl_nid) == 0 || ss.count(nxr_nid) == 0 || ss.count(nx_nid) == 1)
    Rcpp::stop("[compute_ss_prune_multi]: something's wrong with ss_train");
  if(jp_map.count(nxl_nid) == 0 || jp_map.count(nxr_nid) == 0 || jp_map.count(nx_nid) == 1)
    Rcpp::stop("[compute_ss_prune_multi]: something's wrong with jp_map");

  suff_stat_it nxl_it = ss.find(nxl_nid);
  suff_stat_it nxr_it = ss.find(nxr_nid);

  ss.insert(std::pair<int, std::vector<int>>(nx_nid, std::vector<int>()));
  suff_stat_it nx_it = ss.find(nx_nid);

  jp_map.insert(std::pair<int, jump_post>(nx_nid, jump_post()));
  std::map<int, jump_post>::iterator jp_it = jp_map.find(nx_nid);
  jp_it->second.P     = 1.0/pow(tree_pi.tau, 2.0);
  jp_it->second.Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);

  if(nxl_it->second.size() > 0){
    for(std::vector<int>::iterator it = nxl_it->second.begin(); it != nxl_it->second.end(); ++it){
      i = *it;
      z = di.z[r + i*di.R];
      nx_it->second.push_back(i);
      jp_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
      jp_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
    }
  }
  if(nxr_it->second.size() > 0){
    for(std::vector<int>::iterator it = nxr_it->second.begin(); it != nxr_it->second.end(); ++it){
      i = *it;
      z = di.z[r + i*di.R];
      nx_it->second.push_back(i);
      jp_it->second.P     += pow(z, 2.0)/pow(sigma, 2.0);
      jp_it->second.Theta += z * di.rp[i]/pow(sigma, 2.0);
    }
  }
}


// ---------------------------------------------------
// GROW / PRUNE moves + MH accept/reject
// ---------------------------------------------------
void grow_tree_multi(tree &t,
                     suff_stat &ss_train,
                     suff_stat &ss_test,
                     std::map<int, jump_post> &jp_map,
                     int &accept,
                     int &r,
                     double &sigma,
                     data_info &di_train,
                     data_info &di_test,
                     tree_prior_info &tree_pi,
                     RNG &gen)
{
  std::vector<int> bn_nid_vec;
  for(suff_stat_it ss_it = ss_train.begin(); ss_it != ss_train.end(); ++ss_it)
    bn_nid_vec.push_back(ss_it->first);

  int ni = floor(gen.uniform() * bn_nid_vec.size());
  int nx_nid = bn_nid_vec[ni];
  tree::tree_p nx = t.get_ptr(nx_nid);
  tree::tree_cp nxp = nx->get_p();

  double q_grow_old  = tree_pi.prob_b;
  double q_prune_new = 1.0 - tree_pi.prob_b;

  int nleaf_old = t.get_nbots();
  int nnog_old  = t.get_nnogs();
  int nnog_new  = nnog_old;

  if(nxp == 0){
    q_grow_old = 1.0;
    nnog_new = 1;
  } else if(nxp->is_nog()){
    nnog_new = nnog_old;
  } else{
    nnog_new = 1 + nnog_old;
  }

  double log_trans_ratio =
    (log(q_prune_new) - log((double) nnog_new)) - (log(q_grow_old) - log((double) nleaf_old));

  double p_grow_nx  = tree_pi.alpha/pow(1.0 + (double) nx->get_depth(), tree_pi.beta);
  double p_grow_nxl = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta);
  double p_grow_nxr = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta);

  double log_prior_ratio =
    log(p_grow_nx) + log(1.0 - p_grow_nxl) + log(1.0 - p_grow_nxr) - log(1.0 - p_grow_nx);

  rule_t rule;
  draw_rule(rule, t, nx_nid, di_train, tree_pi, gen);

  compute_ss_grow_multi(ss_train, jp_map, nx_nid, rule, r, sigma, di_train, tree_pi);

  int nxl_nid = 2*nx_nid;
  int nxr_nid = 2*nx_nid+1;

  double nxl_lil = compute_lil(nxl_nid, jp_map);
  double nxr_lil = compute_lil(nxr_nid, jp_map);
  double nx_lil  = compute_lil(nx_nid,  jp_map);

  double log_like_ratio =
    nxl_lil + nxr_lil - nx_lil - 0.5 * log(2.0 * M_PI * pow(tree_pi.tau, 2.0)) - 0.5 * pow(tree_pi.mu0/tree_pi.tau, 2.0);

  double log_alpha = log_like_ratio + log_prior_ratio + log_trans_ratio;
  if(log_alpha > 0) log_alpha = 0.0;

  if(gen.log_uniform() <= log_alpha){
    ++(*tree_pi.rule_count);
    if(!rule.is_cat){
      ++(tree_pi.var_count->at(rule.v_aa));
    } else{
      int v_raw = rule.v_cat + di_train.p_cont;
      ++(tree_pi.var_count->at(v_raw));
    }

    ss_train.erase(nx_nid);
    jp_map.erase(nx_nid);

    if(di_test.n > 0){
      compute_ss_grow(ss_test, nx_nid, rule, di_test);
      ss_test.erase(nx_nid);
    }

    t.birth(nx_nid, rule);
    accept = 1;
  } else{
    accept = 0;
    ss_train.erase(nxl_nid);
    ss_train.erase(nxr_nid);
    jp_map.erase(nxl_nid);
    jp_map.erase(nxr_nid);
  }
}

void prune_tree_multi(tree &t,
                      suff_stat &ss_train,
                      suff_stat &ss_test,
                      std::map<int, jump_post> &jp_map,
                      int &accept,
                      int &r,
                      double &sigma,
                      data_info &di_train,
                      data_info &di_test,
                      tree_prior_info &tree_pi,
                      RNG &gen)
{
  tree::npv nogs_vec;
  t.get_nogs(nogs_vec);

  int ni = floor(gen.uniform() * nogs_vec.size());
  tree::tree_p nx  = nogs_vec[ni];
  tree::tree_p nxl = nx->get_l();
  tree::tree_p nxr = nx->get_r();

  double q_prune_old = 1.0 - tree_pi.prob_b;
  double q_grow_new  = tree_pi.prob_b;

  tree::tree_p nxp = nx->get_p();
  if(nxp == 0) q_grow_new = 1.0;

  int nleaf_new = t.get_nbots() - 1;
  int nnog_old  = t.get_nnogs();

  double log_trans_ratio =
    (log(q_grow_new) - log(nleaf_new)) - (log(q_prune_old) - log(nnog_old));

  double p_grow_nx  = tree_pi.alpha/pow(1.0 + (double) nx->get_depth(), tree_pi.beta);
  double p_grow_nxl = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta);
  double p_grow_nxr = tree_pi.alpha/pow(2.0 + (double) nx->get_depth(), tree_pi.beta);

  double log_prior_ratio =
    log(1.0 - p_grow_nx) - (log(1.0 - p_grow_nxl) + log(1.0 - p_grow_nxr) + log(p_grow_nx));

  int nx_nid  = nx->get_nid();
  int nxl_nid = nxl->get_nid();
  int nxr_nid = nxr->get_nid();

  compute_ss_prune_multi(ss_train, jp_map, nxl_nid, nxr_nid, nx_nid, r, sigma, di_train, tree_pi);

  double nxl_lil = compute_lil(nxl_nid, jp_map);
  double nxr_lil = compute_lil(nxr_nid, jp_map);
  double nx_lil  = compute_lil(nx_nid,  jp_map);

  double log_like_ratio =
    nx_lil - nxl_lil - nxr_lil + 0.5 * log(2.0 * M_PI * pow(tree_pi.tau, 2.0)) + 0.5 * pow(tree_pi.mu0/tree_pi.tau, 2.0);

  double log_alpha = log_like_ratio + log_prior_ratio + log_trans_ratio;
  if(log_alpha > 0) log_alpha = 0.0;

  if(gen.log_uniform() <= log_alpha){
    accept = 1;
    --(*tree_pi.rule_count);

    if(!nx->get_is_cat()){
      --(tree_pi.var_count->at(nx->get_v_aa()));
    } else{
      int v_raw = di_train.p_cont + nx->get_v_cat();
      --(tree_pi.var_count->at(v_raw));
    }

    ss_train.erase(nxl_nid);
    ss_train.erase(nxr_nid);
    jp_map.erase(nxl_nid);
    jp_map.erase(nxr_nid);

    if(di_test.n > 0){
      compute_ss_prune(ss_test, nxl_nid, nxr_nid, nx_nid, di_train);
      ss_test.erase(nxl_nid);
      ss_test.erase(nxr_nid);
    }

    t.death(nx_nid);
  } else{
    accept = 0;
    ss_train.erase(nx_nid);
    jp_map.erase(nx_nid);
  }
}

void update_tree_multi(tree &t,
                       suff_stat &ss_train,
                       suff_stat &ss_test,
                       int &accept,
                       int &r,
                       double &sigma,
                       data_info &di_train,
                       data_info &di_test,
                       tree_prior_info &tree_pi,
                       RNG &gen)
{
  accept = 0;
  double PBx = tree_pi.prob_b;
  if(t.get_treesize() == 1) PBx = 1.0;

  std::map<int, jump_post> jp_map;
  compute_jump_posterior_multi(jp_map, ss_train, r, sigma, di_train, tree_pi);

  if(gen.uniform() < PBx)
    grow_tree_multi(t, ss_train, ss_test, jp_map, accept, r, sigma, di_train, di_test, tree_pi, gen);
  else
    prune_tree_multi(t, ss_train, ss_test, jp_map, accept, r, sigma, di_train, di_test, tree_pi, gen);

  draw_mu(t, jp_map, gen);
}
