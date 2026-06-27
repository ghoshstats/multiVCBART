#include "funs.h"

void update_theta_u(std::vector<double> &theta,
                    double &u,
                    std::vector<int> &var_count,
                    double a_u,
                    double b_u,
                    RNG &gen)
{
  int p = theta.size();
  double sum_theta = 0.0;
  
  // 1. Draw independent Gamma variables for the exact Dirichlet posterior
  for(int j = 0; j < p; ++j){
    double shape = a_u + (double) var_count[j];
    theta[j] = R::rgamma(shape, 1.0); 
    sum_theta += theta[j];
  }
  
  // 2. Normalize so probabilities strictly sum to 1.0
  for(int j = 0; j < p; ++j){
    theta[j] /= sum_theta;
  }
  
  // Neutralize u to prevent unused variable warnings
  u = 0.0; 
}

void tree_traversal(suff_stat &ss, tree &t, data_info &di)
{
  double* xx_cont = 0;
  int* xx_cat = 0;
  tree::tree_cp bn;
  int nid;
  ss.clear(); // clear out the sufficient statistic map
  
  tree::npv bnv;
  t.get_bots(bnv);
  suff_stat_it ss_it; // used to look up to which element of ss a particular bottom node corresponds
  
  // add an element to suff stat map for each bottom node
  for(tree::npv_it it = bnv.begin(); it != bnv.end(); ++it){
    nid = (*it)->get_nid(); // bnv is a vector of pointers. it points to elements of bnv so we need (*it) to access the members of these elements
    // add element to sufficient stats map with key = nid, value = empty vector to hold index of observations assigned to this node
    ss.insert(std::pair<int,std::vector<int>>(nid,std::vector<int>()));
  }
  
  // ready to loop over all observations
  for(int i = 0; i < di.n; i++){
    if(di.x_cont != 0) xx_cont = di.x_cont + i * di.p_cont;
    if(di.x_cat != 0) xx_cat = di.x_cat + i * di.p_cat;
    bn = t.get_bn(xx_cont, xx_cat);
    if(bn == 0){
      Rcpp::Rcout << "i = " << i << std::endl;
      t.print();
      Rcpp::stop("[tree_traversal]: could not find bottom node!");
    }
    else{
      nid = bn->get_nid();
      if(ss.count(nid) != 1) Rcpp::stop("[tree_traversal]: bottom node not included in sufficient statistic map!"); // should never be encountered
      else{
        ss_it = ss.find(nid); // iterator now set to element of ss corresponding to the bottom node holding observation i
        ss_it->second.push_back(i);
      } // closes if/else checking that i-th observation's bottom node was in our map
    } // closes if/else checking that i-th observation maps to valid bottom node
  } // closes loop over all observation
}

void update_theta_u_subset(std::vector<double> &theta,
                           double &u,
                           std::vector<int> &var_count,
                           double a_u,
                           double b_u,
                           RNG &gen)
{
  int p = theta.size();
  double sum_theta = 0.0;
  
  // 1. Draw Gamma variables only for available predictors
  for(int j = 0; j < p; ++j){
    if(theta[j] > 0){
      double shape = a_u + (double) var_count[j];
      theta[j] = R::rgamma(shape, 1.0);
      sum_theta += theta[j];
    }
  }
  
  // 2. Normalize the active predictors
  for(int j = 0; j < p; ++j){
    if(theta[j] > 0){
      theta[j] /= sum_theta;
    }
  }
  
  u = 0.0;
}

std::string set_to_string(std::set<int> set_)
{
  std::string str = "";
  for(set_it it = set_.begin(); it != set_.end(); ++it){
    if(it == set_.begin()) str += std::to_string(*it);
    else str += "," + std::to_string(*it);
  }
  return str;
}

std::set<int> string_to_set(std::string str, set_str_conversion &set_str)
{
  if(set_str.str_to_set.count(str) == 1) return set_str.str_to_set[str];
  std::set<int> out;
  std::string s = "";
  for(int i = 0; i < (int) str.size(); ++i){
    if(str[i] != ','){
      s += str[i];
    } else{
      out.insert(std::stoi(s));
      s = "";
    }
  }
  if(s.size() > 0) out.insert(std::stoi(s));
  set_str.str_to_set[str] = out;
  set_str.set_to_str[out] = str;
  return out;
}

std::string write_node(tree::tree_p n,
                       bool is_cat,
                       bool is_leaf,
                       int v,
                       double c,
                       std::set<int> lset,
                       std::set<int> rset,
                       double mu,
                       tree_prior_info &tree_pi,
                       set_str_conversion &set_str)
{
  int nid = n->get_nid();
  int depth = n->get_depth();
  std::string out = "";
  out += "nid=" + std::to_string(nid) + ";";
  out += "depth=" + std::to_string(depth) + ";";
  out += "is_leaf=" + std::to_string(is_leaf) + ";";
  out += "is_cat=" + std::to_string(is_cat) + ";";
  out += "v=" + std::to_string(v) + ";";
  out += "c=" + std::to_string(c) + ";";
  if(is_cat){
    std::string l_str = "";
    std::string r_str = "";
    if(set_str.set_to_str.count(lset) == 1) l_str = set_str.set_to_str[lset];
    else{
      l_str = set_to_string(lset);
      set_str.set_to_str[lset] = l_str;
      set_str.str_to_set[l_str] = lset;
    }
    if(set_str.set_to_str.count(rset) == 1) r_str = set_str.set_to_str[rset];
    else{
      r_str = set_to_string(rset);
      set_str.set_to_str[rset] = r_str;
      set_str.str_to_set[r_str] = rset;
    }
    out += "lset=" + l_str + ";";
    out += "rset=" + r_str + ";";
  }
  out += "mu=" + std::to_string(mu) + ";";
  return out;
}

std::string write_tree(tree &t,
                       tree_prior_info &tree_pi,
                       set_str_conversion &set_str)
{
  std::string out = "";
  tree::npv nodes;
  t.get_nodes(nodes);
  for(size_t i = 0; i < nodes.size(); ++i){
    tree::tree_p n = nodes[i];
    bool is_leaf = n->is_bot();
    bool is_cat = n->get_is_cat();
    int v = -1;
    double c = 0.0;
    std::set<int> lset;
    std::set<int> rset;
    if(!is_leaf){
      if(!is_cat){
        v = n->get_v_aa();
        c = n->get_c();
      } else{
        v = n->get_v_cat();
        lset = n->get_l_vals();
        rset = n->get_r_vals();
      }
    }
    double mu = n->get_mu();
    out += write_node(n, is_cat, is_leaf, v, c, lset, rset, mu, tree_pi, set_str);
    out += "\n";
  }
  return out;
}
