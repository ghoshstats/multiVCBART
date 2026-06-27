#include "data_parsing_funs.h"

// -------------------------
// cutpoints
// -------------------------
void parse_cutpoints(std::vector<std::set<double>> &cutpoints,
                     int p_cont,
                     Rcpp::Nullable<Rcpp::List> &cutpoints_list)
{
  cutpoints.clear();
  cutpoints.resize(p_cont, std::set<double>());

  if (cutpoints_list.isNotNull()) {
    Rcpp::List tmp_cutpoints(cutpoints_list);
    if ((int)tmp_cutpoints.size() != p_cont) {
      Rcpp::Rcout << "p_cont = " << p_cont
                  << " cutpoints_list.size() = " << tmp_cutpoints.size() << "\n";
      Rcpp::stop("cutpoints_list needs length p_cont.");
    }

    for (int j = 0; j < p_cont; ++j) {
      if (tmp_cutpoints[j] != R_NilValue) {
        Rcpp::NumericVector v = Rcpp::as<Rcpp::NumericVector>(tmp_cutpoints[j]);
        if (v.size() <= 1) {
          Rcpp::Rcout << "Only " << v.size()
                      << " cutpoints supplied for X_cont[," << (j+1) << "]\n";
          Rcpp::stop("Not enough cutpoints supplied.");
        }
        for (int l = 0; l < v.size(); ++l) cutpoints[j].insert(v[l]);
      }
    }
  }
}

// -------------------------
// categorical levels
// -------------------------
void parse_cat_levels(std::vector<std::set<int>> &cat_levels,
                      int &p_cat,
                      Rcpp::Nullable<Rcpp::List> &cat_levels_list)
{
  cat_levels.clear();
  cat_levels.resize(p_cat, std::set<int>());

  if (cat_levels_list.isNotNull()) {
    Rcpp::List tmp(cat_levels_list);
    if ((int)tmp.size() != p_cat) {
      Rcpp::Rcout << "[parse_cat_levels]: p_cat = " << p_cat
                  << " cat_levels_list.size() = " << tmp.size() << "\n";
      Rcpp::stop("cat_levels_list must have size p_cat.");
    }
    for (int j = 0; j < p_cat; ++j) {
      if (tmp[j] == R_NilValue) {
        Rcpp::Rcout << "[parse_cat_levels]: j = " << j << " no levels provided\n";
        Rcpp::stop("cat_levels_list cannot contain null elements.");
      }
      Rcpp::IntegerVector lv = Rcpp::as<Rcpp::IntegerVector>(tmp[j]);
      for (int l = 0; l < lv.size(); ++l) cat_levels[j].insert(lv[l]);
    }
  }
}

// -------------------------
// graphs over categorical levels
// edge_mat_list[[j]] is an integer matrix with columns:
//   (source_level, sink_level) or (source, sink, weight)
// levels should be coded as integers consistent with x_cat (usually 0..K-1).
// -------------------------
//void parse_graphs(std::vector<std::vector<edge>> &edges,
//                  int &p_cat,
//                  Rcpp::Nullable<Rcpp::List> &edge_mat_list)
//{
//  edges.clear();
//  edges.resize(p_cat, std::vector<edge>());
//
//  if (edge_mat_list.isNull()) return;
//
//  Rcpp::List tmp(edge_mat_list);
//  if ((int)tmp.size() != p_cat) {
//    Rcpp::Rcout << "[parse_graphs]: p_cat = " << p_cat
//                << " edge_mat_list.size() = " << tmp.size() << "\n";
//    Rcpp::stop("edge_mat_list must have length p_cat.");
//  }
//
//  for (int j = 0; j < p_cat; ++j) {
//    if (tmp[j] == R_NilValue) continue;
//
//    Rcpp::IntegerMatrix em = Rcpp::as<Rcpp::IntegerMatrix>(tmp[j]);
//    int E = em.nrow();
//    int C = em.ncol();
//    if (C < 2) Rcpp::stop("edge matrix must have >= 2 columns (source, sink).");
//
//    edges[j].reserve(E);
//
//    for (int e = 0; e < E; ++e) {
//      int u = em(e, 0);
//      int v = em(e, 1);
//      double w = 1.0;
//      if (C >= 3) w = (double)em(e, 2);
//      edges[j].push_back(edge(u, v, w));
//    }
//  }
//}

// -------------------------
// nesting
// nest_list: list of elements; each element is a list:
//   [[1]] integer(2): (hi_index, lo_index) in *all-covariate indexing*
//   [[2]] integer vector of length K_hi mapping hi_level -> lo_level
// -------------------------
void parse_nesting(std::vector<hi_lo_map> &nesting,
                   std::vector<edge_map> &nest_graph_in,
                   std::vector<edge_map> &nest_graph_out,
                   std::vector<std::map<int, std::set<int>>> &nest_graph_components,
                   int &p_cont,
                   Rcpp::IntegerMatrix &cov_ensm,
                   std::vector<std::set<int>> &cat_levels,
                   Rcpp::Nullable<Rcpp::List> &nest_list)
{
  int R = cov_ensm.cols();
  int p_cat = (int)cat_levels.size();

  nesting.clear();
  std::vector<edge> nest_graph_edges;

  // always init ensemble-specific containers (even if no nesting)
  nest_graph_in.assign(R, edge_map());
  nest_graph_out.assign(R, edge_map());
  nest_graph_components.assign(R, std::map<int, std::set<int>>());

  if (nest_list.isNull()) return;

  Rcpp::List tmp_nest_list(nest_list);
  nesting.resize(tmp_nest_list.size(), hi_lo_map());

  for (int nix = 0; nix < tmp_nest_list.size(); ++nix) {
    Rcpp::List one = tmp_nest_list[nix];

    Rcpp::IntegerVector pair = Rcpp::as<Rcpp::IntegerVector>(one[0]);
    Rcpp::IntegerVector mp   = Rcpp::as<Rcpp::IntegerVector>(one[1]);

    int hi = pair[0] - p_cont; // categorical index
    int lo = pair[1] - p_cont;

    if (hi < 0 || hi >= p_cat || lo < 0 || lo >= p_cat)
      Rcpp::stop("nest_list hi/lo indices out of categorical range.");

    int K_hi = (int)cat_levels[hi].size();
    int K_lo = (int)cat_levels[lo].size();

    if (mp.size() != K_hi) {
      Rcpp::Rcout << "Element " << (nix+1)
                  << " of nest_list: expected mapping length "
                  << K_hi << " (hi levels) but got " << mp.size() << "\n";
      Rcpp::stop("Every hi-res value must be mapped to a lo-res value.");
    }

    nesting[nix].hi = hi;
    nesting[nix].lo = lo;

    for (int hi_val = 0; hi_val < mp.size(); ++hi_val) {
      int lo_val = mp[hi_val];
      if (lo_val < 0 || lo_val >= K_lo) {
        Rcpp::Rcout << "Invalid lo value in mapping: hi_val=" << hi_val
                    << " lo_val=" << lo_val
                    << " but lo has K_lo=" << K_lo << " levels.\n";
        Rcpp::stop("Invalid lo-res value in nesting map.");
      }
      if (nesting[nix].map.count(lo_val) == 0)
        nesting[nix].map.insert({lo_val, std::set<int>()});
      nesting[nix].map.find(lo_val)->second.insert(hi_val);
    }

    nest_graph_edges.push_back(edge(hi, lo));
  }

  // ensemble-specific induced graphs/components
  for (int r = 0; r < R; ++r) {
    std::set<int> verts;
    for (int j = 0; j < p_cat; ++j) {
      if (cov_ensm(j + p_cont, r) == 1) verts.insert(j);
    }
    if (verts.empty()) continue;

    std::vector<edge> tmp_edges = get_induced_edges(nest_graph_edges, verts);
    build_in_out_edge_map(nest_graph_in[r], nest_graph_out[r], tmp_edges, verts);

    std::vector<std::vector<int>> comps;
    find_components(comps, tmp_edges, verts);
    for (int c = 0; c < (int)comps.size(); ++c) {
      nest_graph_components[r].insert({c, std::set<int>()});
      auto it = nest_graph_components[r].find(c);
      for (int v : comps[c]) it->second.insert(v);
    }
  }
}
