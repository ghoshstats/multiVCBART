#ifndef GUARD_data_parsing_funs_h
#define GUARD_data_parsing_funs_h

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "graph_funs.h"

// cutpoints for continuous predictors
void parse_cutpoints(std::vector<std::set<double>> &cutpoints,
                     int p_cont,
                     Rcpp::Nullable<Rcpp::List> &cutpoints_list);

// categorical levels
void parse_cat_levels(std::vector<std::set<int>> &cat_levels,
                      int &p_cat,
                      Rcpp::Nullable<Rcpp::List> &cat_levels_list);

// NOTE: parse_graphs(...) is declared in graph_funs.h (included above).

// nesting structure between categorical variables
void parse_nesting(std::vector<hi_lo_map> &nesting,
                   std::vector<edge_map> &nest_graph_in,
                   std::vector<edge_map> &nest_graph_out,
                   std::vector<std::map<int, std::set<int>>> &nest_graph_components,
                   int &p_cont,
                   Rcpp::IntegerMatrix &cov_ensm,
                   std::vector<std::set<int>> &cat_levels,
                   Rcpp::Nullable<Rcpp::List> &nest_list);

#endif
