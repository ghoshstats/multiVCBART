#ifndef GUARD_graph_funs_h
#define GUARD_graph_funs_h

#include "structs.h"
#include "rng.h"

// Parse optional list of edge matrices for p_cat categorical predictors.
// edge_mat_list can be NULL; edge_mat_list[[j]] can be NULL.
void parse_graphs(std::vector<std::vector<edge>> &edges,
                  int &p_cat,
                  Rcpp::Nullable<Rcpp::List> &edge_mat_list);

void build_symmetric_edge_map(edge_map &emap, std::vector<edge> &edges, std::set<int> &vertices);
void build_in_out_edge_map(edge_map &in_emap, edge_map &out_emap, std::vector<edge> &edges, std::set<int> &vertices);

std::vector<edge> get_induced_edges(std::vector<edge> &edges, std::set<int> &vertex_subset);
void dfs(int v, std::map<int, bool> &visited, std::vector<int> &comp, edge_map &emap);
void find_components(std::vector<std::vector<int> > &components, std::vector<edge> &edges, std::set<int> &vertices);
void get_unique_edges(std::vector<edge> &edges);

arma::mat get_adjacency_matrix(edge_map emap);

arma::mat floydwarshall(std::vector<edge> &edges, std::set<int> &vertices);

void wilson(std::vector<edge> &mst_edges, std::vector<edge> &edges, std::set<int> &vertices, RNG &gen);

void delete_edge(int &cut_index, std::set<int> &l_vals, std::set<int> &r_vals, std::vector<edge> &edges, std::set<int> &vertices);

std::vector<double> get_csize_weights(std::vector<edge> &edges, std::set<int> &vertices);

void fiedler_split(std::set<int> &l_vals, std::set<int> &r_vals,
                   std::vector<edge> &edges, std::set<int> &vertices,
                   bool &random, RNG &gen);

void graph_partition(std::set<int> &l_vals, std::set<int> &r_vals,
                     std::vector<edge> &orig_edges,
                     std::set<int> &avail_levels,
                     int &cut_type,
                     RNG &gen);

#endif /* graph_funs_h */
