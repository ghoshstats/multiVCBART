# multiVCBART: Multivariate BART with Graphical Horseshoe Residual Precision

This repository contains a research implementation of a multivariate Bayesian Additive Regression Trees (BART) model for nonlinear effect modification with a **joint residual dependence** model via a precision matrix $\Omega$.

---

## Model

For $i=1,\dots,n$, we observe:

- responses: $\mathbf Y_i\in\mathbb R^q$
- BART covariates: $\mathbf x_i\in[0,1]^d$
- modulating covariates/treatments: $\mathbf z_i\in\mathbb R^p$

We use the working model:
$\mathbf Y_i \mid \mathbf x_i,\mathbf z_i,\Theta \sim \mathcal N_q\big(B(\mathbf x_i)^\top \mathbf z_i,\ \Omega^{-1}\big), \ \  \Theta=(B,\Omega)$
where $B(\cdot)$ is a $p\times q$ matrix of unknown functions, with entries $B_{jr}(\cdot)$ modeled using BART ensembles.

### Prior Structure

- **BART** on each entry $B_{jr}(\cdot)$, with **SoftBART-style** split proportions $\pi_{rj}$.
- **Group horseshoe** shrinkage for each $(j,r)$ ensemble via local scales $\lambda_{jr}$ and global $\tau_B$.
- **Graphical horseshoe** prior on off-diagonals of $\Omega$.

---

## Usage and "Installation"

Currently, this repository is used by sourcing the R and C++ files directly into your environment. You will need `Rcpp`, `RcppArmadillo`, and `MASS` installed in your R environment.

1. Clone this repository to your local machine.
2. Then open **R** in the repository's root directory
```r
sourceCpp("src/multi_ensm_fit_mv.cpp")
source("R/mv_fit.R")
```
3. Core dependencies: `Rcpp` and `RcppArmadillo`
