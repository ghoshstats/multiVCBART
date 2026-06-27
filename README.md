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

## Usage and Installation

Currently, this repository is used by sourcing the R and C++ files directly into your environment. You will need `Rcpp`, `RcppArmadillo`, and `MASS` installed in your R environment.

1. Clone this repository to your local machine.
2. Then open **R** in the repository's root directory
```r
sourceCpp("src/multi_ensm_fit_mv.cpp")
source("R/mv_fit.R")
```
3. Core dependencies: `Rcpp` and `RcppArmadillo`

## Toy simulation

Here's a small experiment to show how to fit the multiVCBART model.

```r
library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(dbarts)
sourceCpp("src/multi_ensm_fit_mv.cpp")
source("R/mv_fit.R")

generate_vcbart_sur_high_dim <- function(n, d = 50, p = 100, q = 2) {
  # Generate Modifiers (X) and Primary Covariates (Z)
  X <- matrix(runif(n * d, 0, 1), nrow = n, ncol = d)
  Z <- matrix(rnorm(n * p, 0, 1), nrow = n, ncol = p)
  
  beta1 <- matrix(0, nrow = n, ncol = p)
  beta1[, 1] <- 10 * sin(pi * X[,1] * X[,2])
  beta1[, 2] <- 20 * (X[,3] - 0.5)^2
  beta1[, 3] <- 10 * X[,4]
  beta1[, 4] <- 5 * X[,5]
  beta1[, 5] <- 5 
  
  beta2 <- matrix(0, nrow = n, ncol = p)
  beta2[, 6] <- 10 * cos(pi * X[,3] * X[,4])
  beta2[, 7] <- 20 * (X[,5] - 0.5)^2
  beta2[, 8] <- 10 * X[,6]
  beta2[, 9] <- 5 * X[,7]
  beta2[, 10] <- 5 
  
  eta1 <- rowSums(Z * beta1)
  eta2 <- rowSums(Z * beta2)
  Eta <- cbind(eta1, eta2)
  
  Omega <- matrix(c(1.0, 0.6, 
                    0.6, 1.0), nrow = 2, ncol = 2)
  Sigma <- solve(Omega) * 4.0 
  Y <- Eta + MASS::mvrnorm(n, mu = rep(0, q), Sigma = Sigma)
  
  return(list(z = Z, x = X, y = Y, y_true = Eta, Sigma = Sigma))
}

set.seed(2026)
n_train <- 300
n_test  <- 100

train_data <- generate_vcbart_sur_high_dim(n = n_train)
test_data  <- generate_vcbart_sur_high_dim(n = n_test)

# Create ensemble mapping matrix (1 indicates inclusion)
cov_ensm_mat <- matrix(1, nrow = ncol(train_data$x), ncol = ncol(train_data$z))

cat("Fitting multiVCBART...\n")
fit_mb <- mv_flexbart(
  Y_train = train_data$y,
  cov_ensm = cov_ensm_mat,
  Z_train = train_data$z,
  X_cont_train = train_data$x,
  Y_test = test_data$y,
  Z_test = test_data$z,
  X_cont_test = test_data$x,
  M_vec = rep(20, ncol(train_data$z)),
  alpha_vec = rep(0.95, ncol(train_data$z)),
  beta_vec = rep(2.0, ncol(train_data$z)),
  sigma_B_n = 1.0, 
  sparse = TRUE, 
  a_u = 0.5, 
  omega_method = "stable_hs",
  omega_b_tau = 2.0,
  nd = 1000, 
  burn = 300, 
  thin = 1,
  save_samples = TRUE, 
  verbose = TRUE
)

rmse_out_1 <- sqrt(mean((fit_mb$fit_test_mean[, 1] - test_data$y_true[, 1])^2))  ## 2.116
rmse_out_2 <- sqrt(mean((fit_mb$fit_test_mean[, 2] - test_data$y_true[, 2])^2))  ## 2.185

################# Comparing against fitting separate univariate BARTs ####################

x_train_uni <- cbind(train_data$z, train_data$x)
x_test_uni  <- cbind(test_data$z, test_data$x)

cat("Outcome 1...\n")
fit_bart_1 <- dbarts::bart(
  x.train = x_train_uni, y.train = train_data$y[, 1],
  x.test  = x_test_uni,
  ntree = 20, ndpost = 1000, nskip = 300, keepevery = 1, verbose = FALSE
)

cat("Outcome 2...\n")
fit_bart_2 <- dbarts::bart(
  x.train = x_train_uni, y.train = train_data$y[, 2],
  x.test  = x_test_uni,
  ntree = 20, ndpost = 1000, nskip = 300, keepevery = 1, verbose = FALSE
)

rmse_uni_1 <- sqrt(mean((fit_bart_1$yhat.test.mean - test_data$y_true[, 1])^2)) ## 5.39
rmse_uni_2 <- sqrt(mean((fit_bart_2$yhat.test.mean - test_data$y_true[, 2])^2)) ## 7.49

```
