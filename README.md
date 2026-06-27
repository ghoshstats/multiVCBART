# Multivariate BART (mvBART) with Graphical Horseshoe Residual Precision

This repository contains a research implementation of a multivariate Bayesian Additive Regression Trees model for
nonlinear effect modification with a **joint residual dependence** model via a precision matrix $\Omega$.

---

## Model

For $i=1,\dots,n$, we observe

- responses: $\mathbf Y_i\in\mathbb R^q$
- BART covariates: $\mathbf x_i\in[0,1]^d$
- modulating covariates/treatments: $\mathbf z_i\in\mathbb R^p$

and use the working model
$\mathbf Y_i \mid \mathbf x_i,\mathbf z_i,\Theta
\sim \mathcal N_q\big(B(\mathbf x_i)^\top \mathbf z_i,\ \Omega^{-1}\big),
\ \ \Theta=(B,\Omega),$
where $B(\cdot)$ is a $p\times q$ matrix of unknown functions, with entries $B_{jr}(\cdot)$ modeled using BART ensembles.

### Prior

- **BART** on each entry $B_{jr}(\cdot)$, with **SoftBART-style** split proportions $\pi_{rj}$.
- **Group horseshoe** shrinkage for each $(j,r)$ ensemble via local scales $\lambda_{jr}$ and global $\tau_B$.
- **Graphical horseshoe** prior on off-diagonals of $\Omega$.


## Toy simulation, q = 6

```r
library(Rcpp)
library(RcppArmadillo)
sourceCpp("src/multi_ensm_fit_mv.cpp")
source("R/mv_fit.R")
library(MASS)
library(ggplot2)

set.seed(1)
n <- 400
d <- 1
p <- 2
q <- 6

X <- matrix(runif(n*d), n, d)
Z <- matrix(rnorm(n*p), n, p)

f <- function(x) 2*(x - 0.5)             
g <- function(x) sin(2*pi*x)             

a1 <- c(1.0, 1.5, 0.0, -0.7, 1.2, 0.0)    
a2 <- c(0.0, 0.0, 0.0,  0.8, 0.0, -0.6)   

eta_true <- matrix(0, n, q)
for (r in 1:q) {
  eta_true[, r] <- Z[,1] * (a1[r] * f(X[,1])) + Z[,2] * (a2[r] * g(X[,1]))
}

Omega_true <- diag(1.5, q)
for (r in 1:(q-1)) {
  Omega_true[r, r+1] <- Omega_true[r+1, r] <- 0.4
}
Omega_true[1,3] <- Omega_true[3,1] <- 0.2
Sigma_true <- solve(Omega_true)
Y <- eta_true + MASS::mvrnorm(n, mu = rep(0,q), Sigma = Sigma_true)


xg <- seq(0, 1, length.out = 200)
n_g <- length(xg)
Xg <- matrix(xg, ncol = 1)

X_cont_test <- rbind(Xg, Xg)
Z_test <- rbind(
  cbind(rep(1, n_g), rep(0, n_g)), # Isolates the 1st ensemble
  cbind(rep(0, n_g), rep(1, n_g))  # Isolates the 2nd ensemble
)
Y_test <- matrix(0, nrow = 2 * n_g, ncol = q) 
cov_ensm <- matrix(1, nrow = 1, ncol = 2)

draws <- mv_flexbart(
  Y_train = Y,
  cov_ensm = cov_ensm,
  Z_train = Z,
  X_cont_train = X,
  Y_test = Y_test,
  Z_test = Z_test,
  X_cont_test = X_cont_test,
  M_vec = c(30, 30),           
  alpha_vec = c(0.95, 0.95),     
  beta_vec = c(2.0, 2.0),      
  sigma_B_n = 1.0,             
  sparse = FALSE,               
  a_u = 0.5,                   
  b_u = 1.0,                   
  nd = 1000, burn = 1500, thin = 1,
  save_samples = TRUE,
  verbose = TRUE
)

Bgrid <- array(0, dim = c(n_g, p, q, 1000)) # [grid, j, r, iter]

for (s in 1:1000) {
  for (r in 1:q) {
    Bgrid[, 1, r, s] <- draws$fit_test[s, 1:n_g, r]
    Bgrid[, 2, r, s] <- draws$fit_test[s, (n_g + 1):(2 * n_g), r]
  }
}

summ_curve <- function(M) {
  data.frame(
    mean = rowMeans(M),
    lo   = apply(M, 1, quantile, 0.025),
    hi   = apply(M, 1, quantile, 0.975)
  )
}

truth_B <- function(j, r, x) {
  if (j == 1) return(a1[r] * f(x))
  if (j == 2) return(a2[r] * g(x))
  stop("bad j")
}

make_df <- function(j, r) {
  S <- summ_curve(Bgrid[, j, r, ])
  data.frame(
    x = xg,
    mean = S$mean,
    lo = S$lo,
    hi = S$hi,
    truth = truth_B(j, r, xg),
    name = sprintf("B[%d,%d](x)", j, r)
  )
}

df <- do.call(rbind, lapply(1:p, function(j) do.call(rbind, lapply(1:q, function(r) make_df(j,r)))))

ggplot(df, aes(x = x)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.25) +
  geom_line(aes(y = mean), linewidth = 1) +
  geom_line(aes(y = truth), linetype = "dashed", linewidth = 1) +
  facet_wrap(~ name, scales = "free_y", ncol = q) +
  labs(x = "x", y = "value",
       title = "Function recovery: posterior mean vs truth (dashed)") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(hjust = 0.5))


# --- Omega Precision Recovery ---
Om_mean <- apply(draws$Omega, c(2,3), mean)

to_long <- function(M, label) {
  expand.grid(i = 1:q, j = 1:q, KEEP.OUT.ATTRS = FALSE) |>
    transform(val = as.vector(M), which = label)
}

dfO <- rbind(to_long(Omega_true, "true"), to_long(Om_mean, "post-mean"))

ggplot(dfO, aes(x = j, y = i, fill = val)) +
  geom_tile() +
  scale_y_reverse() +
  facet_wrap(~ which) +
  labs(title = "Omega: True vs posterior mean", x = "col", y = "row") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(hjust = 0.5))


```

## Outputs

Function recovery:

![Function recovery](figures/B_recovery_q6.png)

Omega recovery:

![Omega recovery](figures/Omega_q6.png)


