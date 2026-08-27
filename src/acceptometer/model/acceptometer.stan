// Acceptometer joint measurement model.
//
// Latent item acceptability theta links two measurement arms:
//   human arm:      ordinal ratings, participant random effects (the criterion)
//   instrument arm: LLM scores (continuous and binary elicitation cells),
//                   each cell with its own bias, slope, nuisance loadings, noise
//
// Scale convention: theta is in human-logit units. The human-arm discrimination
// is fixed at 1, so the ordered-logistic error's fixed variance anchors the
// latent scale and the within-family item sd (tau_item) is a free parameter;
// construction-family means are sum-to-zero, so the cutpoints absorb overall
// location. Instrument slopes beta are "standardized score units per human
// logit". (An earlier version fixed tau_item at 1; the pilot PPC showed that
// compresses real between-item spread and inflates within-item noise.)
data {
  int<lower=0, upper=1> prior_only;

  // items
  int<lower=1> N_item;
  int<lower=1> N_constr;
  array[N_item] int<lower=1, upper=N_constr> constr;
  int<lower=0> P;                    // nuisance covariates (log length, unigram logfreq), centered+scaled
  matrix[N_item, P] X;

  // human arm (long format; empty arrays allowed)
  int<lower=0> N_h;
  int<lower=2> K;                    // rating scale points
  array[N_h] int<lower=1, upper=N_item> item_h;
  int<lower=0> N_part;
  array[N_h] int<lower=1, upper=max(N_part, 1)> part_h;
  array[N_h] int<lower=1, upper=K> y;

  // continuous instrument cells (scores standardized per cell in Python)
  int<lower=0> N_c;
  int<lower=0> M_c;
  array[N_c] int<lower=1, upper=N_item> item_c;
  array[N_c] int<lower=1, upper=max(M_c, 1)> cell_c;
  vector[N_c] s;

  // binary instrument cells
  int<lower=0> N_b;
  int<lower=0> M_b;
  array[N_b] int<lower=1, upper=N_item> item_b;
  array[N_b] int<lower=1, upper=max(M_b, 1)> cell_b;
  array[N_b] int<lower=0, upper=1> z;
}
parameters {
  // latent acceptability
  sum_to_zero_vector[N_constr] mu_c_raw;
  real<lower=0> tau_constr;
  real<lower=0> tau_item;
  vector[N_item] z_item;

  // human arm
  ordered[K - 1] kappa;
  vector[N_part] u_raw;
  real<lower=0> sigma_u;

  // continuous cells. Family-level deviations of intercept and slope let the
  // linking function vary by construction family: LLM scoring error clusters
  // by phenomenon, and a new family's deviations are drawn from their priors,
  // which is what makes out-of-family intervals honestly wider.
  vector[M_c] alpha;
  vector[M_c] beta;
  matrix[M_c, N_constr] a_dev_raw;
  matrix[M_c, N_constr] b_dev_raw;
  vector<lower=0>[M_c] tau_a;
  vector<lower=0>[M_c] tau_b;
  matrix[M_c, P] gamma;
  // 0.05 floor (standardized-score scale): a contamination-shaped or
  // degenerate cell cannot drive sigma to zero and pass itself off as a
  // noiseless oracle for unrated items.
  vector<lower=0.05>[M_c] sigma_s;

  // binary cells
  vector[M_b] a_b;
  vector[M_b] b_b;
  matrix[M_b, P] g_b;
}
transformed parameters {
  vector[N_constr] mu_c = tau_constr * mu_c_raw;
  vector[N_item] theta = mu_c[constr] + tau_item * z_item;
  vector[N_part] u = sigma_u * u_raw;
  matrix[M_c, N_constr] a_dev = diag_pre_multiply(tau_a, a_dev_raw);
  matrix[M_c, N_constr] b_dev = diag_pre_multiply(tau_b, b_dev_raw);
}
model {
  // priors (weakly informative throughout; no flat priors)
  mu_c_raw ~ normal(0, 1);
  tau_constr ~ normal(0, 1);
  tau_item ~ normal(0, 1.5);
  z_item ~ std_normal();
  kappa ~ normal(0, 2);
  u_raw ~ std_normal();
  sigma_u ~ normal(0, 1);

  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  to_vector(a_dev_raw) ~ std_normal();
  to_vector(b_dev_raw) ~ std_normal();
  tau_a ~ normal(0, 0.5);
  tau_b ~ normal(0, 0.5);
  to_vector(gamma) ~ normal(0, 0.5);
  sigma_s ~ normal(0, 1);

  a_b ~ normal(0, 1.5);
  b_b ~ normal(0, 1);
  to_vector(g_b) ~ normal(0, 0.5);

  if (!prior_only) {
    if (N_h > 0)
      y ~ ordered_logistic(theta[item_h] + u[part_h], kappa);
    if (N_c > 0) {
      vector[N_c] nu;
      for (n in 1:N_c) {
        int m = cell_c[n];
        int i = item_c[n];
        int c = constr[i];
        nu[n] = alpha[m] + a_dev[m, c] + (beta[m] + b_dev[m, c]) * theta[i]
                + dot_product(gamma[m], X[i]);
      }
      s ~ normal(nu, sigma_s[cell_c]);
    }
    if (N_b > 0) {
      vector[N_b] eta;
      for (n in 1:N_b)
        eta[n] = a_b[cell_b[n]] + b_b[cell_b[n]] * theta[item_b[n]]
                 + dot_product(g_b[cell_b[n]], X[item_b[n]]);
      z ~ bernoulli_logit(eta);
    }
  }
}
generated quantities {
  // CONDITIONAL reliability per continuous cell: theta-signal variance over
  // (theta-signal + residual noise) at the realized theta spread, holding the
  // nuisance covariates fixed. Systematic nuisance loading (gamma) and
  // family-level linking deviations are deliberately excluded: they are bias
  // structure, not signal, and the certificate labels this quantity as
  // conditional for exactly that reason.
  real v_theta = variance(theta);
  vector[M_c] reliability;
  for (m in 1:M_c)
    reliability[m] = square(beta[m]) * v_theta
                     / (square(beta[m]) * v_theta + square(sigma_s[m]));
}
