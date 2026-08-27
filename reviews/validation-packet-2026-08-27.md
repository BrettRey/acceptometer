# Acceptometer external validation packet (2026-08-27, rev 3: final pilot model)

## Review brief

You are adversarially validating a Bayesian measurement tool that treats LLMs
as biased instruments for human acceptability judgments. Everything you need is
in this file. Assume nothing is correct until you have checked it. Do not
review prose style. Report findings as a numbered list, each with severity
(major/moderate/minor), the exact location, and, where possible, the concrete
failure it would cause.

Revision history you are auditing (each change was forced by a gate failing on
real data - 120 Sprouse LI items, 1,519 participant-level 7-point ratings):
(1) tau_item freed after a PPC failure showed the fixed within-family sd
compressed real item spread (real-data posterior ~2.0); zero-avoiding
gamma(2,1) prior. (2) One instrument per model+method: correlated paraphrase
cells had entered as independent witnesses and outvoted the human criterion.
(3) A locally-stable reflection mode captured 1-in-4 randomly-initialized
chains; latents now initialize at standardized human item means (basin
selection, no sign constraints). (4) Instrument-by-item error omega for
replicated cells (switched off for single-observation cells where it is a
redundant ridge): repeats cannot average away an instrument's stable per-item
opinion; qwen reliability fell .90 -> .54 and the PPC then passed.

Priority questions, in order:

1. SPEC-VS-CODE DRIFT: does `acceptometer.stan` implement exactly what
   DESIGN.md describes? Any silent difference is a major finding.
2. IDENTIFICATION: with tau_item, tau_constr, sigma_u, kappa, omega, sigma_s
   all free and discrimination fixed at 1, what is identified by data vs
   prior? Is the omega/sigma_s split for replicated cells identified only by
   the replicate structure, and is the has_reps gating sound? Is anything
   still assuming the old unit within-family sd?
3. PRIOR PUSHFORWARDS IN SBC: (a) projected iid normals = sum_to_zero_vector
   prior; (b) sorted iid normals = ordered-cutpoint prior; (c) rejection
   sampling = truncated sigma_s; (d) gamma draw matches gamma(2,1); (e)
   Wilson-Hilferty tail approximation. If any is wrong the SBC pass is
   meaningless.
4. INITIALIZATION: do data-informed inits (z_item at standardized human item
   means) merely select a basin, or can they bias any reported quantity given
   finite chains? Is the reflection-mode analysis right?
5. LOCO PREDICTIVE: is `_predict_observed_means` a correct posterior
   predictive for the observed sample mean of n_j new participants? Any
   leakage of held-out human data into the training fit? Note the training
   fit retains the held-out family's INSTRUMENT scores by design - is the
   design claim ("that is the transfer mechanism") sound, or does anything
   else leak?
6. WARRANT LOGIC: can any path grant on missing/failed evidence? Are the
   thresholds and the contamination cap defensible? The pilot certificate
   granted screening (.54 > .5) and ranking (.68 > .6) and refused aggregate
   estimation on contamination despite coverage .925 - audit that outcome.
7. PPC DESIGN: are the three discrepancies right, and what misfit slips past
   all three? (They already caught two real failures; what would they miss?)
8. LOGPROB ALIGNMENT: verify hf_logprob.py scores P(token_t | tokens_<t)
   exactly, BOS handling included.
9. EVIDENCE CONSISTENCY: do the attached reports support every "passed" and
   every number quoted in the revision history above?
10. WHAT THE FIRST REVIEWER MISSED: the attached glm-5.3-flash review's
    adopted findings are listed in the triage; several of its predictions
    (PPC failure, repeat-noise incoherence) came true. Find what it missed.
    Documented v1 limitations are fair game only if worse than documented.



---

## FILE: DESIGN.md

```
# Acceptometer: a warranted measurement instrument for acceptability judgments
<!-- SUMMARY: Design spec for the Gelman-style LLM-as-instrument tool · status: building · updated: 2026-08-27 -->

## What this is

A tool that treats an LLM not as a judge but as a **biased measurement instrument**
for human acceptability judgments. It answers two questions, and keeps them separate:

1. **The posterior:** for these items, this target population, how much information
   does this instrument carry about the human response distribution, with what
   uncertainty? (Gelman's question.)
2. **The warrant:** what evidence licenses generalizing beyond the fitted domain,
   to new constructions, new items, new model versions? (The projectibility
   question. A posterior conditions on the model and the sampled domain; it cannot
   license projection by itself.)

Every fit therefore emits two artifacts: a posterior (ArviZ/CSV) and a
**validity certificate** (`warrant.yaml`) stating the domain validated, the
invariance evidence, and the claim tier licensed. Claims outside the certificate
are the user's own risk, and the tool says so.

Background: `llm-grammaticality-judgments-survey.md` (state of the art through
2026-08). The survey's bottom line, "validate an instrument for a specified
estimand, not 'a judge' simpliciter," is the requirements document. This tool is
the estimand-first implementation.

## The statistical model (the heart)

One joint Bayesian measurement-error model, fitted in Stan (CmdStan 2.36 via
cmdstanpy). Latent item acceptability links two kinds of noisy measurement.

**Latent structure.** Item `i` belongs to construction family `c(i)`:

    theta_i ~ normal(mu_c[c(i)], tau_item)      # item-level latent acceptability
    mu_c    ~ normal(0, tau_constr)             # partial pooling over families

Scale identified by standardizing theta (soft sum-to-zero, unit-scale prior);
cutpoints and instrument slopes are then interpretable.

**Human arm** (the criterion). Participant `p` gives ordinal rating `y_ip` on a
K-point scale:

    y_ip ~ ordered_logistic(theta_i + u_p, kappa)   # u_p ~ normal(0, sigma_p)

Participants are random effects; `kappa` cutpoints shared. This models the full
response distribution, not the item mean, so posterior predictive checks can
target variance and disagreement structure, exactly where binary labels suppress
legitimate heterogeneity (the Dentella lesson).

**Instrument arm** (the LLM scores). A fitted instrument cell `m` is one
(model, method) pair: prompt paraphrases and sampling repeats are repeated
measurements of that one instrument, never additional witnesses (three
paraphrases of one model share errors, and three transforms of one forward
pass certainly do; treating them as independent lets one model outvote the
human criterion). The grid still records every paraphrase separately for the
descriptive fan and prompt-invariance statistics. The linking function varies
by construction family `c` (LLM scoring error clusters by phenomenon), and
replicated cells carry an instrument-by-item error `e_mi` (repeats average
away draw noise, never the instrument's stable opinion about an item; for
single-observation cells the term is redundant with sigma_m and is switched
off):

    continuous: s_imr ~ normal(alpha_m + a_dev_mc + (beta_m + b_dev_mc) * theta_i
                               + gamma_m' * x_i + e_mi, sigma_m)
                a_dev_mc ~ normal(0, tau_a_m),  b_dev_mc ~ normal(0, tau_b_m),
                e_mi ~ normal(0, omega_m)
    binary:     s_im ~ bernoulli_logit(a_m + b_m * theta_i + g_m' * x_i)
                (v1: descriptive only in fits with near-deterministic repeats;
                no overdispersion term yet)

`x_i` = nuisance covariates: log token length, mean unigram log frequency
(wordfreq). `beta_m` is the population information coefficient; `alpha_m`,
`gamma_m`, and the family deviations are the instrument's bias structure;
`sigma_m` its noise, floored at 0.05 on the standardized scale so no cell can
present itself as a noiseless oracle. For an item in a NEW family, the
deviations are drawn from their priors, which is what makes out-of-family
intervals wider than in-family ones by construction rather than by hope.

**Conditional reliability** per continuous cell =
`beta_m^2 * var(theta) / (beta_m^2 * var(theta) + omega_m^2 + sigma_m^2)`
(omega only for replicated cells). This is
conditional on the nuisance covariates and excludes the systematic bias terms
(gamma, family deviations) deliberately: they are bias structure, not signal.
The certificate labels it as conditional; it is also fit- and item-set-specific,
and the certificate records the item pool it was estimated on. MORCELA is the
special case of one continuous method, no family structure, fixed covariates;
here the linking is Bayesian, multilevel, and jointly estimated with the human
arm, so linking uncertainty propagates into everything computed from the joint
posterior (certificate summaries derived outside it, like LOCO RMSEs, are
labeled as point summaries).

**The instrument use.** Items with LLM scores but no human data get a posterior
for `theta_i` through the fitted linking function, with honestly widened
uncertainty. That posterior, pushed through the human arm, predicts the human
response distribution for unrated items. That's the product.

**Priors.** Weakly informative, documented in the Stan file: normal(0,1) on
standardized coefficients, half-normal on scales, induced-dirichlet-free simple
ordered cutpoints with normal(0,2) prior. No flat priors anywhere.

## Validation ladder (in order, no skipping)

1. **Fake-data recovery.** Simulate the full generative process (known theta,
   participants, instruments); fit; verify parameter recovery, interval coverage,
   and that reliability estimates match the simulation truth. The pipeline is
   untrusted until it passes. `acceptometer simulate --check`.
2. **SBC-lite.** Rank-histogram check on key parameters (beta_m, sigma_m,
   tau_constr) over ~50 simulated fits. Cheap version of simulation-based
   calibration; full SBC documented as future work.
3. **Real fit.** Human criterion data + real elicitation runs.
4. **Posterior predictive checks at the real-fit stage, with consequences.**
   Simulate human samples from the fitted model; compare per-family item-mean
   spread, category usage, and disagreement structure to the real data. A PPC
   failure on disagreement structure refuses the distributional tier (already
   refused in v1) and flags the aggregate tier in the certificate; it is a
   recorded gate, not a decoration.
5. **Generalization tests** (these feed the warrant):
   - **LOCO-CV:** leave-one-construction-family-out. Refit without family c's
     human data; predict its human item means from LLM scores alone; record
     transfer RMSE, interval coverage (against the observed sample mean,
     including its own ordinal sampling noise), and rank correlation. This is
     *a* projectibility test along ONE axis: transfer across construction
     families, within one item source, language, register, protocol, and time.
     Families are a purposive sample, not draws from a superpopulation, so
     LOCO results are descriptive over the families tested; the certificate
     lists the axes NOT tested (population, register, language, item source,
     time) as untested, every time.
   - **Multiverse spread (descriptive):** per item, the spread of standardized
     scores across elicitation cells. Tight fan = the cells agree; wide fan =
     the elicitation is the result and no single-cell number should be
     reported. A tight fan is NOT dispositive: local instruments share
     pretraining data and can share construction-specific error, so the
     certificate carries shared-mode bias as a permanent residual risk.
   - **Prompt invariance:** paraphrase cells treated as replicate instruments;
     their beta_m/alpha_m spread quantifies how much the prompt is part of the
     instrument.
6. **Drift monitoring.** A fixed sentinel item set re-scored per model version /
   date; version enters as an instrument-level covariate; drift bounds go in the
   certificate, and claims for a new epoch are restricted to the
   sentinel-covered domain. "GPT-x judged" becomes "instrument G, calibration
   epoch t."

## The warrant certificate (`warrant.yaml`)

Machine-readable and human-auditable. Fields:

- `estimand`: population, response scale, context (e.g. "mean 7-point rating,
  US-English adult non-linguists, decontextualized sentences").
- `domain`: construction families, item source, language, register actually
  validated.
- `instruments`: model + version + access date + elicitation cells, each with
  posterior reliability and bias summary.
- `evidence`: LOCO transfer error and coverage per held-out family; multiverse
  spread statistic; prompt-invariance spread; human split-half reliability
  benchmark; drift epoch and sentinel bounds; contamination assessment (public
  benchmark items? published before model cutoff?).
- `licensed_claims`: which tiers of the survey's intended-use hierarchy the
  evidence supports (screening / ranking / aggregate estimation / effect-size
  reproduction / distributional claims), each explicitly granted or refused
  with the reason and the pre-registered numeric threshold it met or missed
  (defaults: screening needs diagnostics passed + any cell conditional
  reliability > 0.5; ranking additionally needs mean LOCO Spearman > 0.6;
  aggregate estimation additionally needs LOCO 90% coverage in [0.75, 0.98]).
  **Contamination caps tiers:** if the item source is public-benchmark-suspect
  (or unassessed), aggregate estimation and above are refused regardless of
  the numbers, because contamination inflates exactly the statistics those
  tiers rest on, and LOCO rewards it (the held-out family is published too).
- `refused_claims`: stated, not implied. E.g. "no evidence for individual-level
  human simulation; no mechanism claims."
- `residual_risks`: permanent caveats no test in the ladder can discharge,
  at minimum shared pretraining bias across local instruments (a tight
  multiverse fan does not rule out shared construction-specific error).
- `estimand.population` defaults to the criterion sample's own population;
  claiming a different target population requires a population-transfer test
  the v1 ladder does not contain, so v1 certificates refuse it explicitly.

The certificate is versioned with the fit. Generalizing outside it is possible
but the tool never silently pretends the posterior covers it.

## Elicitation multiverse grid

Methods (per model, where obtainable):

| Cell | Requires | Notes |
|---|---|---|
| `logprob_sum` | logprobs | sum of token logprobs (HF local models) |
| `logprob_mean` | logprobs | length-normalized |
| `slor` | logprobs + unigram freq | Lau et al. 2017 baseline |
| `mp_delta` | logprobs + paired items | minimal-pair delta, the theoretically warranted contrast (Hu et al. 2026) |
| `prompt_binary` | chat | "acceptable yes/no", 3 registered paraphrases |
| `prompt_scalar` | chat | 1-7 rating, 3 registered paraphrases |
| `answer_token_prob` | chat + logprobs | P(yes-token), local models only |

Prompted cells run n=5 repeats, ALL at one temperature (the provider default),
so the likelihood sees a single sampling regime; a temp-0 draw mixed with
stochastic repeats would attach a noise estimate from one regime to a score
from another. Repeats are repeated measurements of one instrument, never
synthetic participants: continuous repeats inform sigma_m, binary repeats
inform the response probability through the Bernoulli likelihood.

**Instruments (local-first, free):** Pythia-160m/410m via transformers (exact
logprobs, open checkpoints, the scaling-study workhorses); Ollama chat models
(qwen3:8b, gemma3:12b, mistral-small:24b, glm-4.7-flash) as prompted judges.
API adapters (Claude, GPT) are stubs with the same interface, to be enabled
deliberately, never by default (cost + version-drift discipline).

## Data

- **Human criterion:** participant-level ordinal ratings. Target: the Sprouse,
  Schütze & Almeida Linguistic Inquiry judgment data (public) and/or the MORCELA
  release if it includes per-participant ratings. Acquired with a provenance
  manifest (URL, access date, SHA-256, license). NO fabricated data anywhere:
  if acquisition fails, the real-fit stage waits and says so; simulation covers
  development.
- **Minimal pairs:** BLiMP sample (public, GitHub) for the `mp_delta` cell,
  with the contamination caveat recorded in the certificate.
- Data files live in `data/` with `data/MANIFEST.yaml`; raw downloads are
  gitignored, the manifest is not.

## Package layout

    src/acceptometer/
      items.py        # Item, MinimalPair, Construction; JSONL schema
      elicit/
        base.py       # Instrument protocol: score(items) -> Measurements
        hf_logprob.py # transformers scorer (logprob cells)
        ollama_chat.py# prompted cells against local Ollama
        grid.py       # multiverse grid runner; caching; run manifest
      model/
        acceptometer.stan   # the joint measurement model
        fit.py        # cmdstanpy wrapper; standardization; diagnostics gate
        simulate.py   # fake-data generator mirroring the Stan model
        sbc.py        # SBC-lite rank checks
        loco.py       # leave-one-construction-out harness
      warrant.py      # certificate builder from fit + validation artifacts
      design.py       # human-budget allocator: rank items by posterior sd(theta), decision relevance
      drift.py        # sentinel set management, epoch comparison
      plots.py        # secret-weapon by-construction plot; calibration; multiverse fan; item scatter with intervals
      cli.py          # acceptometer simulate|elicit|fit|validate|warrant|plot

Environment: uv-managed venv, Python 3.12 (3.14 breaks torch). Dependencies:
cmdstanpy, arviz, pandas, numpy, matplotlib, torch (CPU/MPS), transformers,
wordfreq, pyyaml, click, httpx.

## Division of labor (this build)

- Fable (parent): this spec, the Stan model, simulate.py, fit.py, integration,
  final verification. The statistical core is not delegated.
- codex: elicit/ adapters + grid runner + items.py against the interface spec
  (checkable: golden tests with tiny fixtures).
- claude subagent: plots.py + cli.py + warrant.py skeleton (checkable: runs on
  simulated fit).
- general agent: data acquisition + provenance manifest (checkable: hashes,
  URLs, licenses on record).
- ocx (free tier): adversarial design review of this document; findings triaged,
  not auto-applied.
- Verification: everything passes the fake-data ladder before any real-data
  claim is made. LLM output is a candidate, never a settled result.

## Known v1 misspecifications and limits (documented, not denied)

- **Prompted scalar (1-7) cells enter the continuous likelihood as normal.**
  Bounded ordinal data as normal is wrong at the edges; v2 gives instrument
  scalar cells their own ordered-logistic arm. Standardization makes it
  tolerable for ranking-tier use; the certificate should not rest an
  aggregate-estimation grant on scalar cells alone.
- **Binary instrument cells have no family deviations** (continuous cells do);
  their linking is family-invariant in v1.
- **Fake-data recovery and SBC validate the computation, not the model.**
  Passing them is necessary, never sufficient: real data has features the
  simulation lacks (ceiling effects, bounded scales, contamination,
  non-exchangeable raters). The PPC gate exists because of this.
- **No prior-sensitivity stage in the ladder yet.** The latent scale is
  identified partly by priors; a sensitivity sweep (halve/double the key prior
  scales, compare certificate numbers) is v2 work and the certificate says so.
- **Nonresponse:** prompted cells produce refusals/parse failures, logged per
  cell by the grid. The warrant reports the failure rate; cells above 10% are
  flagged and cannot support tier grants. Missingness correlated with theta is
  a real risk the v1 model does not correct.
- **SBC tracks internals** (slopes, scales), not the product quantities
  (unrated-item theta coverage); the recovery check covers theta in-sample.
  Held-out-item SBC is v2.

## Non-goals (v1)

- No API-model elicitation runs (adapters stubbed, off by default).
- No MRP/persona poststratification yet: requires validated persona
  manipulations and stratified human anchors; the model slot for it exists
  (instrument covariates), the claim tier is refused in v1 certificates.
- No claim about mechanism, ever. The certificate's refused_claims section is
  permanent on that point.

```


---

## FILE: src/acceptometer/model/acceptometer.stan

```
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
  // 1 when the cell has replicated observations of at least one item: the
  // instrument-by-item error is identified only against replicate noise, and
  // for single-observation cells it is redundant with sigma_s (a ridge), so
  // there it is switched off and sigma_s carries the total error.
  array[max(M_c, 1)] int<lower=0, upper=1> has_reps;

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
  // instrument-by-item systematic error: repeats of a cell can average away
  // draw noise (sigma_s) but never the instrument's own stable opinion about
  // an item; without this term, replicated cells claim fictitious precision
  // and drag theta away from the human criterion.
  matrix[M_c, N_item] e_raw;
  vector<lower=0>[M_c] omega;
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
  // zero-avoiding: tau_item -> 0 means items within a family are identical in
  // acceptability, a priori false for designed judgment studies, and it is
  // exactly the region where a spurious reflection mode (flipped instrument
  // slopes) escapes the human arm's sign anchoring. gamma(2,1): mode 1,
  // density vanishing at 0.
  tau_item ~ gamma(2, 1);
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
  to_vector(e_raw) ~ std_normal();
  omega ~ normal(0, 0.5);
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
                + dot_product(gamma[m], X[i])
                + (has_reps[m] == 1 ? omega[m] * e_raw[m, i] : 0);
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
                     / (square(beta[m]) * v_theta
                        + (has_reps[m] == 1 ? square(omega[m]) : 0)
                        + square(sigma_s[m]));
}

```


---

## FILE: src/acceptometer/model/simulate.py

```
"""Fake-data generator mirroring acceptometer.stan exactly.

Gelman's rule: the pipeline is untrusted until it recovers known truth.
This module simulates the full generative process (latent thetas, human
ordinal ratings with participant effects, continuous and binary instrument
cells with known bias/slope/noise), then `recovery_check` fits the Stan model
to the simulation and gates on parameter recovery and interval coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SimTruth:
    theta: np.ndarray
    mu_c: np.ndarray
    tau_constr: float
    kappa: np.ndarray
    sigma_u: float
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    sigma_s: np.ndarray
    omega: np.ndarray
    a_b: np.ndarray
    b_b: np.ndarray
    g_b: np.ndarray
    to_dict: dict = field(default_factory=dict)


def _ordered_logistic_rng(eta: np.ndarray, kappa: np.ndarray, rng) -> np.ndarray:
    """Draw ordinal outcomes 1..K given linear predictor eta and cutpoints."""
    # P(y <= k) = logistic(kappa_k - eta)
    cum = 1.0 / (1.0 + np.exp(-(kappa[None, :] - eta[:, None])))  # (N, K-1)
    u = rng.uniform(size=eta.shape[0])
    return 1 + (u[:, None] > cum).sum(axis=1)  # in 1..K


def simulate(
    n_constr: int = 8,
    items_per_constr: int = 12,
    n_part: int = 40,
    ratings_per_item: int = 12,
    K: int = 7,
    M_c: int = 3,
    M_b: int = 2,
    P: int = 2,
    tau_constr: float = 0.8,
    tau_item: float = 1.2,
    sigma_u: float = 0.5,
    seed: int = 20260827,
) -> tuple[dict, SimTruth]:
    """Returns (stan_data, truth). Scores are generated on the standardized
    scale (alpha near 0), matching what fit.build_stan_data produces for real
    data, so recovery is checked against the parameters as given."""
    rng = np.random.default_rng(seed)
    N = n_constr * items_per_constr
    constr = np.repeat(np.arange(1, n_constr + 1), items_per_constr)

    mu_c = rng.normal(0, tau_constr, n_constr)
    mu_c -= mu_c.mean()  # sum-to-zero, as in the Stan parameterization
    theta = mu_c[constr - 1] + rng.normal(0, tau_item, N)

    X = rng.normal(0, 1, (N, P))  # covariates arrive centered/scaled

    # human arm: each item rated by `ratings_per_item` random participants
    kappa = np.sort(rng.normal(0, 1.5, K - 1))
    u = rng.normal(0, sigma_u, n_part)
    item_h, part_h = [], []
    for i in range(1, N + 1):
        for p in rng.choice(n_part, size=ratings_per_item, replace=False):
            item_h.append(i)
            part_h.append(p + 1)
    item_h = np.array(item_h)
    part_h = np.array(part_h)
    y = _ordered_logistic_rng(theta[item_h - 1] + u[part_h - 1], kappa, rng)

    # continuous cells: every cell scores every item once; the linking function
    # carries family-level intercept/slope deviations (LLM error clusters by
    # phenomenon), mirroring a_dev/b_dev in the Stan model
    alpha = rng.normal(0, 0.2, M_c)
    beta = rng.uniform(0.3, 0.9, M_c)          # informative but imperfect instruments
    tau_a_true, tau_b_true = 0.15, 0.10
    a_dev = rng.normal(0, tau_a_true, (M_c, n_constr))
    b_dev = rng.normal(0, tau_b_true, (M_c, n_constr))
    gamma = rng.normal(0, 0.3, (M_c, P))
    omega_true = rng.uniform(0.2, 0.4, M_c)
    e_item = rng.normal(0, 1, (M_c, N)) * omega_true[:, None]
    sigma_s = rng.uniform(0.4, 0.9, M_c)
    reps_c = 3  # replicated cells, so the instrument-by-item error is identified
    item_c = np.tile(np.arange(1, N + 1), M_c * reps_c)
    cell_c = np.repeat(np.arange(1, M_c + 1), N * reps_c)
    fam_c = constr[item_c - 1]
    nu = alpha[cell_c - 1] + a_dev[cell_c - 1, fam_c - 1] + \
        (beta[cell_c - 1] + b_dev[cell_c - 1, fam_c - 1]) * theta[item_c - 1] + \
        np.einsum("np,np->n", gamma[cell_c - 1], X[item_c - 1]) + \
        e_item[cell_c - 1, item_c - 1]
    s = rng.normal(nu, sigma_s[cell_c - 1])

    # binary cells: every cell judges every item 5 times (repeat draws)
    a_b = rng.normal(0, 0.8, M_b)
    b_b = rng.uniform(0.5, 1.5, M_b)
    g_b = rng.normal(0, 0.3, (M_b, P))
    reps = 5
    item_b = np.tile(np.arange(1, N + 1), M_b * reps)
    cell_b = np.repeat(np.arange(1, M_b + 1), N * reps)
    eta = a_b[cell_b - 1] + b_b[cell_b - 1] * theta[item_b - 1] + \
        np.einsum("np,np->n", g_b[cell_b - 1], X[item_b - 1])
    z = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    data = dict(
        prior_only=0,
        N_item=N, N_constr=n_constr, constr=constr.tolist(),
        P=P, X=X.tolist(),
        N_h=len(y), K=K, item_h=item_h.tolist(), N_part=n_part,
        part_h=part_h.tolist(), y=y.tolist(),
        N_c=len(s), M_c=M_c, item_c=item_c.tolist(), cell_c=cell_c.tolist(),
        s=s.tolist(), has_reps=[1] * M_c,
        N_b=len(z), M_b=M_b, item_b=item_b.tolist(), cell_b=cell_b.tolist(),
        z=z.tolist(),
    )
    truth = SimTruth(theta, mu_c, tau_constr, kappa, sigma_u,
                     alpha, beta, gamma, sigma_s, omega_true, a_b, b_b, g_b)
    return data, truth


def recovery_check(idata, truth: SimTruth) -> dict:
    """Compare posterior to simulation truth. Returns a report dict with a
    boolean `passed`. Gates: theta 90% CI coverage in [0.80, 0.98]; every
    beta_m inside its 95% CI; reliability estimates within 0.15 of truth."""
    post = idata.posterior
    report = {}

    th = post["theta"].stack(d=("chain", "draw")).values  # (N, draws)
    lo, hi = np.percentile(th, [5, 95], axis=1)
    cover = float(np.mean((truth.theta >= lo) & (truth.theta <= hi)))
    report["theta_90ci_coverage"] = round(cover, 3)
    report["theta_post_mean_corr_truth"] = round(
        float(np.corrcoef(th.mean(axis=1), truth.theta)[0, 1]), 3)

    be = post["beta"].stack(d=("chain", "draw")).values
    blo, bhi = np.percentile(be, [2.5, 97.5], axis=1)
    beta_ok = bool(np.all((truth.beta >= blo) & (truth.beta <= bhi)))
    report["beta_within_95ci"] = beta_ok
    report["beta_post_mean"] = [round(v, 3) for v in be.mean(axis=1)]
    report["beta_truth"] = [round(v, 3) for v in truth.beta]

    rel = post["reliability"].stack(d=("chain", "draw")).values.mean(axis=1)
    v_theta = truth.theta.var()
    rel_truth = truth.beta**2 * v_theta / (
        truth.beta**2 * v_theta + truth.omega**2 + truth.sigma_s**2)
    rel_err = float(np.max(np.abs(rel - rel_truth)))
    report["reliability_max_abs_err"] = round(rel_err, 3)
    report["reliability_truth"] = [round(v, 3) for v in rel_truth]

    report["passed"] = bool(0.80 <= cover <= 0.98 and beta_ok and rel_err <= 0.15)
    return report


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n")

```


---

## FILE: src/acceptometer/model/sbc.py

```
"""SBC-lite: simulation-based calibration rank checks.

Unlike simulate.py (which uses convenient truth distributions for a single
recovery check), SBC draws every truth from the model's OWN priors, fits, and
records the rank of the truth within thinned posterior draws. If the pipeline
is self-consistent, ranks are uniform. We track the global scales and the
instrument slopes/noises; ~R=40 small replications is enough to catch gross
miscalibration (full SBC with many more replications is documented future work).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fit import fit_model
from .simulate import _ordered_logistic_rng


def _draw_prior(rng, n_constr, K, M_c, M_b, P):
    """Draw hyperparameters from exactly the priors in acceptometer.stan."""
    x = rng.normal(0, 1, n_constr)
    mu_c_raw = x - x.mean()                       # projected iid normal = sum_to_zero prior
    tau_constr = abs(rng.normal(0, 1))
    tau_item = rng.gamma(2.0, 1.0)
    kappa = np.sort(rng.normal(0, 2, K - 1))      # order statistics of iid normals
    sigma_u = abs(rng.normal(0, 1))
    # sigma_s has a 0.05 floor in the model: rejection-sample the truncation
    sigma_s = np.empty(M_c)
    for m in range(M_c):
        v = abs(rng.normal(0, 1))
        while v < 0.05:
            v = abs(rng.normal(0, 1))
        sigma_s[m] = v
    tau_a = abs(rng.normal(0, 0.5, M_c))
    tau_b = abs(rng.normal(0, 0.5, M_c))
    omega = abs(rng.normal(0, 0.5, M_c))
    return dict(
        mu_c=tau_constr * mu_c_raw,
        tau_constr=tau_constr,
        tau_item=tau_item,
        kappa=kappa,
        sigma_u=sigma_u,
        alpha=rng.normal(0, 1, M_c),
        beta=rng.normal(0, 1, M_c),
        tau_a=tau_a,
        tau_b=tau_b,
        a_dev=rng.normal(0, 1, (M_c, n_constr)) * tau_a[:, None],
        b_dev=rng.normal(0, 1, (M_c, n_constr)) * tau_b[:, None],
        gamma=rng.normal(0, 0.5, (M_c, P)),
        omega=omega,
        sigma_s=sigma_s,
        a_b=rng.normal(0, 1.5, M_b),
        b_b=rng.normal(0, 1, M_b),
        g_b=rng.normal(0, 0.5, (M_b, P)),
    )


def _simulate_given(pr, rng, n_constr, items_per_constr, n_part, ratings_per_item,
                    K, M_c, M_b, P):
    N = n_constr * items_per_constr
    e_item = rng.normal(0, 1, (M_c, N)) * pr["omega"][:, None]
    constr = np.repeat(np.arange(1, n_constr + 1), items_per_constr)
    theta = pr["mu_c"][constr - 1] + rng.normal(0, pr["tau_item"], N)
    X = rng.normal(0, 1, (N, P))

    u = rng.normal(0, pr["sigma_u"], n_part)
    item_h, part_h = [], []
    for i in range(1, N + 1):
        for p in rng.choice(n_part, size=min(ratings_per_item, n_part), replace=False):
            item_h.append(i); part_h.append(p + 1)
    item_h = np.array(item_h); part_h = np.array(part_h)
    y = _ordered_logistic_rng(theta[item_h - 1] + u[part_h - 1], pr["kappa"], rng)

    reps_c = 3
    item_c = np.tile(np.arange(1, N + 1), M_c * reps_c)
    cell_c = np.repeat(np.arange(1, M_c + 1), N * reps_c)
    fam_c = constr[item_c - 1]
    nu = pr["alpha"][cell_c - 1] + pr["a_dev"][cell_c - 1, fam_c - 1] + \
        (pr["beta"][cell_c - 1] + pr["b_dev"][cell_c - 1, fam_c - 1]) * theta[item_c - 1] + \
        np.einsum("np,np->n", pr["gamma"][cell_c - 1], X[item_c - 1]) + \
        e_item[cell_c - 1, item_c - 1]
    s = rng.normal(nu, pr["sigma_s"][cell_c - 1])

    reps = 3
    item_b = np.tile(np.arange(1, N + 1), M_b * reps)
    cell_b = np.repeat(np.arange(1, M_b + 1), N * reps)
    eta = pr["a_b"][cell_b - 1] + pr["b_b"][cell_b - 1] * theta[item_b - 1] + \
        np.einsum("np,np->n", pr["g_b"][cell_b - 1], X[item_b - 1])
    z = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    return dict(
        prior_only=0, N_item=N, N_constr=n_constr, constr=constr.tolist(),
        P=P, X=X.tolist(),
        N_h=len(y), K=K, item_h=item_h.tolist(), N_part=n_part,
        part_h=part_h.tolist(), y=y.tolist(),
        N_c=len(s), M_c=M_c, item_c=item_c.tolist(), cell_c=cell_c.tolist(),
        s=s.tolist(), has_reps=[1] * M_c,
        N_b=len(z), M_b=M_b, item_b=item_b.tolist(), cell_b=cell_b.tolist(),
        z=z.tolist(),
    )


def sbc_run(R: int = 40, n_thin: int = 63, seed: int = 5,
            n_constr: int = 4, items_per_constr: int = 6, n_part: int = 12,
            ratings_per_item: int = 6, K: int = 5, M_c: int = 2, M_b: int = 1,
            P: int = 2, out_path: str | Path | None = None) -> dict:
    """Run R replications; return rank-uniformity report.

    Tracked: tau_constr, sigma_u, beta[m], sigma_s[m]. Gate: chi-square
    uniformity p > 0.005 for every tracked parameter (loose Bonferroni; this is
    a smoke alarm, not a certificate)."""
    rng = np.random.default_rng(seed)
    tracked = ["tau_constr", "tau_item", "sigma_u"] + \
        [f"beta[{m}]" for m in range(1, M_c + 1)] + \
        [f"sigma_s[{m}]" for m in range(1, M_c + 1)]
    ranks: dict[str, list[int]] = {t: [] for t in tracked}
    n_failed = 0

    for r in range(R):
        pr = _draw_prior(rng, n_constr, K, M_c, M_b, P)
        data = _simulate_given(pr, rng, n_constr, items_per_constr, n_part,
                               ratings_per_item, K, M_c, M_b, P)
        try:
            fit, idata = fit_model(data, seed=seed + r, iter_warmup=300,
                                   iter_sampling=300, chains=2)
        except Exception:
            n_failed += 1
            continue
        post = idata.posterior
        for t in tracked:
            if "[" in t:
                base, i = t.split("["); i = int(i.rstrip("]")) - 1
                draws = post[base].stack(d=("chain", "draw")).values[i]
                truth = pr[base][i]
            else:
                draws = post[t].stack(d=("chain", "draw")).values
                truth = pr[t]
            step = max(len(draws) // n_thin, 1)
            thinned = draws[::step][:n_thin]
            ranks[t].append(int(np.sum(thinned < truth)))

    report = {"R": R, "n_failed_fits": n_failed, "n_thin": n_thin, "params": {}}
    B = 7  # rank bins
    passed = True
    for t, rk in ranks.items():
        if len(rk) < 10:
            report["params"][t] = {"error": "too few completed replications"}
            passed = False
            continue
        hist, _ = np.histogram(rk, bins=B, range=(0, n_thin + 1))
        expected = len(rk) / B
        chi2 = float(np.sum((hist - expected) ** 2 / expected))
        # chi-square tail via Wilson-Hilferty normal approximation (adequate
        # for a smoke alarm; avoids a scipy dependency)
        from math import erfc, sqrt
        df = B - 1
        zwh = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
        p = 0.5 * erfc(zwh / sqrt(2))
        report["params"][t] = {"chi2": round(chi2, 2), "p_uniform_approx": round(p, 4),
                               "rank_hist": hist.tolist()}
        if p < 0.005:
            passed = False
    report["passed"] = passed
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report

```


---

## FILE: src/acceptometer/model/loco.py

```
"""Leave-one-construction-family-out: the projectibility test.

For each construction family c, refit the model with family c's HUMAN data
removed (instrument scores retained for every item), then predict the held-out
family's human item means from the instrument arm alone, pushed through the
human response model learned on the other families. If the instrument's
validity projects across families, predictions cover the observed means; if it
was memorizing item structure, they won't. Feeds the warrant certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import build_stan_data, fit_model, diagnostics_gate


def _predict_observed_means(idata, item_indices_1based: list[int],
                            n_ratings: list[int], K: int,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive distribution of the OBSERVED mean rating per item,
    for a fresh sample of n_j new participants. Includes participant-effect
    variation and ordinal sampling noise, so interval coverage against the
    observed sample mean is calibrated by construction (an expected-rating
    interval would omit the criterion's own sampling noise and under-cover).
    Returns (mean, lo90, hi90) arrays over items."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values  # (N_item, L)
    kappa = post["kappa"].stack(d=("chain", "draw")).values  # (K-1, L)
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values  # (L,)
    L = theta.shape[1]

    means, los, his = [], [], []
    for pos, i1 in enumerate(item_indices_1based):
        n_j = max(int(n_ratings[pos]), 1)
        th = theta[i1 - 1]                                   # (L,)
        u = rng.normal(0.0, 1.0, (n_j, L)) * sigma_u[None, :]
        eta = th[None, :] + u                                # (n_j, L)
        cum = 1.0 / (1.0 + np.exp(-(kappa.T[None, :, :] - eta[..., None])))  # (n_j, L, K-1)
        r = rng.uniform(size=(n_j, L))
        y = 1 + (r[..., None] > cum).sum(axis=-1)            # (n_j, L) in 1..K
        sample_mean = y.mean(axis=0)                         # (L,)
        means.append(sample_mean.mean())
        lo, hi = np.percentile(sample_mean, [5, 95])
        los.append(lo)
        his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def loco(items: list, X: np.ndarray, human: pd.DataFrame,
         cont: pd.DataFrame | None, binary: pd.DataFrame | None,
         K: int = 7, families: list[str] | None = None,
         out_path: str | Path | None = None,
         iter_warmup: int = 1000, iter_sampling: int = 1000, seed: int = 11) -> dict:
    """Run the LOCO loop. Returns (and optionally writes) a report dict."""
    all_fams = sorted({it.construction for it in items})
    families = families or all_fams
    item_ids = [it.item_id for it in items]
    ix = {iid: i + 1 for i, iid in enumerate(item_ids)}

    per_family = {}
    for fam in families:
        held_items = [it.item_id for it in items if it.construction == fam]
        if not held_items:
            continue
        train_human = human[~human["item_id"].isin(held_items)]
        held = human[human["item_id"].isin(held_items)]
        obs = held.groupby("item_id")["rating"].mean()
        n_ratings = held.groupby("item_id")["rating"].count()
        if obs.empty:
            continue

        data, maps = build_stan_data(items, X, train_human, cont, binary, K=K)
        fit, idata = fit_model(data, seed=seed,
                               iter_warmup=iter_warmup, iter_sampling=iter_sampling)
        diag = diagnostics_gate(fit, idata)

        held_ix = [ix[i] for i in obs.index]
        pred, lo, hi = _predict_observed_means(
            idata, held_ix, [int(n_ratings[i]) for i in obs.index], K)
        o = obs.to_numpy()
        resid = pred - o
        # Spearman without scipy: rank-transform then Pearson
        def _rank(a):
            r = np.empty_like(a)
            r[np.argsort(a)] = np.arange(len(a), dtype=float)
            return r
        if len(o) > 2 and np.std(o) > 0 and np.std(pred) > 0:
            spearman = float(np.corrcoef(_rank(pred), _rank(o))[0, 1])
        else:
            spearman = float("nan")
        per_family[fam] = {
            "n_items": len(o),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "mean_signed_error": round(float(np.mean(resid)), 3),
            "spearman": round(spearman, 3) if spearman == spearman else None,
            "coverage90": round(float(np.mean((o >= lo) & (o <= hi))), 3),
            "diagnostics_passed": diag["passed"],
            "diagnostics": diag,
        }

    fams_ok = [v for v in per_family.values() if v["diagnostics_passed"]]
    report = {
        "per_family": per_family,
        "n_families": len(per_family),
        "mean_rmse": round(float(np.mean([v["rmse"] for v in fams_ok])), 3) if fams_ok else None,
        "mean_spearman": round(float(np.mean(
            [v["spearman"] for v in fams_ok if v["spearman"] is not None])), 3) if fams_ok else None,
        "mean_coverage90": round(float(np.mean([v["coverage90"] for v in fams_ok])), 3) if fams_ok else None,
        "all_diagnostics_passed": bool(fams_ok) and all(v["diagnostics_passed"] for v in per_family.values()),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report

```


---

## FILE: src/acceptometer/model/fit.py

```
"""CmdStanPy wrapper: data building, sampling, and the diagnostics gate.

Nothing downstream (warrant, plots, design) accepts a fit that fails the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STAN_FILE = Path(__file__).parent / "acceptometer.stan"


def build_stan_data(
    items: list,                      # list[acceptometer.items.Item]
    X: np.ndarray,                    # (N, P) nuisance covariates, raw scale
    human: pd.DataFrame | None,       # columns: item_id, participant_id, rating (1..K)
    cont: pd.DataFrame | None,        # columns: item_id, cell_id, value
    binary: pd.DataFrame | None,      # columns: item_id, cell_id, value (0/1)
    K: int = 7,
    standardize_scores: bool = True,
) -> tuple[dict, dict]:
    """Returns (stan_data, index_maps). index_maps records the id->index
    mappings and per-cell standardization constants so posteriors can be
    mapped back to names and raw scales."""
    item_ids = [it.item_id for it in items]
    item_ix = {iid: i + 1 for i, iid in enumerate(item_ids)}
    constrs = sorted({it.construction for it in items})
    constr_ix = {c: i + 1 for i, c in enumerate(constrs)}

    X = np.asarray(X, dtype=float)
    X_mean, X_sd = X.mean(axis=0), X.std(axis=0)
    X_sd[X_sd == 0] = 1.0
    Xs = (X - X_mean) / X_sd

    data: dict = dict(
        prior_only=0,
        N_item=len(items),
        N_constr=len(constrs),
        constr=[constr_ix[it.construction] for it in items],
        P=Xs.shape[1],
        X=Xs.tolist(),
    )
    maps: dict = dict(
        item_ids=item_ids, constructions=constrs,
        X_mean=X_mean.tolist(), X_sd=X_sd.tolist(),
        cont_cells=[], bin_cells=[], participants=[], cell_standardization={},
    )

    if human is not None and len(human):
        parts = sorted(human["participant_id"].unique())
        part_ix = {p: i + 1 for i, p in enumerate(parts)}
        maps["participants"] = list(parts)
        data.update(
            N_h=len(human), K=K,
            item_h=[item_ix[i] for i in human["item_id"]],
            N_part=len(parts),
            part_h=[part_ix[p] for p in human["participant_id"]],
            y=human["rating"].astype(int).tolist(),
        )
    else:
        data.update(N_h=0, K=K, item_h=[], N_part=1, part_h=[], y=[])

    if cont is not None and len(cont):
        cells = sorted(cont["cell_id"].unique())
        cix = {c: i + 1 for i, c in enumerate(cells)}
        maps["cont_cells"] = cells
        vals = cont["value"].astype(float).to_numpy().copy()
        if standardize_scores:
            for c in cells:
                m = (cont["cell_id"] == c).to_numpy()
                mu, sd = vals[m].mean(), vals[m].std() or 1.0
                vals[m] = (vals[m] - mu) / sd
                maps["cell_standardization"][c] = {"mean": float(mu), "sd": float(sd)}
        reps = cont.groupby(["cell_id", "item_id"]).size()
        has_reps = [int(reps.xs(c, level="cell_id").max() > 1) for c in cells]
        data.update(
            N_c=len(cont), M_c=len(cells),
            item_c=[item_ix[i] for i in cont["item_id"]],
            cell_c=[cix[c] for c in cont["cell_id"]],
            s=vals.tolist(),
            has_reps=has_reps,
        )
    else:
        data.update(N_c=0, M_c=0, item_c=[], cell_c=[], s=[], has_reps=[0])

    if binary is not None and len(binary):
        cells = sorted(binary["cell_id"].unique())
        cix = {c: i + 1 for i, c in enumerate(cells)}
        maps["bin_cells"] = cells
        data.update(
            N_b=len(binary), M_b=len(cells),
            item_b=[item_ix[i] for i in binary["item_id"]],
            cell_b=[cix[c] for c in binary["cell_id"]],
            z=binary["value"].astype(int).tolist(),
        )
    else:
        data.update(N_b=0, M_b=0, item_b=[], cell_b=[], z=[])

    return data, maps


def _default_inits(data: dict) -> dict:
    """Start chains in the identified basin. The joint posterior has a
    spurious reflection mode (instrument slopes and latent orientation jointly
    flipped) that is locally stable: escaping it requires every theta to cross
    the likelihood valley at once, which HMC will not do. Random inits land
    ~1-in-4 chains there. So initialize the latent items at the standardized
    observed human item means (data-informed inits select a basin; they do not
    bias the posterior), with modest positive slopes. A genuinely
    anti-correlated instrument can still walk to negative beta."""
    inits = {
        "tau_constr": 0.5, "tau_item": 1.0, "sigma_u": 0.5,
        "kappa": [float(k) for k in np.linspace(-2, 2, data["K"] - 1)],
    }
    if data.get("N_h", 0) > 0:
        item_h = np.asarray(data["item_h"])
        y = np.asarray(data["y"], dtype=float)
        means = np.full(data["N_item"], y.mean())
        for i in range(1, data["N_item"] + 1):
            m = item_h == i
            if m.any():
                means[i - 1] = y[m].mean()
        sd = means.std() or 1.0
        inits["z_item"] = ((means - means.mean()) / sd).tolist()
    if data.get("M_c", 0) > 0:
        inits["beta"] = [0.3] * data["M_c"]
        inits["sigma_s"] = [0.8] * data["M_c"]
        inits["omega"] = [0.3] * data["M_c"]
    if data.get("M_b", 0) > 0:
        inits["b_b"] = [0.5] * data["M_b"]
    return inits


def fit_model(data: dict, out_dir: str | Path | None = None, seed: int = 1,
              iter_warmup: int = 1000, iter_sampling: int = 1000,
              adapt_delta: float = 0.95, chains: int = 4, inits: dict | None = None):
    """Compile (cached), sample, and return (CmdStanMCMC, arviz.InferenceData)."""
    import arviz as az
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(
        data=data, chains=chains, parallel_chains=min(chains, 4),
        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        adapt_delta=adapt_delta, seed=seed, show_progress=False,
        inits=inits if inits is not None else _default_inits(data),
        output_dir=str(out_dir) if out_dir else None,
    )
    idata = az.from_cmdstanpy(fit)
    return fit, idata


def diagnostics_gate(fit, idata) -> dict:
    """Hard gate: divergences < 0.5%, max R-hat < 1.01, min bulk ESS > 400
    on core parameters. Returns report dict with `passed`."""
    import arviz as az

    div = int(np.sum(fit.method_variables()["divergent__"]))
    n_draws = int(np.prod(fit.method_variables()["divergent__"].shape))
    core = [v for v in ["beta", "sigma_s", "tau_constr", "kappa", "sigma_u", "b_b"]
            if v in idata.posterior]
    summ = az.summary(idata, var_names=core)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    report = {
        "divergences": div,
        "divergence_rate": round(div / n_draws, 4),
        "rhat_max": round(rhat_max, 4),
        "ess_bulk_min": int(ess_min),
        "passed": bool(div / n_draws < 0.005 and rhat_max < 1.01 and ess_min > 400),
    }
    return report


def save_fit(idata, maps: dict, report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(out / "posterior.nc"))
    (out / "index_maps.json").write_text(json.dumps(maps, indent=2))
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2))

```


---

## FILE: src/acceptometer/model/ppc.py

```
"""Posterior predictive checks at the real-fit stage, with consequences.

Simulates human rating datasets from the fitted model (same items,
participants, and design as observed) and compares three statistics the model
could plausibly fail on:

- per-family spread of item mean ratings (does the model reproduce how much
  items differ within a family?)
- global category usage (does it use the response scale the way people did?)
- disagreement structure: mean within-item rating SD (the statistic binary
  labels suppress and this tool exists to respect)

Each check reports a posterior predictive p-value (two-sided, tail mass of the
observed statistic under the predictive distribution). A failure (p < 0.01) on
disagreement structure refuses the distributional tier and flags aggregate
estimation in the warrant; the consequences are wired there, the evidence is
computed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_ratings(theta_d, u_d, kappa_d, item_idx0, part_idx0, rng):
    """One simulated dataset for one posterior draw (vectorized over obs)."""
    eta = theta_d[item_idx0] + u_d[part_idx0]
    cum = 1.0 / (1.0 + np.exp(-(kappa_d[None, :] - eta[:, None])))
    r = rng.uniform(size=eta.shape[0])
    return 1 + (r[:, None] > cum).sum(axis=1)


def _ppp(observed: float, sims: np.ndarray) -> float:
    """Two-sided posterior predictive p-value."""
    lo = float(np.mean(sims <= observed))
    hi = float(np.mean(sims >= observed))
    return round(2 * min(lo, hi), 4)


def ppc_human(idata, maps: dict, human: pd.DataFrame,
              items: list, n_sims: int = 200, seed: int = 2,
              out_path: str | Path | None = None) -> dict:
    """Run the three checks against the observed human data."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    u = post["u"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    K = kappa.shape[0] + 1

    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}
    fam_of = {it.item_id: it.construction for it in items}

    item_idx0 = human["item_id"].map(item_pos).to_numpy()
    part_idx0 = human["participant_id"].map(part_pos).to_numpy()
    y_obs = human["rating"].astype(int).to_numpy()
    fams = human["item_id"].map(fam_of).to_numpy()

    def stats(y):
        df = pd.DataFrame({"y": y, "item": human["item_id"].to_numpy(), "fam": fams})
        item_means = df.groupby("item")["y"].mean()
        fam_of_item = df.groupby("item")["fam"].first()
        fam_spread = item_means.groupby(fam_of_item).std().mean()
        cat_usage = np.bincount(y, minlength=K + 1)[1:] / len(y)
        within_sd = df.groupby("item")["y"].std().mean()
        return float(fam_spread), cat_usage, float(within_sd)

    obs_spread, obs_cats, obs_within = stats(y_obs)

    draws = rng.choice(L, size=min(n_sims, L), replace=False)
    sim_spread, sim_within, sim_cat_dev = [], [], []
    for d in draws:
        y_sim = _simulate_ratings(theta[:, d], u[:, d], kappa[:, d],
                                  item_idx0, part_idx0, rng)
        sp, cats, wi = stats(y_sim)
        sim_spread.append(sp)
        sim_within.append(wi)
        sim_cat_dev.append(float(np.abs(cats - obs_cats).sum() / 2))  # TV distance

    # for category usage the discrepancy is TV distance to observed, so the
    # reference is "how far is a typical replicate from ITS OWN generating
    # distribution": compare each replicate's TV to the mean replicate TV
    tv = np.array(sim_cat_dev)
    report = {
        "family_item_mean_spread": {
            "observed": round(obs_spread, 3),
            "ppp": _ppp(obs_spread, np.array(sim_spread)),
        },
        "within_item_disagreement_sd": {
            "observed": round(obs_within, 3),
            "ppp": _ppp(obs_within, np.array(sim_within)),
        },
        "category_usage_tv_distance": {
            "mean_replicate_tv": round(float(tv.mean()), 3),
            "note": "TV distance between replicate and observed category "
                    "frequencies; large values mean the model uses the scale "
                    "differently than people did",
        },
        "n_sims": len(draws),
    }
    report["passed"] = bool(
        report["family_item_mean_spread"]["ppp"] >= 0.01
        and report["within_item_disagreement_sd"]["ppp"] >= 0.01
        and tv.mean() <= 0.15
    )
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report

```


---

## FILE: src/acceptometer/warrant.py

```
"""Validity-certificate builder (warrant.yaml).

The certificate answers the projectibility question the posterior cannot:
which claim tiers does the evidence on disk license? Every number in the
certificate comes from a file actually read out of the run directory;
missing evidence refuses the dependent tiers, it never grants them.

Files read from run_dir:
  diagnostics.json     required as evidence (missing refuses every tier)
  index_maps.json      required (names for cells, items, families)
  posterior.nc         required (reliability and bias summaries)
  recovery.json        optional (fake-data recovery report)
  loco.json            optional (leave-one-construction-out transfer stats)
  grid_manifest.yaml   optional (contamination record from the grid runner)
  cont.csv             optional (raw continuous scores written by the fit
                       and simulate CLI commands; used for the multiverse
                       spread statistic, which no other artifact carries)
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TIERS = (
    "screening",
    "ranking",
    "aggregate_estimation",
    "effect_reproduction",
    "distributional_claims",
    "population_transfer",
    "individual_simulation",
    "mechanism_claims",
)

_RANK_TOP_KEYS = ("mean_spearman", "mean_heldout_rank_correlation",
                  "mean_rank_correlation", "mean_rank_corr")
_RANK_FOLD_KEYS = ("spearman", "rank_correlation", "rank_corr")
_COV_TOP_KEYS = ("mean_coverage90", "coverage_90", "interval_coverage_90",
                 "coverage")
_COV_FOLD_KEYS = ("coverage90", "coverage_90", "coverage")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _loco_stat(loco: dict, top_keys: tuple, fold_keys: tuple) -> float | None:
    """Pull a scalar from loco.json: a top-level key under any of the known
    names, else the mean of per-fold values. Returns None when the file
    carries no recognizable value (which refuses the dependent tier)."""
    for k in top_keys:
        v = loco.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    folds = loco.get("per_family") or loco.get("families") or loco.get("folds")
    if isinstance(folds, dict):
        folds = list(folds.values())
    vals: list[float] = []
    if isinstance(folds, list):
        for f in folds:
            if not isinstance(f, dict):
                continue
            for k in fold_keys:
                v = f.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
                    break
    return float(np.mean(vals)) if vals else None


def _multiverse_spread(run_dir: Path):
    """Mean over items of the per-item sd of standardized scores across
    continuous cells, from cont.csv. Scores are z-scored per cell here so
    the statistic is scale-free whatever scale the file stores."""
    path = run_dir / "cont.csv"
    if not path.exists():
        return "not assessed (no cont.csv in run dir)"
    df = pd.read_csv(path)
    per = df.groupby(["item_id", "cell_id"], as_index=False)["value"].mean()
    if per["cell_id"].nunique() < 2:
        return "not assessed (fewer than 2 continuous cells)"

    def _z(v: pd.Series) -> pd.Series:
        sd = v.std(ddof=0)
        return (v - v.mean()) / (sd if sd > 0 else 1.0)

    per["z"] = per.groupby("cell_id")["value"].transform(_z)
    pivot = per.pivot(index="item_id", columns="cell_id", values="z")
    sds = pivot.std(axis=1, ddof=1).dropna()
    return {
        "mean_per_item_sd_across_cells": round(float(sds.mean()), 3),
        "n_items": int(len(sds)),
        "n_cells": int(pivot.shape[1]),
    }


def _prompt_invariance(cont_cells: list[str], beta_med: np.ndarray):
    """sd of posterior-median beta across paraphrase cells of the same
    model and method, identified from cell names of the registered form
    model/method/paraphrase. Returns a per-group dict, or a string when no
    such group is identifiable."""
    groups: dict[str, list[int]] = defaultdict(list)
    for j, name in enumerate(cont_cells):
        parts = name.split("/")
        if len(parts) >= 3:
            groups["/".join(parts[:2])].append(j)
    out = {}
    for g, idxs in sorted(groups.items()):
        if len(idxs) >= 2:
            meds = [float(beta_med[j]) for j in idxs]
            out[g] = {
                "sd_of_beta_medians": round(float(np.std(meds, ddof=1)), 3),
                "cells": [cont_cells[j] for j in idxs],
            }
    return out if out else "not identifiable from cell names"


def _plain(x):
    """Recursively convert numpy scalars and paths so yaml.safe_dump accepts
    the certificate."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def build_warrant(run_dir: str | Path, estimand: dict,
                  out_path: str | Path | None = None) -> dict:
    """Build the validity certificate for one fit and write it as YAML to
    run_dir/warrant.yaml (or out_path). Returns the certificate dict.

    Granting is conservative and hierarchical: screening needs the
    diagnostics gate plus at least one cell with reliability median > 0.5;
    ranking additionally needs LOCO mean held-out rank correlation > 0.6;
    aggregate estimation additionally needs LOCO 90% interval coverage in
    [0.75, 0.98]. Effect reproduction and distributional claims are refused
    in v1; individual simulation and mechanism claims are refused
    permanently. A missing evidence file refuses the dependent tiers with
    reason "evidence not produced", never grants them."""
    import arviz as az

    run_dir = Path(run_dir)
    maps_path = run_dir / "index_maps.json"
    post_path = run_dir / "posterior.nc"
    if not maps_path.exists():
        raise FileNotFoundError(f"{maps_path} not found; cannot build a warrant")
    if not post_path.exists():
        raise FileNotFoundError(f"{post_path} not found; cannot build a warrant")
    maps = json.loads(maps_path.read_text())
    idata = az.from_netcdf(str(post_path))

    diagnostics = _read_json(run_dir / "diagnostics.json")
    recovery = _read_json(run_dir / "recovery.json")
    loco = _read_json(run_dir / "loco.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None

    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    rel_meds: list[float] = []
    if cont_cells and "reliability" in idata.posterior:
        rel = np.asarray(idata.posterior["reliability"].values)
        rel = rel.reshape(-1, rel.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        beta = np.asarray(idata.posterior["beta"].values)
        beta = beta.reshape(-1, beta.shape[-1])
        beta_med = np.median(beta, axis=0)
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel[:, j]))
            lo, hi = np.percentile(rel[:, j], [5, 95])
            rel_meds.append(med)
            instruments.append({
                "cell": name,
                "reliability_median": round(med, 3),
                "reliability_90": [round(float(lo), 3), round(float(hi), 3)],
                "bias_alpha_median": round(float(np.median(alpha[:, j])), 3),
            })
        invariance = _prompt_invariance(cont_cells, beta_med)
    else:
        invariance = "not identifiable from cell names"

    estimand = dict(estimand)
    split_half = estimand.pop("human_split_half", "not provided")
    domain = {
        "construction_families": list(maps.get("constructions", [])),
        "n_items": len(maps.get("item_ids", [])),
    }
    user_domain = estimand.pop("domain", None)
    if isinstance(user_domain, dict):
        domain.update(user_domain)

    if isinstance(manifest, dict) and "contamination" in manifest:
        contamination = manifest["contamination"]
    else:
        contamination = "not assessed"
    # contamination caps tiers: only an explicit clean assessment lifts the
    # cap, because contamination inflates exactly the statistics the higher
    # tiers rest on, and LOCO rewards it (the held-out family is public too)
    contamination_clean = (
        contamination == "clean"
        or (isinstance(contamination, dict)
            and contamination.get("status") == "clean"))

    ppc = _read_json(run_dir / "ppc.json")
    ppc_ok = bool(ppc and ppc.get("passed"))

    # nonresponse: cells with a parse-failure rate above 10% cannot support
    # tier grants (missingness plausibly correlates with theta)
    nonresponse = (manifest or {}).get("nonresponse") if isinstance(manifest, dict) else None
    flagged_cells = sorted(
        c for c, r in (nonresponse or {}).items()
        if isinstance(r, (int, float)) and r > 0.10)

    rank_corr = _loco_stat(loco, _RANK_TOP_KEYS, _RANK_FOLD_KEYS) if loco else None
    coverage = _loco_stat(loco, _COV_TOP_KEYS, _COV_FOLD_KEYS) if loco else None

    evidence = {
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "ppc": ppc if ppc is not None else "not produced",
        "multiverse_spread": _multiverse_spread(run_dir),
        "prompt_invariance": invariance,
        "human_split_half": split_half,
        "contamination": contamination,
        "instrument_nonresponse": (
            nonresponse if nonresponse is not None else "not recorded"),
        "flagged_cells": flagged_cells or "none",
        "generalization_axes": {
            "tested": ["construction_family (LOCO-CV, purposive family sample; "
                       "descriptive over the families tested)"],
            "untested": ["population", "register", "language", "item_source",
                         "time", "model_version (beyond drift sentinels)"],
        },
    }

    licensed: dict[str, str] = {}
    refused: dict[str, str] = {}
    diag_ok = bool(diagnostics and diagnostics.get("passed"))
    max_rel = max(rel_meds) if rel_meds else None

    if diagnostics is None:
        refused["screening"] = "evidence not produced: diagnostics.json missing"
    elif not diag_ok:
        refused["screening"] = (
            f"diagnostics gate failed (rhat_max={diagnostics.get('rhat_max')}, "
            f"divergence_rate={diagnostics.get('divergence_rate')}, "
            f"ess_bulk_min={diagnostics.get('ess_bulk_min')})")
    elif max_rel is None:
        refused["screening"] = ("evidence not produced: no continuous cell "
                                "carries a posterior reliability")
    elif max_rel > 0.5:
        licensed["screening"] = (f"diagnostics passed and max cell reliability "
                                 f"median {max_rel:.2f} > 0.5")
    else:
        refused["screening"] = (f"max cell reliability median {max_rel:.2f} "
                                f"does not exceed 0.5")

    if "screening" not in licensed:
        refused["ranking"] = "refused because screening is not granted"
    elif loco is None:
        refused["ranking"] = "evidence not produced: loco.json missing"
    elif rank_corr is None:
        refused["ranking"] = ("evidence not produced: loco.json carries no "
                              "held-out rank correlation")
    elif rank_corr > 0.6:
        licensed["ranking"] = (f"screening granted and LOCO mean held-out rank "
                               f"correlation {rank_corr:.2f} > 0.6")
    else:
        refused["ranking"] = (f"LOCO mean held-out rank correlation "
                              f"{rank_corr:.2f} does not exceed 0.6")

    if "ranking" not in licensed:
        refused["aggregate_estimation"] = "refused because ranking is not granted"
    elif not contamination_clean:
        refused["aggregate_estimation"] = (
            "contamination cap: item source not explicitly assessed clean "
            f"(status: {contamination if isinstance(contamination, str) else contamination.get('status', 'unknown')}); "
            "contamination inflates exactly the statistics this tier rests on, "
            "and LOCO rewards it")
    elif ppc is not None and not ppc_ok:
        refused["aggregate_estimation"] = (
            "posterior predictive check failed; the human arm misfits the "
            "criterion data, so aggregate predictions inherit unquantified bias")
    elif coverage is None:
        refused["aggregate_estimation"] = ("evidence not produced: loco.json "
                                           "carries no 90% interval coverage")
    elif 0.75 <= coverage <= 0.98:
        licensed["aggregate_estimation"] = (f"LOCO 90% interval coverage "
                                            f"{coverage:.2f} within [0.75, 0.98], "
                                            "contamination assessed clean"
                                            + ("" if ppc is None else ", PPC passed"))
    else:
        refused["aggregate_estimation"] = (f"LOCO 90% interval coverage "
                                           f"{coverage:.2f} outside [0.75, 0.98]")

    refused["effect_reproduction"] = ("not yet tested: requires matched "
                                      "experimental contrasts")
    refused["distributional_claims"] = (
        "no participant-level validation of variance structure"
        + ("" if ppc is None or ppc_ok
           else "; posterior predictive check on disagreement structure failed"))
    refused["population_transfer"] = (
        "refused: the v1 ladder contains no population-transfer test; the "
        "estimand population defaults to the criterion sample's own")
    refused["individual_simulation"] = ("refused permanently: an item-level "
                                        "instrument licenses no individual-level "
                                        "human simulation")
    refused["mechanism_claims"] = ("refused permanently: the model estimates a "
                                   "linking function, not a mechanism")

    if "population" not in estimand:
        estimand["population"] = ("the criterion sample's population "
                                  "(unspecified); population transfer untested")

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "estimand": estimand,
        "domain": domain,
        "instruments": instruments,
        "evidence": evidence,
        "licensed_claims": licensed,
        "refused_claims": refused,
        "residual_risks": [
            "shared pretraining bias: local instruments share web-scale "
            "training data and can share construction-specific error; a tight "
            "multiverse fan does not rule this out",
            "conditional reliability is fit- and item-set-specific; it goes "
            "stale when the item pool changes",
        ],
    })

    out_path = Path(out_path) if out_path else run_dir / "warrant.yaml"
    out_path.write_text(yaml.safe_dump(cert, sort_keys=False, allow_unicode=True))
    return cert

```


---

## FILE: src/acceptometer/elicit/hf_logprob.py

```
"""Exact sentence log-probability cells from local Hugging Face models."""

from __future__ import annotations

import math
import string
from datetime import datetime, timezone

from ..items import Item
from .base import CONTINUOUS, CellSpec, Measurement


class HFLogprobScorer:
    """Score causal-LM token probabilities in padded batches."""

    def __init__(
        self,
        model_id: str = "EleutherAI/pythia-160m",
        device: str | None = None,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "HFLogprobScorer requires the 'hf' optional dependencies"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        self.model_id = model_id
        self.model_name = model_id.rsplit("/", 1)[-1]
        mps = getattr(torch.backends, "mps", None)
        self.device = str(
            device or ("mps" if mps is not None and mps.is_available() else "cpu")
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        model_revision = getattr(self.model.config, "_name_or_path", model_id)
        self.revision = f"{model_revision}; transformers={transformers.__version__}"

    def cells(self) -> list[CellSpec]:
        """Return the three deterministic log-probability cells."""
        model = f"hf:{self.model_id}"
        return [
            CellSpec(
                cell_id=f"{self.model_name}/{method}",
                model=model,
                method=method,
                kind=CONTINUOUS,
                params={"deterministic": True},
            )
            for method in ("logprob_sum", "logprob_mean", "slor")
        ]

    def score(
        self,
        items: list[Item],
        cell: CellSpec,
        repeats: int = 1,
    ) -> list[Measurement]:
        """Score one cell; deterministic cells ignore requested repeats."""
        if cell.method not in {"logprob_sum", "logprob_mean", "slor"}:
            raise ValueError(f"unsupported HF log-probability cell: {cell.cell_id}")

        now = datetime.now(timezone.utc).isoformat()
        rows = self._sentence_scores(items)
        out = []
        for item, row in zip(items, rows, strict=True):
            meta = {
                "model_id": self.model_id,
                "revision": self.revision,
                "device": self.device,
                "date": now,
                "n_scored_tokens": row["n_tokens"],
                "first_token_skipped": row["first_token_skipped"],
            }
            if cell.method == "slor":
                meta["unigram_source"] = "wordfreq-zipf"
            out.append(
                Measurement(
                    item_id=item.item_id,
                    cell_id=cell.cell_id,
                    kind=CONTINUOUS,
                    value=float(row[cell.method]),
                    meta=meta,
                )
            )
        return out

    def _sentence_scores(self, items: list[Item]) -> list[dict[str, float | int | bool]]:
        bos_id = self.tokenizer.bos_token_id
        encoded = [
            self.tokenizer.encode(item.text, add_special_tokens=False) for item in items
        ]
        results: list[dict[str, float | int | bool]] = []

        for start in range(0, len(items), 16):
            batch_ids = encoded[start : start + 16]
            sequences = [([bos_id] + ids if bos_id is not None else ids) for ids in batch_ids]
            batch_sums = self._batch_logprob_sums(sequences)
            for item, token_ids, logprob_sum in zip(
                items[start : start + 16], batch_ids, batch_sums, strict=True
            ):
                n_tokens = len(token_ids) if bos_id is not None else max(len(token_ids) - 1, 0)
                unigram_sum = self._unigram_logprob_sum(item)
                denominator = float(n_tokens)
                results.append(
                    {
                        "logprob_sum": logprob_sum,
                        "logprob_mean": logprob_sum / denominator if n_tokens else math.nan,
                        "slor": (
                            (logprob_sum - unigram_sum) / denominator
                            if n_tokens
                            else math.nan
                        ),
                        "n_tokens": n_tokens,
                        "first_token_skipped": bos_id is None,
                    }
                )
        return results

    def _batch_logprob_sums(self, sequences: list[list[int]]) -> list[float]:
        if not sequences:
            return []
        torch = self._torch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.bos_token_id
        if pad_id is None:
            pad_id = 0
        width = max(max((len(sequence) for sequence in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), width), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(sequences), width), dtype=torch.long, device=self.device
        )
        for row, sequence in enumerate(sequences):
            if sequence:
                input_ids[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
                attention_mask[row, : len(sequence)] = 1

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            log_probs = logits.log_softmax(dim=-1)

        sums = []
        for row, sequence in enumerate(sequences):
            total = 0.0
            for position in range(1, len(sequence)):
                total += log_probs[
                    row, position - 1, sequence[position]
                ].item()
            sums.append(total)
        return sums

    @staticmethod
    def _unigram_logprob_sum(item: Item) -> float:
        from wordfreq import zipf_frequency

        total = 0.0
        for raw_word in item.text.split():
            word = raw_word.strip(string.punctuation).lower()
            if not word:
                continue
            zipf = max(float(zipf_frequency(word, item.language)), 1.0)
            total += (zipf - 9.0) * math.log(10.0)
        return total

```


---

## FILE: PILOT-REPORT.md

```
# Acceptometer pilot report: Sprouse LI items, local instruments
<!-- SUMMARY: First real-data run of the warranted measurement pipeline; screening+ranking licensed, aggregate estimation refused on contamination · status: complete · updated: 2026-08-27 -->

## What ran

120 items from the Sprouse, Schütze & Almeida (2013) Linguistic Inquiry
judgment study (10 source-paper families, 6 starred / 6 good each), with all
1,519 available participant-level 7-point ratings (304 participants, ~12.7
ratings/item). Instruments, all local and free: Pythia-160m exact log-probability
cells (SLOR entering the fit) and qwen3:8b prompted judgments (3 registered
paraphrases x binary and 1-7 scalar x 3 repeats at temperature 0.7; 2,160
chat calls, zero parse failures, 11.5 minutes).

Every stage ran behind its gate: fake-data recovery, SBC-lite (40 prior-drawn
replications, all rank distributions uniform), convergence diagnostics,
posterior predictive checks, and leave-one-construction-family-out transfer.

## Headline numbers

| Quantity | Value |
|---|---|
| Human split-half reliability (item means, 200 splits) | r = .857 (Spearman-Brown .923) |
| qwen3:8b pooled scalar, item-mean r with humans | .724 |
| Pythia-160m SLOR, item-mean r with humans | .345 |
| qwen3:8b conditional reliability (joint model) | .54 |
| Pythia SLOR conditional reliability | .34 |
| qwen instrument-by-item error (omega) | .56 latent-logit units |
| LOCO mean held-out Spearman (10 families) | .675 |
| LOCO mean 90% coverage of observed means | .925 |
| LOCO mean RMSE (7-point scale) | 1.00 |

Per-family LOCO tells the projectibility story: nine of ten families transfer
at Spearman .51-.94, and one (34.1.fox) fails outright (-.08, RMSE 1.47). The
instrument's validity is family-dependent, which is exactly what the
family-varying linking structure exists to expose; a single pooled correlation
would have hidden fox entirely.

## The warrant (runs/pilot/warrant.yaml)

Licensed: **screening** (max cell conditional reliability .54 > .5) and
**ranking** (LOCO held-out rank correlation .68 > .6). Refused: **aggregate
estimation**, despite in-band coverage (.925), because the item source is
contamination-suspect (LI materials, public since 2013) and contamination
inflates exactly the statistics that tier rests on. Also refused, with reasons:
effect reproduction, distributional claims, population transfer, individual
simulation (permanent), mechanism claims (permanent). Residual risks recorded:
shared pretraining bias across local instruments; reliability is item-set
specific.

## What real data corrected in the model (all caught by gates)

1. **PPC failure 1 -> freed latent scale.** Fixing within-family item sd at 1
   compressed real item spread (obs 1.53 vs replicated 1.33) and inflated
   within-item noise (obs 1.44 vs 1.63). tau_item freed; the data put it at
   ~2.0; PPC passed.
2. **Pseudo-replicated instruments.** Three scalar paraphrases (and three
   transforms of one Pythia forward pass) entered as independent witnesses and
   outvoted 1,519 human ratings (fictitious reliability .93). Rule now: one
   instrument per model+method; paraphrases and repeats are repeated
   measurements.
3. **Reflection mode.** A locally-stable mirrored posterior mode captured
   1-in-4 randomly-initialized chains. Data-informed initialization (latents
   start at standardized human item means) plus a zero-avoiding gamma(2,1)
   prior on tau_item; no sign constraints anywhere.
4. **PPC failure 2 -> instrument-by-item error.** Repeats average away draw
   noise but never the instrument's stable opinion about an item; without an
   omega term the qwen cell claimed reliability .90 and dragged theta off the
   human criterion. With it: reliability .54, omega .56, PPC passes (ppp
   .86/.78). The .90 -> .54 drop is the honest price of correlated repeats.

## Reading the numbers against the literature

The published r = .8 human-LLM correlations (Qiu et al. 2024, ChatGPT) sit
between this pilot's qwen3:8b (.72 raw item-mean correlation) and the human
split-half ceiling (.857). The pilot's contribution is not the correlation but
the decomposition: how much of that correlation is theta-signal (reliability
.54), how much is stable per-item disagreement (omega .56), how it transfers
family-by-family (.51-.94, with one outright failure), and which claims the
whole package does and does not license, on a certificate.

## Caveats

Single language, single item source, constructed sentences, two instruments
from two model families, one register, one date. The certificate lists the
untested generalization axes explicitly. Prompt invariance was strong
descriptively (paraphrase item-mean correlations .67-.73 with humans, pooled
.72) but is not yet a model-based statistic in the collapsed-cell fit.
Contamination is suspected, not shown; a clean post-cutoff item set is the
v2 test that would lift the aggregate-estimation cap.

## Artifacts

`runs/pilot/`: posterior.nc, diagnostics.json, ppc.json, loco.json,
warrant.yaml, estimand.yaml, split_half.json, measurements.jsonl,
grid_manifest.yaml, and four plots (item_scatter, reliability_forest,
secret_weapon, multiverse_fan). Data provenance: `data/MANIFEST.yaml`.

```


---

## EVIDENCE: run reports (final model)

### fake-data recovery + diagnostics (runs/verify-cli is pre-omega; the current recovery numbers)

```
diag: divergences 0, rhat_max 1.0054, ess_bulk_min 830, passed true
rec: theta_90ci_coverage 0.875, corr 0.961, beta_within_95ci true, reliability_max_abs_err 0.079, omega recovered [0.41,0.31,0.21] vs truth [0.37,0.27,0.21], passed true
```

### SBC-lite, final model (runs/sbc-report.json)

```json
{
  "R": 40,
  "n_failed_fits": 0,
  "n_thin": 63,
  "params": {
    "tau_constr": {
      "chi2": 3.4,
      "p_uniform_approx": 0.7592,
      "rank_hist": [
        7,
        5,
        7,
        3,
        6,
        4,
        8
      ]
    },
    "tau_item": {
      "chi2": 3.75,
      "p_uniform_approx": 0.7126,
      "rank_hist": [
        5,
        6,
        3,
        8,
        4,
        6,
        8
      ]
    },
    "sigma_u": {
      "chi2": 7.95,
      "p_uniform_approx": 0.2409,
      "rank_hist": [
        11,
        6,
        3,
        5,
        5,
        3,
        7
      ]
    },
    "beta[1]": {
      "chi2": 4.8,
      "p_uniform_approx": 0.5714,
      "rank_hist": [
        8,
        6,
        4,
        5,
        5,
        3,
        9
      ]
    },
    "beta[2]": {
      "chi2": 9.35,
      "p_uniform_approx": 0.1537,
      "rank_hist": [
        4,
        12,
        3,
        6,
        5,
        4,
        6
      ]
    },
    "sigma_s[1]": {
      "chi2": 8.65,
      "p_uniform_approx": 0.1932,
      "rank_hist": [
        6,
        4,
        3,
        4,
        4,
        11,
        8
      ]
    },
    "sigma_s[2]": {
      "chi2": 8.3,
      "p_uniform_approx": 0.2159,
      "rank_hist": [
        4,
        9,
        2,
        9,
        7,
        6,
        3
      ]
    }
  },
  "passed": true
}

```

### Pilot fit diagnostics + PPC (runs/pilot)

```json
{
  "divergences": 0,
  "divergence_rate": 0.0,
  "rhat_max": 1.0094,
  "ess_bulk_min": 501,
  "passed": true
}
{
  "family_item_mean_spread": {
    "observed": 1.528,
    "ppp": 0.86
  },
  "within_item_disagreement_sd": {
    "observed": 1.444,
    "ppp": 0.78
  },
  "category_usage_tv_distance": {
    "mean_replicate_tv": 0.032,
    "note": "TV distance between replicate and observed category frequencies; large values mean the model uses the scale differently than people did"
  },
  "n_sims": 200,
  "passed": true
}

```

### Real-data LOCO (runs/pilot/loco.json)

```json
{
  "per_family": {
    "32.1.martin": {
      "n_items": 12,
      "rmse": 0.796,
      "mean_signed_error": -0.447,
      "spearman": 0.874,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.006,
        "ess_bulk_min": 942,
        "passed": true
      }
    },
    "32.3.Culicover": {
      "n_items": 12,
      "rmse": 0.878,
      "mean_signed_error": -0.046,
      "spearman": 0.51,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0036,
        "ess_bulk_min": 787,
        "passed": true
      }
    },
    "32.3.fanselow": {
      "n_items": 12,
      "rmse": 1.155,
      "mean_signed_error": 0.538,
      "spearman": 0.937,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0053,
        "ess_bulk_min": 823,
        "passed": true
      }
    },
    "33.4.neeleman": {
      "n_items": 12,
      "rmse": 0.989,
      "mean_signed_error": 0.37,
      "spearman": 0.783,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0078,
        "ess_bulk_min": 809,
        "passed": true
      }
    },
    "34.1.fox": {
      "n_items": 12,
      "rmse": 1.468,
      "mean_signed_error": -0.181,
      "spearman": -0.084,
      "coverage90": 0.833,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 2,
        "divergence_rate": 0.0003,
        "rhat_max": 1.0039,
        "ess_bulk_min": 930,
        "passed": true
      }
    },
    "34.1.phillips": {
      "n_items": 12,
      "rmse": 0.835,
      "mean_signed_error": -0.121,
      "spearman": 0.769,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0037,
        "ess_bulk_min": 880,
        "passed": true
      }
    },
    "34.3.heycock": {
      "n_items": 12,
      "rmse": 1.295,
      "mean_signed_error": -0.177,
      "spearman": 0.664,
      "coverage90": 0.833,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 7,
        "divergence_rate": 0.0009,
        "rhat_max": 1.0038,
        "ess_bulk_min": 1171,
        "passed": true
      }
    },
    "34.4.boskovic": {
      "n_items": 12,
      "rmse": 0.784,
      "mean_signed_error": 0.382,
      "spearman": 0.832,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 2,
        "divergence_rate": 0.0003,
        "rhat_max": 1.0065,
        "ess_bulk_min": 1120,
        "passed": true
      }
    },
    "41.3.Landau": {
      "n_items": 12,
      "rmse": 1.177,
      "mean_signed_error": 0.166,
      "spearman": 0.629,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 5,
        "divergence_rate": 0.0006,
        "rhat_max": 1.0065,
        "ess_bulk_min": 839,
        "passed": true
      }
    },
    "41.3.Vicente": {
      "n_items": 12,
      "rmse": 0.655,
      "mean_signed_error": -0.078,
      "spearman": 0.832,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0056,
        "ess_bulk_min": 959,
        "passed": true
      }
    }
  },
  "n_families": 10,
  "mean_rmse": 1.003,
  "mean_spearman": 0.675,
  "mean_coverage90": 0.925,
  "all_diagnostics_passed": true
}

```

### Warrant certificate (runs/pilot/warrant.yaml)

```yaml
generated: '2026-08-27T14:32:26'
run_dir: runs/pilot
estimand:
  target: mean 7-point Likert acceptability rating per decontextualized sentence
  population: the participant population of Sprouse, Schütze & Almeida (2013), Lingua
    (Linguistic Inquiry 2001-2010 random-sample judgment study); no demographic claims
    beyond their sampling are made or tested
  response_scale: 1-7 Likert, decontextualized written sentences
domain:
  construction_families:
  - 32.1.martin
  - 32.3.Culicover
  - 32.3.fanselow
  - 33.4.neeleman
  - 34.1.fox
  - 34.1.phillips
  - 34.3.heycock
  - 34.4.boskovic
  - 41.3.Landau
  - 41.3.Vicente
  n_items: 120
  item_source: Sprouse-Schütze-Almeida 2013 LI materials, pilot subset
  language: English (US)
  register: constructed linguistic example sentences
  families: 10 LI source papers, 12 items each (6 starred / 6 good)
instruments:
- cell: pythia-160m/slor
  reliability_median: 0.338
  reliability_90:
  - 0.106
  - 0.534
  bias_alpha_median: 0.012
- cell: qwen3:8b/prompt_scalar
  reliability_median: 0.54
  reliability_90:
  - 0.424
  - 0.629
  bias_alpha_median: -0.011
evidence:
  diagnostics:
    divergences: 0
    divergence_rate: 0.0
    rhat_max: 1.0094
    ess_bulk_min: 501
    passed: true
  fake_data_recovery: not produced
  loco_transfer:
    per_family:
      32.1.martin:
        n_items: 12
        rmse: 0.796
        mean_signed_error: -0.447
        spearman: 0.874
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.006
          ess_bulk_min: 942
          passed: true
      32.3.Culicover:
        n_items: 12
        rmse: 0.878
        mean_signed_error: -0.046
        spearman: 0.51
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0036
          ess_bulk_min: 787
          passed: true
      32.3.fanselow:
        n_items: 12
        rmse: 1.155
        mean_signed_error: 0.538
        spearman: 0.937
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0053
          ess_bulk_min: 823
          passed: true
      33.4.neeleman:
        n_items: 12
        rmse: 0.989
        mean_signed_error: 0.37
        spearman: 0.783
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0078
          ess_bulk_min: 809
          passed: true
      34.1.fox:
        n_items: 12
        rmse: 1.468
        mean_signed_error: -0.181
        spearman: -0.084
        coverage90: 0.833
        diagnostics_passed: true
        diagnostics:
          divergences: 2
          divergence_rate: 0.0003
          rhat_max: 1.0039
          ess_bulk_min: 930
          passed: true
      34.1.phillips:
        n_items: 12
        rmse: 0.835
        mean_signed_error: -0.121
        spearman: 0.769
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0037
          ess_bulk_min: 880
          passed: true
      34.3.heycock:
        n_items: 12
        rmse: 1.295
        mean_signed_error: -0.177
        spearman: 0.664
        coverage90: 0.833
        diagnostics_passed: true
        diagnostics:
          divergences: 7
          divergence_rate: 0.0009
          rhat_max: 1.0038
          ess_bulk_min: 1171
          passed: true
      34.4.boskovic:
        n_items: 12
        rmse: 0.784
        mean_signed_error: 0.382
        spearman: 0.832
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 2
          divergence_rate: 0.0003
          rhat_max: 1.0065
          ess_bulk_min: 1120
          passed: true
      41.3.Landau:
        n_items: 12
        rmse: 1.177
        mean_signed_error: 0.166
        spearman: 0.629
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 5
          divergence_rate: 0.0006
          rhat_max: 1.0065
          ess_bulk_min: 839
          passed: true
      41.3.Vicente:
        n_items: 12
        rmse: 0.655
        mean_signed_error: -0.078
        spearman: 0.832
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0056
          ess_bulk_min: 959
          passed: true
    n_families: 10
    mean_rmse: 1.003
    mean_spearman: 0.675
    mean_coverage90: 0.925
    all_diagnostics_passed: true
  ppc:
    family_item_mean_spread:
      observed: 1.528
      ppp: 0.86
    within_item_disagreement_sd:
      observed: 1.444
      ppp: 0.78
    category_usage_tv_distance:
      mean_replicate_tv: 0.032
      note: TV distance between replicate and observed category frequencies; large
        values mean the model uses the scale differently than people did
    n_sims: 200
    passed: true
  multiverse_spread:
    mean_per_item_sd_across_cells: 0.633
    n_items: 120
    n_cells: 2
  prompt_invariance: not identifiable from cell names
  human_split_half: 0.857 (mean of 200 random participant splits, 120 items; Spearman-Brown
    corrected 0.923; computed from data/pilot_human.csv)
  contamination:
    status: suspect
    reason: Sprouse-Schutze-Almeida LI materials are drawn from Linguistic Inquiry
      articles (2001-2010) and the dataset has been publicly posted since 2013; presence
      in LLM pretraining corpora cannot be excluded for either instrument.
  instrument_nonresponse:
    qwen3:8b/prompt_binary: 0.0
    qwen3:8b/prompt_scalar: 0.0
  flagged_cells: none
  generalization_axes:
    tested:
    - construction_family (LOCO-CV, purposive family sample; descriptive over the
      families tested)
    untested:
    - population
    - register
    - language
    - item_source
    - time
    - model_version (beyond drift sentinels)
licensed_claims:
  screening: diagnostics passed and max cell reliability median 0.54 > 0.5
  ranking: screening granted and LOCO mean held-out rank correlation 0.68 > 0.6
refused_claims:
  aggregate_estimation: 'contamination cap: item source not explicitly assessed clean
    (status: suspect); contamination inflates exactly the statistics this tier rests
    on, and LOCO rewards it'
  effect_reproduction: 'not yet tested: requires matched experimental contrasts'
  distributional_claims: no participant-level validation of variance structure
  population_transfer: 'refused: the v1 ladder contains no population-transfer test;
    the estimand population defaults to the criterion sample''s own'
  individual_simulation: 'refused permanently: an item-level instrument licenses no
    individual-level human simulation'
  mechanism_claims: 'refused permanently: the model estimates a linking function,
    not a mechanism'
residual_risks:
- 'shared pretraining bias: local instruments share web-scale training data and can
  share construction-specific error; a tight multiverse fan does not rule this out'
- conditional reliability is fit- and item-set-specific; it goes stale when the item
  pool changes

```

---

## PRIOR REVIEW (glm-5.3-flash) AND TRIAGE

# Adversarial review: Acceptometer design spec

Top-line: the two most consequential problems are (A) the model assumes the linking function is invariant across construction families while selling cross-family projection as the product, and (B) contamination is recorded as metadata but has no statistical consequence, so the validation ladder actively rewards it. Details below, organized by your four buckets.

## 1. Identification problems in the Stan spec (§ The statistical model)

**1.1 The latent scale is identified by the prior, not the data — and the product lives exactly where the prior dominates.** [major]
The human arm's ordered-logistic likelihood identifies theta only up to affine transformation; the "soft sum-to-zero, unit-scale" standardization resolves this *by prior*. That is legitimate, but note where it bites: for items with no human ratings ("The instrument use"), theta is identified by the instrument likelihood plus that prior. If a cell's sigma_m gets small (see 1.6 and 2.2), theta for unrated items collapses toward a rescaled LLM score with arbitrarily tight intervals. There is no structural floor on sigma_m, so "honestly widened uncertainty" is not a property of the model — it is a hope. Add an explicit noise floor or a contamination-aware downweighting.

**1.2 The reliability formula is algebraically wrong as stated.** [major]
`beta_m^2 var(theta) / (beta_m^2 var(theta) + sigma_m^2)` drops the `gamma_m' x_i` terms. With `s_im = alpha_m + beta_m·theta_i + gamma_m'·x_i + eps`, Var(s_im) includes `gamma'Σ_x γ + 2·beta·cov(theta, x)`. Length and frequency are almost certainly correlated with acceptability, so the omitted covariance term is not negligible. If you intend a *conditional* reliability (given x), say so and relabel it — it excludes systematic bias variance, which conflates reliability with validity in exactly the way the certificate shouldn't.

**1.3 Under the unit-scale constraint, `var(theta) ≈ 1`, making the formula mostly a beta/sigma ratio.** [moderate]
The soft standardization pins var(theta) near 1, so including `var(theta)` in the formula is decorative. Consequence: the reported "posterior reliability" is a prior-scaled quantity, internal to this fit. The certificate's plan to benchmark it against "human split-half reliability" (§ The warrant certificate, evidence) compares a latent-scale ratio to a raw-score quantity — incoherent without a common scale (e.g., Spearman-Brown'd, same latent metric).

**1.4 Binary cells have no sigma_m, so the replicate logic is incoherent.** [major]
§ Elicitation multiverse grid says repeats "enter as replicate observations of s_im, informing sigma_m only." For `prompt_binary` and `answer_token_prob` there is no sigma_m in the likelihood — Bernoulli noise is fixed at 1 by construction. What do binary repeats inform? Nothing in the stated model. Either add an overdispersion parameter or restrict the replicate logic to continuous cells. Related: at temperature 0, binary outputs are (near-)deterministic, so the Bernoulli likelihood's aleatoric noise is pure model fiction for those cells; and `prompt_scalar` (integers 1–7, bounded) and `answer_token_prob` (∈ [0,1]) modeled as *normal* outcomes is the wrong likelihood — use ordered-logistic with its own cutpoints and a logit transform respectively.

**1.5 The multiverse fan does not exist in the joint model.** [moderate]
"Multiverse spread: per item, the spread of theta posteriors across elicitation cells" (§ Validation ladder, 4) is undefined: the joint model has *one* theta per item. Computing a fan requires per-cell (or leave-one-cell-out) refits, which is a different, undocumented pipeline whose thetas are no longer pooled. As specified, this artifact cannot be produced from the stated model.

**1.6 Weakly separated variance components.** [minor]
With Sprouse-scale data (~8–12 items per paradigm, limited number of paradigms), `tau_item` vs `tau_constr` are weakly separately identified; the priors do real work, and since var(theta) enters reliability, that work propagates into certificate fields. Similarly, `normal(0,2)` on ordered cutpoints with standardized theta puts awkward prior mass on extreme category configurations; fine, but it's a choice that should appear in a prior-sensitivity check (which the ladder omits entirely).

**1.7 `gamma_m` vs `beta_m` collinearity.** [moderate]
If x_i (length, frequency) correlates with theta across items, beta_m and gamma_m are individually weakly identified even when predictions are fine. Reported per-instrument "bias summaries" and reliability will be unstable; the certificate presents them as instrument properties when they are fit-specific and item-set-specific.

## 2. Flaws in the validity/warrant logic

**2.1 The linking function is assumed family-invariant; the product assumes it.** [major — the central flaw]
There is no family-level random effect in the instrument arm (no beta_mc, no family-specific linking deviations). Instrument residuals are i.i.d. by construction, but LLM scoring errors cluster by phenomenon (islands, binding, center-embedding systematically misjudged). For unrated items *within fitted families* this is tolerable; for items in *new* families — the regime the tool is explicitly sold for ("new constructions," § What this is) — the theta posterior excludes between-family linking uncertainty entirely. LOCO measures average transfer but cannot inject that uncertainty back into the intervals. "Honestly widened uncertainty" is therefore structurally false. Fix: random coefficients of the linking by family, or inflate unrated-item intervals by the between-family linking variance estimated from LOCO.

**2.2 Contamination is recorded, never used.** [major]
BLiMP minimal pairs are public and post-training-cutoff-verifiable; published paradigms are in Pythia/qwen/gemma pretraining corpora with high probability. Contamination produces exactly the degenerate attractor from 1.1: sigma_m shrinks, beta inflates, reliability looks great, unrated-item thetas become confident rescaled LLM scores. Worse, **LOCO rewards contamination**: the held-out family is published, hence likely in training data, so transfer looks excellent *because of* the contaminant. The certificate lists "contamination assessment" (§ The warrant certificate) but there is no rule forcing refusal of projection/effect-size tiers for contaminated cells. A recorded caveat is not a warrant constraint.

**2.3 LOCO is one axis of generalization dressed as *the* projectibility test.** [major]
LOCO tests transfer across construction families within one item source, one language, one register, one elicitation protocol, one time point. The intro claims the warrant covers "new constructions, new items, new model versions." New items within validated families, new item sources, and new populations are untested by anything in the ladder. Also: construction families are a purposive, historically-theory-driven sample, not an exchangeable draw from a superpopulation — so `tau_constr` and LOCO coverage are descriptive over the sampled families, not inferential over "constructions" in general. The certificate's language should say this; currently "LOCO-CV … This is the projectibility test" (§ Validation ladder, 4) overstates by exactly the move the document's own philosophy forbids.

**2.4 Estimand population vs criterion sample is an unbridged gap.** [major]
The estimand example says "US-English adult non-linguists" (§ The warrant certificate). The criterion candidates (Sprouse/Schütze/Almeida; MORCELA) have their own sampling frames. Nothing in the ladder tests participant-population transfer or measurement invariance across rater groups; the model assumes exchangeable participants via `u_p ~ normal(0, sigma_p)` only. The certificate records both populations but the warrant logic contains no step connecting them — every downstream claim silently rests on an untested population-transfer assumption. Either declare the estimand to be the criterion population, or add a generalization test.

**2.5 No decision thresholds mapping evidence to tiers.** [moderate]
`licensed_claims` are "explicitly granted or refused with the reason," but no pre-registered quantitative criteria exist (e.g., screening licensed if LOCO RMSE < X·SD(human means), coverage within [a,b]). Without thresholds, "machine-readable" licensing is judgment wearing a YAML costume — precisely the discretion the certificate exists to remove.

**2.6 Tight-fan logic has a blind spot.** [moderate]
"Tight fan = measurement" (§ Validation ladder, 4) fails under shared-mode bias: all local instruments are web-crawl-trained and can share nonlinear, construction-specific errors that the linear `gamma'x` terms don't absorb. All cells agree; theta is wrong; LOCO catches it only if the bias is family-specific *and* the held-out family exhibits it. The certificate should carry this as a permanent residual-risk caveat, not treat fan width as dispositive.

**2.7 Drift monitoring underspecified.** [moderate]
(§ Validation ladder, 5.) (a) Sentinels bound drift on sentinels; drift can be construction-non-uniform. (b) For a new model version, alpha/beta/sigma for the new epoch are estimated from sentinel re-scores only — a small, biased item set — yet the certificate's drift bounds and any re-licensed claims inherit that narrow basis; nothing says new-instrument licenses are restricted to sentinel-covered domain. (c) "Drift bounds" have no statistical definition (posterior of what function, with what uncertainty, propagated from sentinel theta uncertainty?). (d) The design assumes version identity is observable; silent weight updates under a stable name break the epoch covariate (moot for v1 local models, but the architecture is being sold for API use).

## 3. Failure modes the validation ladder misses

**3.1 No PPC gate at the real-fit stage.** [major]
Step 3 is "Human criterion data + real elicitation runs" with no stated acceptance criteria. The model section touts PPCs on "variance and disagreement structure," but the ladder has no step where PPC failure blocks licensing. Since the human arm (theta + additive u_p, single kappa, no item discrimination, no rater cutpoint heterogeneity) *cannot represent* family-varying disagreement or differential scale use (raters using only 3–5 aren't shifted versions of raters using 1–7), those PPCs will fail — and the misfit will flow silently into licensed claims. Either add rater cutpoints/item discrimination or pre-commit to which PPC failures refuse which tiers.

**3.2 Instrument nonresponse.** [moderate]
Prompted chat cells produce refusals and parse failures. Silent filtering is selection on an unmodeled mechanism, plausibly correlated with theta (refusals concentrate on odd content). Nothing in grid.py's described responsibilities or the model handles missingness.

**3.3 Fake-data recovery validates computation, not the model.** [moderate]
"Simulate the full generative process… The pipeline is untrusted until it passes" is correct as far as it goes, but simulate.py "mirroring the Stan model" can never detect model inadequacy — and real data has features the simulation won't (deterministic temp-0 outputs, bounded scales, ceiling effects, contamination). Passing the ladder is necessary, not licensing. The text flirts with the stronger reading; make the scope explicit.

**3.4 SBC-lite misses the product quantities.** [moderate]
Ranks are checked for beta_m, sigma_m, tau_constr — internals. The things being sold are the theta posterior for unrated items and the derived reliability ratio; neither is in the SBC set, and neither is in step 1's "parameter recovery" unless you deliberately hold out items' human data and check unrated-item coverage. Also, ~50 fits gives essentially no tail resolution on rank histograms; and there is no prior-sensitivity stage anywhere, despite priors carrying identification (see 1.1).

**3.5 Adaptive budget feedback.** [minor]
design.py ranks by posterior sd(theta). If selection uses only instrument data already in the model, ignorability holds; if it touches human pilot data outside the likelihood, it doesn't. State which.

**3.6 LOCO evaluation target underspecified.** [moderate]
"Predict its human item means… record transfer RMSE, interval coverage" — coverage of which interval: latent theta, predictive mean rating, or the observed mean? The observed mean carries ordinal sampling noise that must be included or coverage is miscalibrated by construction. Also, evaluating means while the certificate sells a "distributional claims" tier (§ The warrant certificate) leaves that tier untested by the only generalization test.

## 4. Statistically wrong or overclaimed (specific lines)

- **Reliability formula** (§ The statistical model): drops gamma/cov(x,θ) terms — wrong unless x ⊥ θ (see 1.2); inapplicable to binary cells (1.4).
- **"repeats … informing sigma_m only"** (§ Elicitation multiverse grid): incoherent for binary cells; and the temp-0 score enters the fit while sigma_m is estimated from default-temperature repeats — a noise estimate from a different sampling regime than the score it's attached to.
- **"Tight fan = measurement"** (§ Validation ladder, 4): false under shared pretraining bias (2.6).
- **"This is the projectibility test"** (§ Validation ladder, 4): single-axis test over a purposive family sample; cannot support the intro's three-way generalization claim (2.3).
- **"honestly widened uncertainty"** (§ The instrument use): not structurally guaranteed; degenerate under contamination (1.1, 2.2).
- **"linking uncertainty propagates into every downstream claim"** (§ The statistical model, MORCELA paragraph): only if every downstream artifact is computed from the full joint posterior; certificate summaries and LOCO RMSEs are not.
- **PPC claim on disagreement structure** (§ Human arm paragraph): the specified model cannot express item- or family-varying disagreement; the check will fail or the claim overstates capability (3.1).
- **"posterior reliability" as a per-instrument certificate field** (§ The warrant certificate): it is item-set-, prior-, and fit-specific; stale the moment the item pool changes.

Most load-bearing fixes, in order: (1) family-level random coefficients or variance inflation in the linking arm; (2) contamination rules with tier consequences plus a sigma_m floor; (3) a PPC/refusal gate at ladder step 3 tied to tiers; (4) pre-registered evidence→tier thresholds; (5) correct likelihoods for binary/scalar/probability cells with a coherent replicate story.


### Triage (from DECISIONS.md)

2026-08-27 — ocx (glm-5.3-flash) adversarial review of DESIGN.md accepted in
large part; model upgraded before any real-data fit. Changes: (a) family-level
intercept/slope deviations in the continuous instrument arm (LLM error
clusters by phenomenon; new-family predictions now inherit between-family
linking uncertainty from the priors); (b) sigma_s floored at 0.05 so no cell
can pose as a noiseless oracle; (c) reliability relabeled *conditional*
reliability, excluding bias terms by construction; (d) prompted cells sample
at ONE temperature (default), no temp-0/stochastic mixing in one likelihood;
(e) LOCO coverage target now the observed sample mean including its ordinal
sampling noise; (f) contamination caps warrant tiers (aggregate estimation and
above refused for public-benchmark-suspect item sources) because LOCO rewards
contamination; (g) LOCO demoted from "the projectibility test" to one axis,
with untested axes listed in every certificate; (h) PPC gate added at the
real-fit stage with pre-committed tier consequences; (i) shared pretraining
bias recorded as a permanent residual risk (a tight multiverse fan is not
dispositive). Effect visible immediately in simulation: LOCO out-of-family
coverage rose 0.84 -> 0.97 while held-out Spearman fell 0.79 -> 0.58; the
earlier precision was the model borrowing family-specific linking it could
not have known. Review findings NOT adopted: scalar-cells-as-ordered-logistic
(documented v1 misspecification instead), prior-sensitivity stage (v2),
held-out-item SBC (v2), rater-specific cutpoints (v2).
