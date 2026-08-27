# Acceptometer external validation packet (2026-08-27, rev 4: post-review-2 model)

## Review brief

You are adversarially validating a Bayesian measurement tool that treats LLMs
as biased instruments for human acceptability judgments. Everything you need
is in this file. Assume nothing is correct until you have checked it. Report
findings as a numbered list with severity, exact location, and the concrete
failure each would cause. Do not review prose style.

Two prior adversarial reviews (attached with full triage) have already been
adopted: family-varying linking with instrument-by-item error; freed latent
scale (link-anchored); new/held-out families excluded from the training
sum-to-zero vector with independent predictive effects; new-family predictive
reliability as the screening gate; a literally-enforced ladder
(diagnostics + recovery + SBC + two-mode per-family PPC + complete,
diagnostics-clean, hash-bound LOCO); contamination caps on ranking and above;
tie-aware pooled Spearman with family-cluster bootstrap; same-raters LOCO
target with training-only standardization; SLOR unit consistency and
checkpoint provenance.

Priority questions:

1. SPEC-VS-CODE DRIFT: DESIGN.md vs acceptometer.stan and warrant.py, line by
   line where it matters (identification list, reliability quantities, ladder
   enforcement, run binding).
2. NEW-FAMILY MECHANISM: is the is_new construction (transformed-data index
   mapping, sum_to_zero_vector over training families only, independent
   normal(0, tau_constr) for new) correct and identified? Does anything else
   still leak training-set information to a held-out family?
3. PREDICTIVE RELIABILITY: reliability_new draws a fresh slope deviation per
   posterior draw. Is that the right predictive quantity, and is gating on
   (median > .5, q10 > .35) coherent given it mixes posterior and predictive
   uncertainty?
4. LADDER ENFORCEMENT: find a path through warrant.py that grants a tier on
   missing, failed, stale, or unbound evidence. The binding uses SHA-256 of
   posterior.nc and input files via run.json; find the hole.
5. LOCO: same-raters target, training-only standardization, tie-aware pooled
   Spearman, family-cluster bootstrap. What still biases the transfer
   estimate or its lower bound?
6. PPC: two participant modes, per-family minima gates, category-usage ppp
   against the replicate reference, instrument residual flags. What misfit
   slips past all of it?
7. SBC: prior pushforwards (sum-zero projection, sorted normals, truncated
   sigma, gamma(2,1)), diagnostics-aware failure cap at 20%, R=100. Any
   remaining incoherence between the SBC generative process and the model?
8. EVIDENCE CONSISTENCY: verify every number in PILOT-REPORT.md against the
   attached reports.
9. WHAT BOTH PRIOR REVIEWERS MISSED.



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

Identification, stated precisely because each piece comes from a different
source:

- **Scale**: fixed by convention, via the human arm's discrimination (1) and
  the ordered-logistic error's fixed variance. tau_item, tau_constr, and
  sigma_u are therefore all free and data-identified relative to that anchor
  (tau_item carries a zero-avoiding gamma(2,1) prior: a family of identical
  items is a priori false for designed judgment studies, and tau_item -> 0 is
  where a spurious reflection mode escapes the human arm's sign anchoring).
- **Family-effect location**: training families sit in a `sum_to_zero_vector`.
  A family marked NEW (LOCO holdouts, genuinely new families) is excluded
  from that vector and gets an independent normal(0, tau_constr) effect: a
  held-out family inside the constrained vector would inherit information
  through the finite-set centering.
- **Residual overall location**: conventional/prior-anchored. Joint shifts of
  theta, kappa, and instrument intercepts are likelihood-equivalent until the
  z_item and intercept priors intervene; absolute location is therefore not a
  reportable quantity.
- **Variance-component separation** (tau_item vs tau_constr vs sigma_u) is
  data-driven only insofar as the participant-by-item design is crossed;
  sparse or nested rating designs push it back onto the priors.

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

**Reliability quantities** (all conditional on the nuisance covariates, all
fit- and item-set-specific, with noise = omega_m^2 + sigma_m^2 for replicated
cells):

- `reliability[m]` — the **global-slope signal ratio**, using beta_m alone.
  NOT a warrant quantity: a family slope deviation changes how much
  theta-signal that family's scores carry, so a cell can look reliable
  globally while carrying no signal in a particular family.
- `reliability_family[m, c]` — per-family, using the realized slope
  `beta_m + b_dev_mc`.
- `reliability_new[m]` — **new-family predictive**: each posterior draw
  samples a fresh slope deviation, so this posterior is the predictive
  distribution for a family the instrument has never seen. This is the
  screening-gate quantity, because screening claims project.

MORCELA is the special case of one continuous method, no family structure,
fixed covariates; here the linking is Bayesian, multilevel, and jointly
estimated with the human arm, so linking uncertainty propagates into
everything computed from the joint posterior (certificate summaries derived
outside it, like LOCO RMSEs, are labeled as point summaries).

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
  evidence supports, each explicitly granted or refused with the reason and
  the pre-registered numeric threshold it met or missed. **The ladder is
  enforced literally:** screening requires diagnostics + fake-data recovery +
  SBC all present and passed, plus an unflagged cell whose new-family
  predictive reliability clears (median > 0.5, q10 > 0.35); ranking
  additionally requires LOCO present, hash-bound to this run, covering every
  family, all fold diagnostics passed, pooled tie-aware Spearman > 0.6 with
  family-cluster bootstrap lower-90 > 0.5; aggregate estimation additionally
  requires the PPC (both participant modes) passed and LOCO coverage in
  [0.75, 0.98]. Missing, failed, or unbound evidence refuses; nothing grants
  by default. Thresholds are pre-registered decision defaults, not
  estimand-specific loss analyses, and the certificate says so.
  **Contamination caps tiers:** a public-benchmark-suspect (or unassessed)
  item source refuses ranking and above regardless of the numbers, because
  contamination inflates exactly the statistics those tiers rest on and LOCO
  rewards it (the held-out family is published too).
- **Run binding**: every fit writes `run.json` (run id, posterior SHA-256,
  Stan-file SHA, code commit, input hashes); PPC and LOCO stamp what they
  read; the warrant refuses evidence whose stamps do not match, so a stale or
  copied report cannot license a different posterior.
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
- **No prior-sensitivity stage in the ladder yet.** The latent scale itself is
  fixed by the link convention, but absolute location and the variance-component
  split remain prior-sensitive (see the identification list above); a
  sensitivity sweep (halve/double the key prior scales, compare certificate
  numbers) is v2 work and the certificate says so.
- **Divergence tolerance is 0.5%, not zero.** Stan's strict reading is that
  any divergence is evidence of unexplored posterior; the gate trades that
  against realistic run costs and treats a sub-0.5% rate with clean R-hat/ESS
  as acceptable. The rate is always reported, so a stricter reader can refuse.
- **LOCO's evaluation target is same-participants-new-items** (raters' posterior
  effects are used where known). Fresh-population transfer is a different
  target the v1 ladder does not test; the marginal-mode PPC is the closest
  check it has.
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
  // 1 marks a family treated as NEW: excluded from the sum-to-zero training
  // vector and given an independent normal(0, tau_constr) effect. A held-out
  // family inside the constrained vector would inherit information through
  // the finite-set centering (its mean is the negative sum of the others),
  // which is not what a genuinely new construction family receives.
  array[N_constr] int<lower=0, upper=1> is_new;
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
transformed data {
  int N_old = 0;
  int N_newf = 0;
  array[N_constr] int old_ix = rep_array(0, N_constr);
  array[N_constr] int new_ix = rep_array(0, N_constr);
  for (c in 1:N_constr) {
    if (is_new[c] == 1) { N_newf += 1; new_ix[c] = N_newf; }
    else { N_old += 1; old_ix[c] = N_old; }
  }
}
parameters {
  // latent acceptability
  sum_to_zero_vector[N_old] mu_c_raw;
  vector[N_newf] mu_new_raw;
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
  vector[N_constr] mu_c;
  for (c in 1:N_constr)
    mu_c[c] = tau_constr * (is_new[c] == 1 ? mu_new_raw[new_ix[c]]
                                           : mu_c_raw[old_ix[c]]);
  vector[N_item] theta = mu_c[constr] + tau_item * z_item;
  vector[N_part] u = sigma_u * u_raw;
  matrix[M_c, N_constr] a_dev = diag_pre_multiply(tau_a, a_dev_raw);
  matrix[M_c, N_constr] b_dev = diag_pre_multiply(tau_b, b_dev_raw);
}
model {
  // priors (weakly informative throughout; no flat priors)
  mu_c_raw ~ normal(0, 1);
  mu_new_raw ~ std_normal();
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
  // global-slope signal ratio: uses the population slope only. NOT a warrant
  // quantity on its own, because a family's slope deviation changes how much
  // theta-signal that family's scores carry.
  vector[M_c] reliability;
  // family-specific reliability, using the family's realized slope
  matrix[M_c, N_constr] reliability_family;
  // new-family predictive reliability: the projectible quantity. Each draw
  // samples a fresh slope deviation from its posterior scale, so the
  // posterior of reliability_new is the predictive distribution for a family
  // the instrument has never seen.
  vector[M_c] reliability_new;
  for (m in 1:M_c) {
    real noise = (has_reps[m] == 1 ? square(omega[m]) : 0) + square(sigma_s[m]);
    reliability[m] = square(beta[m]) * v_theta
                     / (square(beta[m]) * v_theta + noise);
    for (c in 1:N_constr) {
      real bfc = beta[m] + b_dev[m, c];
      reliability_family[m, c] = square(bfc) * v_theta
                                 / (square(bfc) * v_theta + noise);
    }
    real b_new = beta[m] + normal_rng(0, tau_b[m]);
    reliability_new[m] = square(b_new) * v_theta
                         / (square(b_new) * v_theta + noise);
  }
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
    tau_item: float
    tau_a: float
    tau_b: float
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
        is_new=[0] * n_constr,
        N_c=len(s), M_c=M_c, item_c=item_c.tolist(), cell_c=cell_c.tolist(),
        s=s.tolist(), has_reps=[1] * M_c,
        N_b=len(z), M_b=M_b, item_b=item_b.tolist(), cell_b=cell_b.tolist(),
        z=z.tolist(),
    )
    truth = SimTruth(theta, mu_c, tau_constr, tau_item, tau_a_true, tau_b_true,
                     kappa, sigma_u, alpha, beta, gamma, sigma_s, omega_true,
                     a_b, b_b, g_b)
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

    for name in ("tau_item", "tau_a", "tau_b"):
        draws = post[name].stack(d=("chain", "draw")).values
        truths = np.atleast_1d(getattr(truth, name))
        draws2 = draws if draws.ndim == 2 else draws[None, :]
        ok = True
        for j, tv in enumerate(np.broadcast_to(truths, (draws2.shape[0],))):
            lo, hi = np.percentile(draws2[j], [2.5, 97.5])
            ok = ok and (lo <= tv <= hi)
        report[f"{name}_within_95ci"] = bool(ok)
        report[f"{name}_post_median"] = [round(float(np.median(d)), 3) for d in draws2]

    rel = post["reliability"].stack(d=("chain", "draw")).values.mean(axis=1)
    v_theta = truth.theta.var()
    rel_truth = truth.beta**2 * v_theta / (
        truth.beta**2 * v_theta + truth.omega**2 + truth.sigma_s**2)
    rel_err = float(np.max(np.abs(rel - rel_truth)))
    report["reliability_max_abs_err"] = round(rel_err, 3)
    report["reliability_truth"] = [round(v, 3) for v in rel_truth]

    report["passed"] = bool(
        0.80 <= cover <= 0.98 and beta_ok and rel_err <= 0.15
        and report["tau_item_within_95ci"] and report["tau_a_within_95ci"]
        and report["tau_b_within_95ci"])
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
        is_new=[0] * n_constr,
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
    n_diag_failed = 0

    for r in range(R):
        pr = _draw_prior(rng, n_constr, K, M_c, M_b, P)
        data = _simulate_given(pr, rng, n_constr, items_per_constr, n_part,
                               ratings_per_item, K, M_c, M_b, P)
        try:
            fit, idata = fit_model(data, seed=seed + r, iter_warmup=500,
                                   iter_sampling=500, chains=2)
        except Exception:
            n_failed += 1
            continue
        # per-fit sanity check scaled to SBC's small fits: the production
        # gate's ESS>400 is unreachable with 2x300 draws by construction.
        # Ranks need R-hat convergence, few divergences, and enough effective
        # draws to support n_thin thinned ranks.
        import numpy as _np
        div = int(_np.sum(fit.method_variables()["divergent__"]))
        ndr = int(_np.prod(fit.method_variables()["divergent__"].shape))
        import arviz as _az
        summ = _az.summary(idata, var_names=[v for v in ("beta", "sigma_s",
                           "tau_constr", "tau_item", "sigma_u")
                           if v in idata.posterior])
        if not (div / ndr < 0.01 and float(summ["r_hat"].max()) < 1.02
                and float(summ["ess_bulk"].min()) > 100):
            n_diag_failed += 1
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

    report = {"R": R, "n_failed_fits": n_failed,
              "n_diag_failed": n_diag_failed, "n_thin": n_thin, "params": {}}
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
    # a pipeline whose small-data fits routinely fail or misbehave is not
    # validated by the survivors' rank uniformity
    if (n_failed + n_diag_failed) > 0.2 * R:
        passed = False
        report["failure_note"] = "more than 20% of replications failed or failed diagnostics"
    report["passed"] = passed
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report

```


---

## FILE: src/acceptometer/model/loco.py

```
"""Leave-one-construction-family-out: the family-axis projectibility test.

For each construction family c, refit the model with (a) family c's HUMAN data
removed and (b) family c marked NEW, so its family effect and linking
deviations come from their predictive distributions rather than the training
sum-to-zero vector (a held-out family inside that vector inherits information
through the finite-set centering). Instrument scores for family c's items are
retained: they are the transfer mechanism under test.

Evaluation target (stated, not implied): SAME participants, NEW items. The
observed criterion is the mean rating given by the actual raters of each
held-out item, so the predictive uses those raters' posterior effects where
they are known from training items, and fresh draws otherwise.

Standardization is training-only: per-cell constants are computed from
training-family measurements and applied unchanged to held-out scores, so a
held-family location or scale shift is confronted, not absorbed.

Rank transfer is reported tie-aware, pooled across all held-out items, with a
family-cluster bootstrap lower bound; per-family values are descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import build_stan_data, fit_model, diagnostics_gate


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman: average ranks, then Pearson."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _predict_observed_means_same_raters(idata, maps: dict, held: pd.DataFrame,
                                        K: int, seed: int = 0
                                        ) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive of each held-out item's observed mean rating for
    its ACTUAL raters: raters known from training items use their posterior
    u draws; raters absent from training draw fresh u from sigma_u. Includes
    ordinal sampling noise, so coverage against the observed sample mean is
    calibrated for the same-participants-new-items target."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values
    u_post = post["u"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}

    items = held.groupby("item_id")
    ids, means, los, his = [], [], [], []
    for iid, grp in items:
        th = theta[item_pos[iid]]                        # (L,)
        raters = list(grp["participant_id"])
        u_rows = []
        for r in raters:
            if r in part_pos:
                u_rows.append(u_post[part_pos[r]])
            else:
                u_rows.append(rng.normal(0.0, 1.0, L) * sigma_u)
        u = np.stack(u_rows)                             # (n_j, L)
        eta = th[None, :] + u
        cum = 1.0 / (1.0 + np.exp(-(kappa.T[None, :, :] - eta[..., None])))
        draw = rng.uniform(size=(len(raters), L))
        y = 1 + (draw[..., None] > cum).sum(axis=-1)     # (n_j, L)
        sm = y.mean(axis=0)
        ids.append(iid)
        means.append(sm.mean())
        lo, hi = np.percentile(sm, [5, 95])
        los.append(lo)
        his.append(hi)
    return pd.Index(ids), np.array(means), np.array(los), np.array(his)


def _standardize_training_only(cont: pd.DataFrame, train_items: set) -> tuple[pd.DataFrame, dict]:
    out = cont.copy()
    constants = {}
    for cell, grp in cont.groupby("cell_id"):
        tr = grp[grp["item_id"].isin(train_items)]["value"]
        mu, sd = float(tr.mean()), float(tr.std() or 1.0)
        constants[cell] = {"mean": mu, "sd": sd}
        m = out["cell_id"] == cell
        out.loc[m, "value"] = (out.loc[m, "value"] - mu) / sd
    return out, constants


def loco(items: list, X: np.ndarray, human: pd.DataFrame,
         cont: pd.DataFrame | None, binary: pd.DataFrame | None,
         K: int = 7, families: list[str] | None = None,
         out_path: str | Path | None = None,
         iter_warmup: int = 1000, iter_sampling: int = 1000, seed: int = 11,
         input_hashes: dict | None = None) -> dict:
    """Run the LOCO loop. Returns (and optionally writes) a report dict."""
    all_fams = sorted({it.construction for it in items})
    families = families or all_fams
    fam_of = {it.item_id: it.construction for it in items}

    per_family = {}
    pooled_pred, pooled_obs, pooled_fam = [], [], []
    for fam in families:
        held_items = {it.item_id for it in items if it.construction == fam}
        train_items = {it.item_id for it in items} - held_items
        train_human = human[~human["item_id"].isin(held_items)]
        held = human[human["item_id"].isin(held_items)]
        obs = held.groupby("item_id")["rating"].mean()
        if obs.empty:
            continue

        cont_std, _ = (_standardize_training_only(cont, train_items)
                       if cont is not None and len(cont) else (None, {}))
        data, maps = build_stan_data(items, X, train_human, cont_std, binary,
                                     K=K, standardize_scores=False,
                                     new_families={fam})
        fit, idata = fit_model(data, seed=seed,
                               iter_warmup=iter_warmup, iter_sampling=iter_sampling)
        diag = diagnostics_gate(fit, idata)

        ids, pred, lo, hi = _predict_observed_means_same_raters(idata, maps, held, K)
        o = obs.reindex(ids).to_numpy()
        resid = pred - o
        per_family[fam] = {
            "n_items": len(o),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "mean_signed_error": round(float(np.mean(resid)), 3),
            "spearman": (round(_spearman(pred, o), 3)
                         if _spearman(pred, o) == _spearman(pred, o) else None),
            "coverage90": round(float(np.mean((o >= lo) & (o <= hi))), 3),
            "diagnostics_passed": diag["passed"],
            "diagnostics": diag,
        }
        pooled_pred.extend(pred.tolist())
        pooled_obs.extend(o.tolist())
        pooled_fam.extend([fam] * len(o))

    fams_ok = [v for v in per_family.values() if v["diagnostics_passed"]]
    pooled_pred = np.array(pooled_pred)
    pooled_obs = np.array(pooled_obs)
    pooled_fam = np.array(pooled_fam)

    pooled_rho = _spearman(pooled_pred, pooled_obs)
    # family-cluster bootstrap: rank transfer uncertainty at the level the
    # sampling actually happened (families are the purposive units)
    rng = np.random.default_rng(seed)
    ufams = np.unique(pooled_fam)
    boots = []
    for _ in range(2000):
        pick = rng.choice(ufams, size=len(ufams), replace=True)
        idx = np.concatenate([np.flatnonzero(pooled_fam == f) for f in pick])
        r = _spearman(pooled_pred[idx], pooled_obs[idx])
        if r == r:
            boots.append(r)
    lower90 = float(np.percentile(boots, 10)) if boots else None

    report = {
        "target": "same participants, new items",
        "per_family": per_family,
        "n_families": len(per_family),
        "families_tested": sorted(per_family.keys()),
        "pooled_spearman": round(pooled_rho, 3) if pooled_rho == pooled_rho else None,
        "pooled_spearman_cluster_boot_lower90": (round(lower90, 3)
                                                 if lower90 is not None else None),
        "mean_spearman": round(float(np.mean(
            [v["spearman"] for v in fams_ok if v["spearman"] is not None])), 3) if fams_ok else None,
        "mean_rmse": round(float(np.mean([v["rmse"] for v in fams_ok])), 3) if fams_ok else None,
        "mean_coverage90": round(float(np.mean([v["coverage90"] for v in fams_ok])), 3) if fams_ok else None,
        "all_diagnostics_passed": bool(fams_ok) and all(
            v["diagnostics_passed"] for v in per_family.values()),
        "input_hashes": input_hashes or {},
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

import hashlib
import json
import subprocess
import uuid
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
    new_families: set | None = None,
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

    new_families = new_families or set()
    data: dict = dict(
        prior_only=0,
        N_item=len(items),
        N_constr=len(constrs),
        constr=[constr_ix[it.construction] for it in items],
        is_new=[int(c in new_families) for c in constrs],
        P=Xs.shape[1],
        X=Xs.tolist(),
    )
    maps: dict = dict(
        item_ids=item_ids, constructions=constrs,
        X_mean=X_mean.tolist(), X_sd=X_sd.tolist(), X_standardized=Xs.tolist(),
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
    core = [v for v in ["beta", "sigma_s", "tau_constr", "tau_item", "tau_a",
                        "tau_b", "omega", "kappa", "sigma_u", "b_b"]
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


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_fit(idata, maps: dict, report: dict, out_dir: str | Path,
             data: dict | None = None, input_hashes: dict | None = None) -> None:
    """Persist the fit plus a run manifest (run.json) binding the artifacts:
    downstream evidence writers stamp posterior_sha256 and the warrant refuses
    evidence whose stamp does not match the posterior it certifies."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(out / "posterior.nc"))
    (out / "index_maps.json").write_text(json.dumps(maps, indent=2))
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        commit = "unknown"
    n_crit = 0
    if data is not None and data.get("N_h", 0) > 0:
        n_crit = len(set(data["item_h"]))
    run = {
        "run_id": str(uuid.uuid4()),
        "posterior_sha256": sha256_file(out / "posterior.nc"),
        "stan_sha256": sha256_file(STAN_FILE),
        "code_commit": commit,
        "n_items_criterion": n_crit,
        "n_items_total": len(maps.get("item_ids", [])),
        "input_hashes": input_hashes or {},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))

```


---

## FILE: src/acceptometer/model/ppc.py

```
"""Posterior predictive checks at the real-fit stage, with consequences.

Two participant modes, both reported:

- conditional: replicates reuse the fitted participants' posterior u (checks
  the model against exactly the people observed);
- marginal: replicates draw fresh u from sigma_u (checks the participant-effect
  distribution itself, the one LOCO's fresh-rater predictions rely on).

Discrepancies, per mode:

- per-family spread of item mean ratings (vector of family ppps; the min is
  gated, so opposite family-specific failures cannot cancel into a passing
  average);
- within-item disagreement SD (global and per-family min);
- category usage, as a proper ppp: T(y) = total-variation distance between
  y's category frequencies and the mean replicate frequencies, compared for
  the observed data against the replicate distribution of the same statistic.

Plus an instrument-arm residual check: standardized mean residual per
(cell, family) from the posterior-mean linking; |z| > 3 is flagged (warn-only:
family deviations should absorb real structure, so a flag here means the
fitted deviations did not).

Gates (pre-committed): global ppps >= 0.01, min per-family ppp >= 0.005
(Bonferroni-flavored), category ppp >= 0.01, in BOTH participant modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_ratings(theta_d, u_obs, kappa_d, item_idx0, part_idx0, rng):
    eta = theta_d[item_idx0] + u_obs[part_idx0]
    cum = 1.0 / (1.0 + np.exp(-(kappa_d[None, :] - eta[:, None])))
    r = rng.uniform(size=eta.shape[0])
    return 1 + (r[:, None] > cum).sum(axis=1)


def _ppp(observed: float, sims: np.ndarray) -> float:
    lo = float(np.mean(sims <= observed))
    hi = float(np.mean(sims >= observed))
    return round(2 * min(lo, hi), 4)


def ppc_human(idata, maps: dict, human: pd.DataFrame,
              items: list, n_sims: int = 1000, seed: int = 2,
              out_path: str | Path | None = None,
              cont: pd.DataFrame | None = None,
              posterior_sha256: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    u = post["u"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    K = kappa.shape[0] + 1

    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}
    fam_of = {it.item_id: it.construction for it in items}

    item_idx0 = human["item_id"].map(item_pos).to_numpy()
    part_idx0 = human["participant_id"].map(part_pos).to_numpy()
    y_obs = human["rating"].astype(int).to_numpy()
    fam_col = human["item_id"].map(fam_of).to_numpy()
    fams = sorted(set(fam_col))

    def stats(y):
        df = pd.DataFrame({"y": y, "item": human["item_id"].to_numpy(),
                           "fam": fam_col})
        item_means = df.groupby("item")["y"].mean()
        fam_of_item = df.groupby("item")["fam"].first()
        spread_by_fam = item_means.groupby(fam_of_item).std()
        within = df.groupby("item")["y"].std()
        within_by_fam = within.groupby(fam_of_item).mean()
        cats = np.bincount(y, minlength=K + 1)[1:] / len(y)
        return (spread_by_fam.reindex(fams).to_numpy(),
                float(within.mean()),
                within_by_fam.reindex(fams).to_numpy(),
                cats)

    obs_spread, obs_within, obs_within_fam, obs_cats = stats(y_obs)

    draws = rng.choice(L, size=min(n_sims, L), replace=True)
    report: dict = {"n_sims": len(draws), "families": fams}
    if posterior_sha256:
        report["posterior_sha256"] = posterior_sha256
    all_pass = True

    for mode in ("conditional", "marginal"):
        sp, wi, wif, cats_list = [], [], [], []
        for d in draws:
            if mode == "conditional":
                u_d = u[:, d]
            else:
                u_d = rng.normal(0.0, sigma_u[d], u.shape[0])
            y_sim = _simulate_ratings(theta[:, d], u_d, kappa[:, d],
                                      item_idx0, part_idx0, rng)
            a, b, c, k = stats(y_sim)
            sp.append(a); wi.append(b); wif.append(c); cats_list.append(k)
        sp = np.array(sp); wi = np.array(wi); wif = np.array(wif)
        cats_arr = np.array(cats_list)

        fam_ppps = {f: _ppp(obs_spread[j], sp[:, j]) for j, f in enumerate(fams)
                    if obs_spread[j] == obs_spread[j]}
        wif_ppps = {f: _ppp(obs_within_fam[j], wif[:, j]) for j, f in enumerate(fams)
                    if obs_within_fam[j] == obs_within_fam[j]}
        # proper predictive reference for category usage: compare T(obs) with
        # the replicate distribution of the SAME statistic
        ref = cats_arr.mean(axis=0)
        t_obs = float(np.abs(obs_cats - ref).sum() / 2)
        t_rep = np.abs(cats_arr - ref[None, :]).sum(axis=1) / 2
        cat_ppp = float(np.mean(t_rep >= t_obs))

        mode_report = {
            "family_spread_ppp": fam_ppps,
            "family_spread_ppp_min": min(fam_ppps.values()),
            "within_item_sd_global_ppp": _ppp(obs_within, wi),
            "within_item_sd_family_ppp_min": min(wif_ppps.values()),
            "category_usage_ppp": round(cat_ppp, 4),
        }
        mode_pass = (mode_report["within_item_sd_global_ppp"] >= 0.01
                     and mode_report["family_spread_ppp_min"] >= 0.005
                     and mode_report["within_item_sd_family_ppp_min"] >= 0.005
                     and mode_report["category_usage_ppp"] >= 0.01)
        mode_report["passed"] = bool(mode_pass)
        report[mode] = mode_report
        all_pass = all_pass and mode_pass

    # instrument-arm residual check (warn-only)
    if cont is not None and len(cont) and "beta" in post:
        flags = []
        beta = post["beta"].mean(dim=("chain", "draw")).values
        alpha = post["alpha"].mean(dim=("chain", "draw")).values
        a_dev = post["a_dev"].mean(dim=("chain", "draw")).values
        b_dev = post["b_dev"].mean(dim=("chain", "draw")).values
        gamma = post["gamma"].mean(dim=("chain", "draw")).values
        sigma_s = post["sigma_s"].mean(dim=("chain", "draw")).values
        th = theta.mean(axis=1)
        Xs = np.asarray(maps.get("X_standardized")) if maps.get("X_standardized") else None
        cix = {c: j for j, c in enumerate(maps["cont_cells"])}
        constr_of = {iid: fam_of[iid] for iid in maps["item_ids"]}
        fam_ix = {f: j for j, f in enumerate(maps["constructions"])}
        for (cell, fam), grp in cont.assign(
                fam=cont["item_id"].map(constr_of)).groupby(["cell_id", "fam"]):
            m = cix.get(cell)
            if m is None:
                continue
            c = fam_ix[fam]
            ii = grp["item_id"].map(item_pos).to_numpy()
            mu = (alpha[m] + a_dev[m, c] + (beta[m] + b_dev[m, c]) * th[ii])
            if Xs is not None:
                mu = mu + Xs[ii] @ gamma[m]
            resid = grp["value"].to_numpy() - mu
            z = resid.mean() / (sigma_s[m] / np.sqrt(len(resid)))
            if abs(z) > 3:
                flags.append({"cell": cell, "family": fam, "z": round(float(z), 2)})
        report["instrument_residual_flags"] = flags
    report["passed"] = bool(all_pass)
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
certificate comes from a file actually read out of the run directory; missing
or failed evidence refuses the dependent tiers, it never grants them; and
evidence is refused unless it is BOUND to the posterior it claims to certify
(via run.json hashes), so a stale or copied report cannot license a different
fit.

The validation ladder is enforced literally:
  screening            needs diagnostics + fake-data recovery + SBC, all
                       passed, plus an unflagged cell whose NEW-FAMILY
                       predictive reliability clears the gate (the global
                       slope ratio is not a warrant quantity: a family slope
                       deviation changes how much signal that family's scores
                       carry).
  ranking              additionally needs LOCO: present, bound, every family
                       tested, all fold diagnostics passed, pooled tie-aware
                       Spearman > 0.6 with family-cluster bootstrap lower-90
                       > 0.5, and contamination assessed clean (LOCO rewards
                       contamination, so a suspect item source caps ranking
                       too).
  aggregate_estimation additionally needs the PPC (present, bound, both
                       participant modes passed) and LOCO coverage in band.
  everything above     refused in v1 with reasons; individual simulation and
                       mechanism claims permanently.

Thresholds are pre-registered decision defaults, not estimand-specific loss
analyses; the certificate says so.
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


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _multiverse_spread(run_dir: Path):
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


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _bound(child: dict | None, run: dict | None, how: str) -> tuple[bool, str]:
    """Check an evidence file's binding to the run manifest."""
    if child is None:
        return False, "evidence not produced"
    if run is None:
        return False, "run.json missing; evidence cannot be bound to a posterior"
    if how == "posterior":
        want = run.get("posterior_sha256")
        got = child.get("posterior_sha256")
        if got is None:
            return False, "evidence carries no posterior_sha256 stamp"
        if got != want:
            return False, "evidence is stamped for a different posterior"
    elif how == "inputs":
        want = run.get("input_hashes") or {}
        got = child.get("input_hashes") or {}
        if not got:
            return False, "evidence carries no input_hashes stamp"
        shared = set(want) & set(got)
        if not shared:
            return False, "evidence and run share no input hashes"
        for k in shared:
            if want[k] != got[k]:
                return False, f"input hash mismatch on {k}"
    return True, "bound"


def build_warrant(run_dir: str | Path, estimand: dict,
                  out_path: str | Path | None = None) -> dict:
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

    run = _read_json(run_dir / "run.json")
    diagnostics = _read_json(run_dir / "diagnostics.json")
    recovery = _read_json(run_dir / "recovery.json")
    sbc = _read_json(run_dir / "sbc.json")
    loco = _read_json(run_dir / "loco.json")
    ppc = _read_json(run_dir / "ppc.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None

    # ---- instruments: new-family predictive reliability is the gate quantity
    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    rel_new_stats: dict[str, dict] = {}
    if cont_cells and "reliability_new" in idata.posterior:
        rel_new = np.asarray(idata.posterior["reliability_new"].values)
        rel_new = rel_new.reshape(-1, rel_new.shape[-1])
        rel_glob = np.asarray(idata.posterior["reliability"].values)
        rel_glob = rel_glob.reshape(-1, rel_glob.shape[-1])
        rel_fam = np.asarray(idata.posterior["reliability_family"].values)
        rel_fam = rel_fam.reshape(-1, rel_fam.shape[-2], rel_fam.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel_new[:, j]))
            q10, q90 = np.percentile(rel_new[:, j], [10, 90])
            fam_meds = np.median(rel_fam[:, j, :], axis=0)
            rel_new_stats[name] = {"median": med, "q10": float(q10)}
            entry = {
                "cell": name,
                "reliability_new_family_median": round(med, 3),
                "reliability_new_family_80": [round(float(q10), 3), round(float(q90), 3)],
                "reliability_by_family": {
                    f: round(float(v), 3)
                    for f, v in zip(maps.get("constructions", []), fam_meds)
                },
                "global_slope_signal_ratio_median": round(
                    float(np.median(rel_glob[:, j])), 3),
                "alpha_standardized_median": round(float(np.median(alpha[:, j])), 3),
                "standardization": maps.get("cell_standardization", {}).get(name,
                                                                            "none recorded"),
            }
            instruments.append(entry)

    # prompt invariance from paraphrase-labeled cell names, if any
    groups: dict[str, list[str]] = defaultdict(list)
    for name in cont_cells:
        parts = name.split("/")
        if len(parts) >= 3:
            groups["/".join(parts[:2])].append(name)
    invariance = ("assessed descriptively at the grid level; paraphrases enter "
                  "the fit as one instrument, so no model-based invariance "
                  "statistic exists in this fit" if not groups else groups)

    estimand = dict(estimand)
    split_half = estimand.pop("human_split_half", "not provided")
    domain = {
        "construction_families": list(maps.get("constructions", [])),
        "n_items_total": (run or {}).get("n_items_total",
                                         len(maps.get("item_ids", []))),
        "n_items_with_human_criterion": (run or {}).get("n_items_criterion",
                                                        "not recorded"),
    }
    user_domain = estimand.pop("domain", None)
    if isinstance(user_domain, dict):
        domain.update(user_domain)
    if "population" not in estimand:
        estimand["population"] = ("the criterion sample's population "
                                  "(unspecified); population transfer untested")

    if isinstance(manifest, dict) and "contamination" in manifest:
        contamination = manifest["contamination"]
    else:
        contamination = "not assessed"
    contamination_clean = (
        contamination == "clean"
        or (isinstance(contamination, dict)
            and contamination.get("status") == "clean"))

    nonresponse = (manifest or {}).get("nonresponse") if isinstance(manifest, dict) else None
    flagged_cells = sorted(
        c for c, r in (nonresponse or {}).items()
        if isinstance(r, (int, float)) and r > 0.10)

    loco_ok, loco_bind_reason = _bound(loco, run, "inputs")
    ppc_ok_bind, ppc_bind_reason = _bound(ppc, run, "posterior")

    evidence = {
        "run": run if run is not None else "not produced",
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "sbc": sbc if sbc is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "loco_binding": loco_bind_reason,
        "ppc": ppc if ppc is not None else "not produced",
        "ppc_binding": ppc_bind_reason,
        "multiverse_spread": _multiverse_spread(run_dir),
        "prompt_invariance": invariance,
        "human_split_half": split_half,
        "contamination": contamination,
        "instrument_nonresponse": (
            nonresponse if nonresponse is not None else "not recorded"),
        "flagged_cells": flagged_cells or "none",
        "generalization_axes": {
            "tested": (["construction_family (LOCO-CV, purposive family "
                        "sample; descriptive over the families tested)"]
                       if loco is not None else []),
            "untested": ["population", "register", "language", "item_source",
                         "time", "model_version (beyond drift sentinels)"]
                        + ([] if loco is not None else ["construction_family"]),
        },
        "threshold_status": ("pre-registered decision defaults; not yet "
                             "estimand-specific loss analyses"),
    }

    licensed: dict[str, str] = {}
    refused: dict[str, str] = {}

    # ---- screening: full ladder prerequisite + unflagged predictive reliability
    diag_ok = bool(diagnostics and diagnostics.get("passed") is True)
    rec_ok = bool(recovery and recovery.get("passed") is True)
    sbc_ok = bool(sbc and sbc.get("passed") is True)
    candidates = {c: v for c, v in rel_new_stats.items() if c not in flagged_cells}
    best = max(candidates.items(), key=lambda kv: kv[1]["median"]) if candidates else None

    if diagnostics is None:
        refused["screening"] = "evidence not produced: diagnostics.json missing"
    elif not diag_ok:
        refused["screening"] = (
            f"diagnostics gate failed (rhat_max={diagnostics.get('rhat_max')}, "
            f"divergence_rate={diagnostics.get('divergence_rate')}, "
            f"ess_bulk_min={diagnostics.get('ess_bulk_min')})")
    elif recovery is None:
        refused["screening"] = "evidence not produced: recovery.json missing (ladder step 1)"
    elif not rec_ok:
        refused["screening"] = "fake-data recovery failed (ladder step 1)"
    elif sbc is None:
        refused["screening"] = "evidence not produced: sbc.json missing (ladder step 2)"
    elif not sbc_ok:
        refused["screening"] = "SBC failed (ladder step 2)"
    elif best is None:
        refused["screening"] = ("no unflagged continuous cell carries a "
                                "new-family predictive reliability")
    elif best[1]["median"] > 0.5 and best[1]["q10"] > 0.35:
        licensed["screening"] = (
            f"full ladder passed and cell {best[0]} new-family predictive "
            f"reliability median {best[1]['median']:.2f} > 0.5 with "
            f"q10 {best[1]['q10']:.2f} > 0.35")
    else:
        refused["screening"] = (
            f"best unflagged cell {best[0]}: new-family predictive reliability "
            f"median {best[1]['median']:.2f}, q10 {best[1]['q10']:.2f} "
            f"(need median > 0.5 and q10 > 0.35)")

    # ---- ranking
    all_fams = set(maps.get("constructions", []))
    if "screening" not in licensed:
        refused["ranking"] = "refused because screening is not granted"
    elif loco is None:
        refused["ranking"] = "evidence not produced: loco.json missing"
    elif not loco_ok:
        refused["ranking"] = f"LOCO evidence not bound to this run: {loco_bind_reason}"
    elif not loco.get("all_diagnostics_passed"):
        refused["ranking"] = "one or more LOCO fold fits failed diagnostics"
    elif set(loco.get("families_tested", [])) != all_fams:
        missing = sorted(all_fams - set(loco.get("families_tested", [])))
        refused["ranking"] = f"LOCO did not cover every family (missing: {missing})"
    elif not contamination_clean:
        refused["ranking"] = (
            "contamination cap: item source not assessed clean; contamination "
            "inflates the held-out rank statistic itself (LOCO rewards it)")
    elif (loco.get("pooled_spearman") or -1) > 0.6 and \
         (loco.get("pooled_spearman_cluster_boot_lower90") or -1) > 0.5:
        licensed["ranking"] = (
            f"pooled tie-aware held-out Spearman "
            f"{loco['pooled_spearman']:.2f} > 0.6 with family-cluster "
            f"bootstrap lower-90 {loco['pooled_spearman_cluster_boot_lower90']:.2f} > 0.5")
    else:
        refused["ranking"] = (
            f"pooled held-out Spearman {loco.get('pooled_spearman')} "
            f"(lower-90 {loco.get('pooled_spearman_cluster_boot_lower90')}) "
            "does not clear (0.6, 0.5)")

    # ---- aggregate estimation
    coverage = loco.get("mean_coverage90") if loco else None
    if "ranking" not in licensed:
        refused["aggregate_estimation"] = "refused because ranking is not granted"
    elif ppc is None:
        refused["aggregate_estimation"] = "evidence not produced: ppc.json missing (ladder step 4)"
    elif not ppc_ok_bind:
        refused["aggregate_estimation"] = f"PPC evidence not bound to this run: {ppc_bind_reason}"
    elif not ppc.get("passed"):
        refused["aggregate_estimation"] = (
            "posterior predictive check failed; the human arm misfits the "
            "criterion data, so aggregate predictions inherit unquantified bias")
    elif coverage is None:
        refused["aggregate_estimation"] = "loco.json carries no 90% interval coverage"
    elif 0.75 <= coverage <= 0.98:
        licensed["aggregate_estimation"] = (
            f"LOCO 90% interval coverage {coverage:.2f} within [0.75, 0.98], "
            "PPC passed in both participant modes, contamination assessed clean")
    else:
        refused["aggregate_estimation"] = (
            f"LOCO 90% interval coverage {coverage:.2f} outside [0.75, 0.98]")

    refused["effect_reproduction"] = ("not yet tested: requires matched "
                                      "experimental contrasts")
    refused["distributional_claims"] = (
        "no participant-level validation of variance structure"
        + ("" if ppc is None or ppc.get("passed")
           else "; posterior predictive check failed"))
    refused["population_transfer"] = (
        "refused: the v1 ladder contains no population-transfer test; the "
        "estimand population defaults to the criterion sample's own")
    refused["individual_simulation"] = ("refused permanently: an item-level "
                                        "instrument licenses no individual-level "
                                        "human simulation")
    refused["mechanism_claims"] = ("refused permanently: the model estimates a "
                                   "linking function, not a mechanism")

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_id": (run or {}).get("run_id", "not recorded"),
        "posterior_sha256": (run or {}).get("posterior_sha256", "not recorded"),
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
            "reliability quantities are fit- and item-set-specific; they go "
            "stale when the item pool changes",
            "thresholds are pre-registered defaults, not estimand-specific "
            "loss analyses",
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
        commit = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.revision = (f"{model_revision}; commit={commit}; "
                         f"transformers={transformers.__version__}")

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
                meta["slor_definition"] = "(subword_logprob_sum - word_unigram_sum) / n_words"
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
                # slor is a hybrid quantity here: subword model logprob minus a
                # word-level (wordfreq) unigram sum, normalized per WORD so the
                # two sums share a unit; when the first model token is unscored
                # (no BOS) the first word's unigram term is skipped to match
                unigram_sum, n_words = self._unigram_logprob_sum(
                    item, skip_first=bos_id is None)
                results.append(
                    {
                        "logprob_sum": logprob_sum,
                        "logprob_mean": logprob_sum / n_tokens if n_tokens else math.nan,
                        "slor": (
                            (logprob_sum - unigram_sum) / n_words
                            if n_tokens and n_words
                            else math.nan
                        ),
                        "n_tokens": n_tokens,
                        "n_words": n_words,
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
    def _unigram_logprob_sum(item: Item, skip_first: bool = False) -> tuple[float, int]:
        from wordfreq import zipf_frequency

        total, n = 0.0, 0
        first = True
        for raw_word in item.text.split():
            word = raw_word.strip(string.punctuation).lower()
            if not word:
                continue
            if first and skip_first:
                first = False
                continue
            first = False
            zipf = max(float(zipf_frequency(word, item.language)), 1.0)
            total += (zipf - 9.0) * math.log(10.0)
            n += 1
        return total, n

```


---

## FILE: scripts/pilot.py

```
"""Run the Sprouse-LI pilot end to end against the current model.

Usage: uv run python scripts/pilot.py [--skip-loco]

Reads data/pilot_items.jsonl, data/pilot_human.csv, and the cached
measurements in runs/pilot/measurements.jsonl; refits; runs recovery, PPC,
and LOCO; leaves every warrant prerequisite in runs/pilot/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.items import load_items, nuisance_covariates
from acceptometer.elicit.base import load_measurements, CONTINUOUS
from acceptometer.model.fit import (build_stan_data, fit_model, diagnostics_gate,
                                    save_fit, sha256_file)
from acceptometer.model.simulate import simulate, recovery_check, write_report
from acceptometer.model.ppc import ppc_human
from acceptometer.model.loco import loco

RUN = Path("runs/pilot")


def fit_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One instrument per model+method: Pythia via SLOR, qwen via pooled
    scalar; binary cells stay descriptive (see DECISIONS 2026-08-27)."""
    cont = df[df.kind == CONTINUOUS].copy()
    cont = cont[(cont.cell_id == "pythia-160m/slor")
                | cont.cell_id.str.startswith("qwen3:8b/prompt_scalar")]
    cont["cell_id"] = cont.cell_id.str.replace(r"/p\d$", "", regex=True)
    return cont[["item_id", "cell_id", "value"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loco", action="store_true")
    args = ap.parse_args()

    items = load_items("data/pilot_items.jsonl")
    human = pd.read_csv("data/pilot_human.csv")
    ms = load_measurements(RUN / "measurements.jsonl")
    df = pd.DataFrame([{"item_id": m.item_id, "cell_id": m.cell_id,
                        "kind": m.kind, "value": m.value} for m in ms])
    cont = fit_cells(df)
    X = np.array(nuisance_covariates(items))
    input_hashes = {
        "items": sha256_file("data/pilot_items.jsonl"),
        "human": sha256_file("data/pilot_human.csv"),
        "measurements": sha256_file(RUN / "measurements.jsonl"),
    }

    # ladder step 1: fake-data recovery against the CURRENT model
    sdata, truth = simulate()
    sfit, sidata = fit_model(sdata, seed=7)
    srec = recovery_check(sidata, truth)
    srec["diagnostics"] = diagnostics_gate(sfit, sidata)
    write_report(srec, RUN / "recovery.json")
    print("recovery passed:", srec["passed"])

    # real fit
    data, maps = build_stan_data(items, X, human, cont, None, K=7)
    fit, idata = fit_model(data, seed=42, iter_warmup=1500, iter_sampling=1500)
    diag = diagnostics_gate(fit, idata)
    save_fit(idata, maps, diag, RUN, data=data, input_hashes=input_hashes)
    cont.to_csv(RUN / "cont.csv", index=False)
    print("fit diag:", json.dumps(diag))

    run = json.loads((RUN / "run.json").read_text())
    rep = ppc_human(idata, maps, human, items, out_path=RUN / "ppc.json",
                    cont=cont, posterior_sha256=run["posterior_sha256"])
    print("PPC passed:", rep["passed"],
          "| marginal:", rep["marginal"]["passed"],
          "| conditional:", rep["conditional"]["passed"])

    if not args.skip_loco:
        # held-out-family geometry mixes slowly; 2000/2000 keeps fold fits
        # inside the production diagnostics gate
        lrep = loco(items, X, human, cont, None, K=7,
                    iter_warmup=2000, iter_sampling=2000,
                    out_path=RUN / "loco.json", input_hashes=input_hashes)
        print("LOCO:", json.dumps({k: v for k, v in lrep.items()
                                   if k not in ("per_family",)}))
    return 0 if diag["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

```


---

## FILE: PILOT-REPORT.md

```
# Acceptometer pilot report: Sprouse LI items, local instruments
<!-- SUMMARY: Real-data pilot of the warranted measurement pipeline after two external reviews; screening licensed, ranking refused on the contamination cap despite passing numbers · status: complete · updated: 2026-08-27 -->

## What ran

120 items from the Sprouse, Schütze & Almeida (2013) Linguistic Inquiry
judgment study (10 source-paper families, 6 starred / 6 good each), with all
1,519 available participant-level 7-point ratings (304 participants, ~12.7
ratings/item). Instruments, all local and free: Pythia-160m exact
log-probability cells (SLOR entering the fit, unit-consistent per-word
definition) and qwen3:8b prompted judgments (3 registered paraphrases x binary
and 1-7 scalar x 3 repeats at temperature 0.7; 2,160 chat calls, zero parse
failures, 11.5 minutes). One instrument per model+method enters the fit;
paraphrases and repeats are repeated measurements.

Every stage ran behind its gate, and the warrant enforces the ladder
literally: convergence diagnostics, fake-data recovery (including the freed
scale and family-deviation hyperparameters), SBC (R=100, diagnostics-aware),
posterior predictive checks in BOTH participant modes with per-family gates,
and leave-one-construction-family-out transfer in which the held-out family
is excluded from the training sum-to-zero vector and receives an independent
predictive family effect. All evidence is hash-bound to the posterior it
certifies.

## Headline numbers

| Quantity | Value |
|---|---|
| Human split-half reliability (item means, 200 splits) | r = .857 (Spearman-Brown .923) |
| qwen3:8b pooled scalar, item-mean r with humans | .724 |
| qwen3:8b new-family predictive reliability | .54 [80%: .39, .65]; by-family .50-.58 |
| Pythia-160m SLOR new-family predictive reliability | .36 [80%: .04, .68]; by-family .09-.57 |
| qwen instrument-by-item error (omega) | ~.56 latent-logit units |
| PPC (conditional and marginal modes) | both passed |
| LOCO pooled tie-aware Spearman (same raters, new items) | .742; family-cluster bootstrap lower-90 .654 |
| LOCO mean 90% coverage of observed means | .933; RMSE .99 |
| LOCO fold diagnostics | 10/10 clean at 2000/2000 iterations |

Per-family LOCO is the projectibility story: nine of ten families transfer at
Spearman .55-.95; one (34.1.fox) fails outright (-.03). Pythia's per-family
reliability spread (.09-.57) shows an instrument whose validity does not
project across families; qwen's (.50-.58) is family-stable on this domain.

## The warrant (runs/pilot/warrant.yaml)

Licensed: **screening only** — the full ladder passed and qwen's new-family
predictive reliability clears (median .54 > .5, q10 .39 > .35).

Refused: **ranking, on the contamination cap, despite passing numbers.** The
pooled held-out Spearman (.742, lower-90 .654) would clear the pre-registered
gate, but the LI materials are public since 2013 and contamination inflates
exactly the held-out rank statistic (LOCO rewards it), so a suspect item
source cannot license ranking. Also refused with reasons: aggregate
estimation (hierarchically), effect reproduction, distributional claims,
population transfer, individual simulation (permanent), mechanism claims
(permanent). Residual risks recorded: shared pretraining bias; item-set
specificity; thresholds are pre-registered defaults, not loss analyses.

Lifting the cap is a v2 experiment, not a rule change: a post-cutoff,
never-published item set with fresh human norms.

## What external review and real data changed (all logged in DECISIONS.md)

Review 1 (glm-5.3-flash, adversarial, pre-data): family-varying linking,
noise floor, conditional-reliability relabel, single-temperature elicitation,
observed-mean LOCO target, contamination caps, PPC gate.

Real data (each caught by a gate): freed latent scale (tau_item ~ 2.0);
one-instrument-per-model+method after correlated paraphrase cells outvoted
the human criterion; reflection-mode handling (data-informed inits +
zero-avoiding tau_item prior); instrument-by-item error, which repriced qwen
from a fictitious .90 to an honest .54.

Review 2 (GPT-family, via Brett): new-family effects outside the sum-to-zero
vector; ladder enforcement with hash-bound evidence; new-family predictive
reliability as the gate quantity (global ratio demoted); contamination cap
extended to ranking; tie-aware pooled Spearman with cluster bootstrap;
same-raters LOCO target; two-mode per-family PPC with a proper category-usage
reference; training-only standardization; SLOR unit consistency; checkpoint
provenance; expanded recovery and diagnostics gates.

## Reading the numbers against the literature

Published r = .8 human-LLM correlations (Qiu et al. 2024, ChatGPT) sit between
this pilot's qwen3:8b (.72) and the human split-half ceiling (.857). The
contribution is the decomposition and the certificate: how much is theta
signal, how much is the instrument's stable per-item opinion, how transfer
behaves family by family, and which claims the package does and does not
license, with every number bound to the posterior it came from.

## Caveats

Single language, single (contamination-suspect) item source, constructed
sentences, two instruments, one register, one date. The certificate lists the
untested axes explicitly. v2 priorities, in value order: post-cutoff item set
(tests aggregate estimation for real), ordered-logistic scalar arm,
binary-arm overdispersion, rater-specific cutpoints, prior-sensitivity sweep,
population-transfer test.

## Artifacts

`runs/pilot/`: posterior.nc, run.json (hashes), diagnostics.json,
recovery.json, sbc.json, ppc.json, loco.json, warrant.yaml, estimand.yaml,
split_half.json, measurements.jsonl, grid_manifest.yaml, plots. Data
provenance: `data/MANIFEST.yaml`. Pipeline: `scripts/pilot.py`.

```


---

## EVIDENCE (all from the post-review-2 model, hash-bound; see run.json)

### run.json

```json
{
  "run_id": "2868417c-3344-40c5-9f5f-bdd7f1af8407",
  "posterior_sha256": "80db0630980a5a751e45c033267c6c3fb8e07a1aea262d1461fd0973e3c69779",
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86",
  "code_commit": "f13c22b8ba8703f0c109bd2e0616fb4f376249b4",
  "n_items_criterion": 120,
  "n_items_total": 120,
  "input_hashes": {
    "items": "f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c",
    "human": "5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec",
    "measurements": "1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a"
  }
}
```

### recovery.json

```json
{
  "theta_90ci_coverage": 0.875,
  "theta_post_mean_corr_truth": 0.961,
  "beta_within_95ci": true,
  "beta_post_mean": [
    0.27,
    0.453,
    0.368
  ],
  "beta_truth": [
    0.312,
    0.421,
    0.359
  ],
  "tau_item_within_95ci": true,
  "tau_item_post_median": [
    1.126
  ],
  "tau_a_within_95ci": true,
  "tau_a_post_median": [
    0.081,
    0.115,
    0.121
  ],
  "tau_b_within_95ci": true,
  "tau_b_post_median": [
    0.099,
    0.139,
    0.147
  ],
  "reliability_max_abs_err": 0.08,
  "reliability_truth": [
    0.271,
    0.529,
    0.481
  ],
  "passed": true,
  "diagnostics": {
    "divergences": 0,
    "divergence_rate": 0.0,
    "rhat_max": 1.0046,
    "ess_bulk_min": 598,
    "passed": true
  }
}

```

### sbc.json (R=100, 500/500, diagnostics-aware)

```json
{
  "R": 100,
  "n_failed_fits": 0,
  "n_diag_failed": 19,
  "n_thin": 63,
  "params": {
    "tau_constr": {
      "chi2": 1.08,
      "p_uniform_approx": 0.9808,
      "rank_hist": [
        16,
        16,
        12,
        15,
        13,
        13,
        15
      ]
    },
    "tau_item": {
      "chi2": 1.78,
      "p_uniform_approx": 0.938,
      "rank_hist": [
        12,
        17,
        14,
        17,
        14,
        12,
        14
      ]
    },
    "sigma_u": {
      "chi2": 1.36,
      "p_uniform_approx": 0.9668,
      "rank_hist": [
        11,
        15,
        16,
        16,
        15,
        14,
        13
      ]
    },
    "beta[1]": {
      "chi2": 2.2,
      "p_uniform_approx": 0.9005,
      "rank_hist": [
        14,
        18,
        13,
        13,
        11,
        15,
        16
      ]
    },
    "beta[2]": {
      "chi2": 7.24,
      "p_uniform_approx": 0.2987,
      "rank_hist": [
        18,
        10,
        11,
        21,
        11,
        16,
        13
      ]
    },
    "sigma_s[1]": {
      "chi2": 1.5,
      "p_uniform_approx": 0.9582,
      "rank_hist": [
        14,
        11,
        16,
        14,
        13,
        16,
        16
      ]
    },
    "sigma_s[2]": {
      "chi2": 12.84,
      "p_uniform_approx": 0.0453,
      "rank_hist": [
        25,
        11,
        18,
        14,
        9,
        11,
        12
      ]
    }
  },
  "passed": true
}

```

### diagnostics.json + ppc.json

```json
{
  "divergences": 2,
  "divergence_rate": 0.0003,
  "rhat_max": 1.0032,
  "ess_bulk_min": 708,
  "passed": true
}
{
  "n_sims": 1000,
  "families": [
    "32.1.martin",
    "32.3.Culicover",
    "32.3.fanselow",
    "33.4.neeleman",
    "34.1.fox",
    "34.1.phillips",
    "34.3.heycock",
    "34.4.boskovic",
    "41.3.Landau",
    "41.3.Vicente"
  ],
  "posterior_sha256": "80db0630980a5a751e45c033267c6c3fb8e07a1aea262d1461fd0973e3c69779",
  "conditional": {
    "family_spread_ppp": {
      "32.1.martin": 0.99,
      "32.3.Culicover": 0.634,
      "32.3.fanselow": 0.246,
      "33.4.neeleman": 0.528,
      "34.1.fox": 0.994,
      "34.1.phillips": 0.788,
      "34.3.heycock": 0.772,
      "34.4.boskovic": 0.392,
      "41.3.Landau": 0.82,
      "41.3.Vicente": 0.858
    },
    "family_spread_ppp_min": 0.246,
    "within_item_sd_global_ppp": 0.754,
    "within_item_sd_family_ppp_min": 0.02,
    "category_usage_ppp": 1.0,
    "passed": true
  },
  "marginal": {
    "family_spread_ppp": {
      "32.1.martin": 0.926,
      "32.3.Culicover": 0.528,
      "32.3.fanselow": 0.186,
      "33.4.neeleman": 0.44,
      "34.1.fox": 0.906,
      "34.1.phillips": 0.954,
      "34.3.heycock": 0.768,
      "34.4.boskovic": 0.486,
      "41.3.Landau": 0.812,
      "41.3.Vicente": 0.922
    },
    "family_spread_ppp_min": 0.186,
    "within_item_sd_global_ppp": 0.752,
    "within_item_sd_family_ppp_min": 0.026,
    "category_usage_ppp": 1.0,
    "passed": true
  },
  "instrument_residual_flags": [
    {
      "cell": "pythia-160m/slor",
      "family": "32.1.martin",
      "z": 8.56
    },
    {
      "cell": "pythia-160m/slor",
      "family": "32.3.Culicover",
      "z": 9.22
    },
    {
      "cell": "pythia-160m/slor",
      "family": "32.3.fanselow",
      "z": 9.67
    },
    {
      "cell": "pythia-160m/slor",
      "family": "33.4.neeleman",
      "z": 7.38
    },
    {
      "cell": "pythia-160m/slor",
      "family": "34.1.fox",
      "z": 9.09
    },
    {
      "cell": "pythia-160m/slor",
      "family": "34.1.phillips",
      "z": 8.64
    },
    {
      "cell": "pythia-160m/slor",
      "family": "34.3.heycock",
      "z": 7.37
    },
    {
      "cell": "pythia-160m/slor",
      "family": "34.4.boskovic",
      "z": 8.95
    },
    {
      "cell": "pythia-160m/slor",
      "family": "41.3.Landau",
      "z": 8.82
    },
    {
      "cell": "pythia-160m/slor",
      "family": "41.3.Vicente",
      "z": 7.04
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "32.1.martin",
      "z": 130.22
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "32.3.Culicover",
      "z": 126.4
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "32.3.fanselow",
      "z": 106.4
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "33.4.neeleman",
      "z": 138.65
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "34.1.fox",
      "z": 154.5
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "34.1.phillips",
      "z": 128.28
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "34.3.heycock",
      "z": 140.16
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "34.4.boskovic",
      "z": 146.1
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "41.3.Landau",
      "z": 151.41
    },
    {
      "cell": "qwen3:8b/prompt_scalar",
      "family": "41.3.Vicente",
      "z": 129.85
    }
  ],
  "passed": true
}

```

### loco.json

```json
{
  "target": "same participants, new items",
  "per_family": {
    "32.1.martin": {
      "n_items": 12,
      "rmse": 0.847,
      "mean_signed_error": -0.467,
      "spearman": 0.884,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 5,
        "divergence_rate": 0.0006,
        "rhat_max": 1.006,
        "ess_bulk_min": 1248,
        "passed": true
      }
    },
    "32.3.Culicover": {
      "n_items": 12,
      "rmse": 0.862,
      "mean_signed_error": -0.016,
      "spearman": 0.546,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 4,
        "divergence_rate": 0.0005,
        "rhat_max": 1.0096,
        "ess_bulk_min": 799,
        "passed": true
      }
    },
    "32.3.fanselow": {
      "n_items": 12,
      "rmse": 1.145,
      "mean_signed_error": 0.481,
      "spearman": 0.951,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0027,
        "ess_bulk_min": 1449,
        "passed": true
      }
    },
    "33.4.neeleman": {
      "n_items": 12,
      "rmse": 1.003,
      "mean_signed_error": 0.344,
      "spearman": 0.789,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 3,
        "divergence_rate": 0.0004,
        "rhat_max": 1.0067,
        "ess_bulk_min": 1243,
        "passed": true
      }
    },
    "34.1.fox": {
      "n_items": 12,
      "rmse": 1.452,
      "mean_signed_error": -0.178,
      "spearman": -0.028,
      "coverage90": 0.833,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 2,
        "divergence_rate": 0.0003,
        "rhat_max": 1.0043,
        "ess_bulk_min": 963,
        "passed": true
      }
    },
    "34.1.phillips": {
      "n_items": 12,
      "rmse": 0.742,
      "mean_signed_error": -0.099,
      "spearman": 0.83,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 17,
        "divergence_rate": 0.0021,
        "rhat_max": 1.0052,
        "ess_bulk_min": 1462,
        "passed": true
      }
    },
    "34.3.heycock": {
      "n_items": 12,
      "rmse": 1.31,
      "mean_signed_error": -0.216,
      "spearman": 0.657,
      "coverage90": 0.833,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0044,
        "ess_bulk_min": 991,
        "passed": true
      }
    },
    "34.4.boskovic": {
      "n_items": 12,
      "rmse": 0.742,
      "mean_signed_error": 0.371,
      "spearman": 0.846,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0043,
        "ess_bulk_min": 813,
        "passed": true
      }
    },
    "41.3.Landau": {
      "n_items": 12,
      "rmse": 1.15,
      "mean_signed_error": 0.159,
      "spearman": 0.613,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0056,
        "ess_bulk_min": 1048,
        "passed": true
      }
    },
    "41.3.Vicente": {
      "n_items": 12,
      "rmse": 0.689,
      "mean_signed_error": -0.126,
      "spearman": 0.881,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 2,
        "divergence_rate": 0.0003,
        "rhat_max": 1.0057,
        "ess_bulk_min": 1011,
        "passed": true
      }
    }
  },
  "n_families": 10,
  "families_tested": [
    "32.1.martin",
    "32.3.Culicover",
    "32.3.fanselow",
    "33.4.neeleman",
    "34.1.fox",
    "34.1.phillips",
    "34.3.heycock",
    "34.4.boskovic",
    "41.3.Landau",
    "41.3.Vicente"
  ],
  "pooled_spearman": 0.742,
  "pooled_spearman_cluster_boot_lower90": 0.654,
  "mean_spearman": 0.697,
  "mean_rmse": 0.994,
  "mean_coverage90": 0.933,
  "all_diagnostics_passed": true,
  "input_hashes": {
    "items": "f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c",
    "human": "5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec",
    "measurements": "1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a"
  }
}

```

### warrant.yaml (the emitted certificate)

```yaml
generated: '2026-08-27T15:53:15'
run_dir: runs/pilot
run_id: 2868417c-3344-40c5-9f5f-bdd7f1af8407
posterior_sha256: 80db0630980a5a751e45c033267c6c3fb8e07a1aea262d1461fd0973e3c69779
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
  n_items_total: 120
  n_items_with_human_criterion: 120
  item_source: Sprouse-Schütze-Almeida 2013 LI materials, pilot subset
  language: English (US)
  register: constructed linguistic example sentences
  families: 10 LI source papers, 12 items each (6 starred / 6 good)
instruments:
- cell: pythia-160m/slor
  reliability_new_family_median: 0.362
  reliability_new_family_80:
  - 0.041
  - 0.678
  reliability_by_family:
    32.1.martin: 0.368
    32.3.Culicover: 0.189
    32.3.fanselow: 0.388
    33.4.neeleman: 0.538
    34.1.fox: 0.089
    34.1.phillips: 0.564
    34.3.heycock: 0.296
    34.4.boskovic: 0.375
    41.3.Landau: 0.167
    41.3.Vicente: 0.569
  global_slope_signal_ratio_median: 0.347
  alpha_standardized_median: 0.013
  standardization:
    mean: 1.1467543535215112
    sd: 1.201321652494467
- cell: qwen3:8b/prompt_scalar
  reliability_new_family_median: 0.538
  reliability_new_family_80:
  - 0.393
  - 0.648
  reliability_by_family:
    32.1.martin: 0.552
    32.3.Culicover: 0.538
    32.3.fanselow: 0.513
    33.4.neeleman: 0.516
    34.1.fox: 0.504
    34.1.phillips: 0.557
    34.3.heycock: 0.538
    34.4.boskovic: 0.566
    41.3.Landau: 0.511
    41.3.Vicente: 0.575
  global_slope_signal_ratio_median: 0.538
  alpha_standardized_median: -0.006
  standardization:
    mean: 4.241666666666666
    sd: 1.9625958339422325
evidence:
  run:
    run_id: 2868417c-3344-40c5-9f5f-bdd7f1af8407
    posterior_sha256: 80db0630980a5a751e45c033267c6c3fb8e07a1aea262d1461fd0973e3c69779
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
    code_commit: f13c22b8ba8703f0c109bd2e0616fb4f376249b4
    n_items_criterion: 120
    n_items_total: 120
    input_hashes:
      items: f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c
      human: 5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec
      measurements: 1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a
  diagnostics:
    divergences: 2
    divergence_rate: 0.0003
    rhat_max: 1.0032
    ess_bulk_min: 708
    passed: true
  fake_data_recovery:
    theta_90ci_coverage: 0.875
    theta_post_mean_corr_truth: 0.961
    beta_within_95ci: true
    beta_post_mean:
    - 0.27
    - 0.453
    - 0.368
    beta_truth:
    - 0.312
    - 0.421
    - 0.359
    tau_item_within_95ci: true
    tau_item_post_median:
    - 1.126
    tau_a_within_95ci: true
    tau_a_post_median:
    - 0.081
    - 0.115
    - 0.121
    tau_b_within_95ci: true
    tau_b_post_median:
    - 0.099
    - 0.139
    - 0.147
    reliability_max_abs_err: 0.08
    reliability_truth:
    - 0.271
    - 0.529
    - 0.481
    passed: true
    diagnostics:
      divergences: 0
      divergence_rate: 0.0
      rhat_max: 1.0046
      ess_bulk_min: 598
      passed: true
  sbc:
    R: 100
    n_failed_fits: 0
    n_diag_failed: 19
    n_thin: 63
    params:
      tau_constr:
        chi2: 1.08
        p_uniform_approx: 0.9808
        rank_hist:
        - 16
        - 16
        - 12
        - 15
        - 13
        - 13
        - 15
      tau_item:
        chi2: 1.78
        p_uniform_approx: 0.938
        rank_hist:
        - 12
        - 17
        - 14
        - 17
        - 14
        - 12
        - 14
      sigma_u:
        chi2: 1.36
        p_uniform_approx: 0.9668
        rank_hist:
        - 11
        - 15
        - 16
        - 16
        - 15
        - 14
        - 13
      beta[1]:
        chi2: 2.2
        p_uniform_approx: 0.9005
        rank_hist:
        - 14
        - 18
        - 13
        - 13
        - 11
        - 15
        - 16
      beta[2]:
        chi2: 7.24
        p_uniform_approx: 0.2987
        rank_hist:
        - 18
        - 10
        - 11
        - 21
        - 11
        - 16
        - 13
      sigma_s[1]:
        chi2: 1.5
        p_uniform_approx: 0.9582
        rank_hist:
        - 14
        - 11
        - 16
        - 14
        - 13
        - 16
        - 16
      sigma_s[2]:
        chi2: 12.84
        p_uniform_approx: 0.0453
        rank_hist:
        - 25
        - 11
        - 18
        - 14
        - 9
        - 11
        - 12
    passed: true
  loco_transfer:
    target: same participants, new items
    per_family:
      32.1.martin:
        n_items: 12
        rmse: 0.847
        mean_signed_error: -0.467
        spearman: 0.884
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 5
          divergence_rate: 0.0006
          rhat_max: 1.006
          ess_bulk_min: 1248
          passed: true
      32.3.Culicover:
        n_items: 12
        rmse: 0.862
        mean_signed_error: -0.016
        spearman: 0.546
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 4
          divergence_rate: 0.0005
          rhat_max: 1.0096
          ess_bulk_min: 799
          passed: true
      32.3.fanselow:
        n_items: 12
        rmse: 1.145
        mean_signed_error: 0.481
        spearman: 0.951
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0027
          ess_bulk_min: 1449
          passed: true
      33.4.neeleman:
        n_items: 12
        rmse: 1.003
        mean_signed_error: 0.344
        spearman: 0.789
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 3
          divergence_rate: 0.0004
          rhat_max: 1.0067
          ess_bulk_min: 1243
          passed: true
      34.1.fox:
        n_items: 12
        rmse: 1.452
        mean_signed_error: -0.178
        spearman: -0.028
        coverage90: 0.833
        diagnostics_passed: true
        diagnostics:
          divergences: 2
          divergence_rate: 0.0003
          rhat_max: 1.0043
          ess_bulk_min: 963
          passed: true
      34.1.phillips:
        n_items: 12
        rmse: 0.742
        mean_signed_error: -0.099
        spearman: 0.83
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 17
          divergence_rate: 0.0021
          rhat_max: 1.0052
          ess_bulk_min: 1462
          passed: true
      34.3.heycock:
        n_items: 12
        rmse: 1.31
        mean_signed_error: -0.216
        spearman: 0.657
        coverage90: 0.833
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0044
          ess_bulk_min: 991
          passed: true
      34.4.boskovic:
        n_items: 12
        rmse: 0.742
        mean_signed_error: 0.371
        spearman: 0.846
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0043
          ess_bulk_min: 813
          passed: true
      41.3.Landau:
        n_items: 12
        rmse: 1.15
        mean_signed_error: 0.159
        spearman: 0.613
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0056
          ess_bulk_min: 1048
          passed: true
      41.3.Vicente:
        n_items: 12
        rmse: 0.689
        mean_signed_error: -0.126
        spearman: 0.881
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 2
          divergence_rate: 0.0003
          rhat_max: 1.0057
          ess_bulk_min: 1011
          passed: true
    n_families: 10
    families_tested:
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
    pooled_spearman: 0.742
    pooled_spearman_cluster_boot_lower90: 0.654
    mean_spearman: 0.697
    mean_rmse: 0.994
    mean_coverage90: 0.933
    all_diagnostics_passed: true
    input_hashes:
      items: f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c
      human: 5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec
      measurements: 1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a
  loco_binding: bound
  ppc:
    n_sims: 1000
    families:
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
    posterior_sha256: 80db0630980a5a751e45c033267c6c3fb8e07a1aea262d1461fd0973e3c69779
    conditional:
      family_spread_ppp:
        32.1.martin: 0.99
        32.3.Culicover: 0.634
        32.3.fanselow: 0.246
        33.4.neeleman: 0.528
        34.1.fox: 0.994
        34.1.phillips: 0.788
        34.3.heycock: 0.772
        34.4.boskovic: 0.392
        41.3.Landau: 0.82
        41.3.Vicente: 0.858
      family_spread_ppp_min: 0.246
      within_item_sd_global_ppp: 0.754
      within_item_sd_family_ppp_min: 0.02
      category_usage_ppp: 1.0
      passed: true
    marginal:
      family_spread_ppp:
        32.1.martin: 0.926
        32.3.Culicover: 0.528
        32.3.fanselow: 0.186
        33.4.neeleman: 0.44
        34.1.fox: 0.906
        34.1.phillips: 0.954
        34.3.heycock: 0.768
        34.4.boskovic: 0.486
        41.3.Landau: 0.812
        41.3.Vicente: 0.922
      family_spread_ppp_min: 0.186
      within_item_sd_global_ppp: 0.752
      within_item_sd_family_ppp_min: 0.026
      category_usage_ppp: 1.0
      passed: true
    instrument_residual_flags:
    - cell: pythia-160m/slor
      family: 32.1.martin
      z: 8.56
    - cell: pythia-160m/slor
      family: 32.3.Culicover
      z: 9.22
    - cell: pythia-160m/slor
      family: 32.3.fanselow
      z: 9.67
    - cell: pythia-160m/slor
      family: 33.4.neeleman
      z: 7.38
    - cell: pythia-160m/slor
      family: 34.1.fox
      z: 9.09
    - cell: pythia-160m/slor
      family: 34.1.phillips
      z: 8.64
    - cell: pythia-160m/slor
      family: 34.3.heycock
      z: 7.37
    - cell: pythia-160m/slor
      family: 34.4.boskovic
      z: 8.95
    - cell: pythia-160m/slor
      family: 41.3.Landau
      z: 8.82
    - cell: pythia-160m/slor
      family: 41.3.Vicente
      z: 7.04
    - cell: qwen3:8b/prompt_scalar
      family: 32.1.martin
      z: 130.22
    - cell: qwen3:8b/prompt_scalar
      family: 32.3.Culicover
      z: 126.4
    - cell: qwen3:8b/prompt_scalar
      family: 32.3.fanselow
      z: 106.4
    - cell: qwen3:8b/prompt_scalar
      family: 33.4.neeleman
      z: 138.65
    - cell: qwen3:8b/prompt_scalar
      family: 34.1.fox
      z: 154.5
    - cell: qwen3:8b/prompt_scalar
      family: 34.1.phillips
      z: 128.28
    - cell: qwen3:8b/prompt_scalar
      family: 34.3.heycock
      z: 140.16
    - cell: qwen3:8b/prompt_scalar
      family: 34.4.boskovic
      z: 146.1
    - cell: qwen3:8b/prompt_scalar
      family: 41.3.Landau
      z: 151.41
    - cell: qwen3:8b/prompt_scalar
      family: 41.3.Vicente
      z: 129.85
    passed: true
  ppc_binding: bound
  multiverse_spread:
    mean_per_item_sd_across_cells: 0.627
    n_items: 120
    n_cells: 2
  prompt_invariance: assessed descriptively at the grid level; paraphrases enter the
    fit as one instrument, so no model-based invariance statistic exists in this fit
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
  threshold_status: pre-registered decision defaults; not yet estimand-specific loss
    analyses
licensed_claims:
  screening: full ladder passed and cell qwen3:8b/prompt_scalar new-family predictive
    reliability median 0.54 > 0.5 with q10 0.39 > 0.35
refused_claims:
  ranking: 'contamination cap: item source not assessed clean; contamination inflates
    the held-out rank statistic itself (LOCO rewards it)'
  aggregate_estimation: refused because ranking is not granted
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
- reliability quantities are fit- and item-set-specific; they go stale when the item
  pool changes
- thresholds are pre-registered defaults, not estimand-specific loss analyses

```

---

## PRIOR REVIEWS AND TRIAGE

### Review 1 (glm-5.3-flash)

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


### Triage of both reviews (from DECISIONS.md)

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
