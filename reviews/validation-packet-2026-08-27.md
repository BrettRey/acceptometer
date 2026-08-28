# Acceptometer external validation packet (2026-08-27, rev 5: final pilot)

## Review brief

You are adversarially validating a Bayesian measurement tool that treats LLMs
as biased instruments for human acceptability judgments. Three prior
adversarial reviews (triage attached) have been adopted; the emitted
certificate now licenses NO deployment tier (descriptive_only): deployment
claims are refused as affirmative_failure (a participant-entropy PPC failure)
and would be vitiated by contamination regardless. Your job: find what all
three reviewers and the gates still missed, and audit whether the
descriptive-only verdict and its evidence are themselves sound.

Priority questions:
1. SPEC-VS-CODE DRIFT across DESIGN.md, acceptometer.stan, warrant.py,
   thresholds.yaml.
2. The directional new-family quantities (p_pos_new, zeroed reversed-slope
   reliability, within-family variance): right quantities, right gates?
3. The binding scheme (recomputed posterior/Stan hashes, exact input maps,
   Stan-hash binding for simulation evidence and LOCO): find a bypass.
4. newfam_check: does it validate what the warrant uses? Is the
   family-mean-centered coverage the right identified claim?
5. The PPC verdict: participant-entropy ppp .000/.006 with everything else
   passing - is the diagnosis (rater scale-use heterogeneity vs additive
   intercept) right, and is any-failure-blocks-all the right pre-commitment?
6. The contested-band analysis (band [3,5], per-cell r inside/outside):
   any artifact that could produce the observed collapse (restriction of
   range within the band, measurement error in band membership, regression
   artifacts)? The band r's are computed on ~47 items with human-mean
   splitting - quantify what part of the inside/outside gap is mechanical.
7. The Opus 4.6 batched protocol (20 items/call): what does batching do to
   comparability with single-item cells, and is the cell-identity separation
   sufficient?
8. EVIDENCE CONSISTENCY: verify every number in PILOT-REPORT.md against the
   attached artifacts.


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

- `reliability[m]` — the **global-slope signal ratio**, using beta_m alone
  and the pooled theta variance. Descriptive only, never a warrant quantity.
- `reliability_family[m, c]` — per-family, WITHIN-family signal variance
  (tau_item^2), sign-aware: a reversed realized slope scores zero rather than
  being laundered into signal by squaring.
- `p_pos_new[m]` and `reliability_new_directional[m]` — the new-family
  predictive pair: each posterior draw samples a fresh slope deviation;
  p_pos_new is the predictive probability the new family's slope has the
  validated orientation, and the directional reliability zeroes reversed
  draws. Both are model-based extrapolations CONDITIONAL on exchangeability
  of construction families, which are purposive — the certificate labels them
  as such wherever they appear. Screening gates on the pair
  (P(positive) >= 0.95, median > 0.5, q10 > 0.35).
- **Structural limit**: absolute location of a family with no human anchor is
  prior-identified, not likelihood-identified (delta-invariance against
  per-cell family intercepts), so unanchored family-location claims are
  refused permanently; within-family contrasts are the identified claims.

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
   Human arm: two participant modes (conditional on fitted raters, marginal
   with fresh raters), per-family discrepancy vectors gated on minima,
   category usage with a proper replicate reference, and participant
   response-style checks (entropy, range). Instrument arm: a GATED
   posterior-predictive check per (cell, family) on the fitted standardized
   scale with every model term including the item effect — the measurement
   model that produces every instrument-based claim must fit before any is
   granted.
4b. **Simulated new-family recovery** (`newfam_check`): simulate under known
   truth, withhold whole families' human data, mark them new, and verify the
   branch the warrant actually uses: within-family theta coverage, positive
   within-family rank recovery, and reported (unGated) family-location bias —
   the empirical face of the prior-identification limit.
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

The certificate implements the assurance structure of the portfolio's
AI-safety papers. A grant is an inference licence with a life-cycle
(ai-evaluation-kinds): the `licence` block states issuer, threshold-spec
version and hash, expiry conditions (instrument, item-pool, or model change),
erosion conditions (optimizing anything against the certificate's own gate
statistics vitiates them), defeat/supersession, and a contestation route.
Each licensed claim carries the four projective-claim blocks from the shared
claim-types protocol: declaration (bearer, unit, population, range,
time/version), the analyst's stated projectibility profile (a hypothesis,
never fused with the warrant), the warrant itself, and defeaters. Refusals
are typed with remedies per evidentiary-assurance's verdict separation:
`unevaluable` (produce the evidence), `shortfall` (adequate evidence, gate
missed), `affirmative_failure` (fix the model), `vitiated` (contamination),
`structural` (outside the design's claim space). Thresholds live in a frozen
versioned spec (`thresholds.yaml`), cited by version and SHA-256; changing a
gate is a spec revision, never a silent code edit.


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

## FILE: src/acceptometer/thresholds.yaml

```
# Frozen decision-threshold specification for acceptometer warrants.
#
# This file is the single versioned source for every gate value. Certificates
# cite spec_version and the file's SHA-256, so a threshold change is a visible
# spec revision, never a silent code edit (practice adopted from the
# delegation-assurance frozen-specification discipline).
#
# Status: pre-registered decision defaults. These are NOT estimand-specific
# loss analyses; a deployment with a real loss function should derive its own
# spec revision and say so.

spec_version: 1.0.0
frozen: 2026-08-27

diagnostics:
  divergence_rate_max: 0.005
  rhat_max: 1.01
  ess_bulk_min: 400

sbc:
  failure_frac_max: 0.10

ppc:
  global_ppp_min: 0.01
  family_ppp_min: 0.005
  instrument_ppp_min: 0.005

nonresponse_flag_rate: 0.10

screening:
  p_pos_min: 0.95
  rel_median_min: 0.5
  rel_q10_min: 0.35

ranking_within_family:
  mean_spearman_min: 0.6
  frac_families_gt_0p3_min: 0.8

aggregate_estimation:
  coverage_band: [0.75, 0.98]
  rmse_max_frac_of_observed_sd: 0.75

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


def newfam_check(n_new: int = 2, seed: int = 99, **fit_kw) -> dict:
    """Simulated-LOCO validation of the new-family branch under known truth:
    simulate, mark n_new families NEW (excluded from the sum-to-zero vector),
    drop their human observations, fit, and check the quantities the warrant
    would actually use for a new family. Gates: held-item theta 90% coverage
    in [0.75, 0.98] AFTER family-mean centering (absolute family location is
    prior-identified without an anchor -- the delta-invariance with a_dev --
    so the calibrated claim is the within-family one); within-family rank
    recovery positive in every held family; and the family-location bias is
    REPORTED, not gated, as the empirical face of the prior-identification
    limit."""
    from .fit import fit_model, diagnostics_gate

    data, truth = simulate(seed=seed)
    n_constr = data["N_constr"]
    new_fams = list(range(n_constr - n_new + 1, n_constr + 1))
    constr = np.array(data["constr"])
    held_items = {i + 1 for i in range(data["N_item"]) if constr[i] in new_fams}

    keep = [j for j, it in enumerate(data["item_h"]) if it not in held_items]
    data = dict(data)
    data["item_h"] = [data["item_h"][j] for j in keep]
    data["part_h"] = [data["part_h"][j] for j in keep]
    data["y"] = [data["y"][j] for j in keep]
    data["N_h"] = len(keep)
    data["is_new"] = [1 if c in new_fams else 0 for c in range(1, n_constr + 1)]

    fit, idata = fit_model(data, seed=seed, **fit_kw)
    diag = diagnostics_gate(fit, idata)
    post = idata.posterior
    th = post["theta"].stack(d=("chain", "draw")).values

    report = {"n_new_families": n_new, "diagnostics": diag, "per_family": {}}
    cover_all, rank_ok = [], True
    for c in new_fams:
        idx = np.flatnonzero(constr == c)
        t_true = truth.theta[idx]
        t_draws = th[idx]                                   # (n_items, L)
        # center within family, per draw, to test the identified claim
        t_draws_c = t_draws - t_draws.mean(axis=0, keepdims=True)
        t_true_c = t_true - t_true.mean()
        lo, hi = np.percentile(t_draws_c, [5, 95], axis=1)
        cov = float(np.mean((t_true_c >= lo) & (t_true_c <= hi)))
        cover_all.append(cov)
        pm = t_draws.mean(axis=1)
        rho = float(np.corrcoef(
            np.argsort(np.argsort(pm)), np.argsort(np.argsort(t_true)))[0, 1])
        rank_ok = rank_ok and rho > 0
        loc_bias = float(t_draws.mean() - t_true.mean())
        report["per_family"][f"fam{c}"] = {
            "theta_within_coverage90": round(cov, 3),
            "within_rank_corr_truth": round(rho, 3),
            "family_location_bias": round(loc_bias, 3),
        }
    mean_cov = float(np.mean(cover_all))
    report["mean_within_coverage90"] = round(mean_cov, 3)
    report["passed"] = bool(diag["passed"] and 0.75 <= mean_cov <= 0.98 and rank_ok)
    from .fit import STAN_FILE, sha256_file
    report["stan_sha256"] = sha256_file(STAN_FILE)
    report["note"] = ("family_location_bias is reported unGated: absolute "
                      "location of an unanchored family is prior-identified "
                      "(delta-invariance with per-cell family intercepts)")
    return report

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
        [f"sigma_s[{m}]" for m in range(1, M_c + 1)] + \
        [f"tau_b[{m}]" for m in range(1, M_c + 1)] + \
        [f"omega[{m}]" for m in range(1, M_c + 1)]
    ranks: dict[str, list[int]] = {t: [] for t in tracked}
    n_failed = 0
    n_diag_failed = 0
    failed_truths: list[dict] = []

    for r in range(R):
        pr = _draw_prior(rng, n_constr, K, M_c, M_b, P)
        data = _simulate_given(pr, rng, n_constr, items_per_constr, n_part,
                               ratings_per_item, K, M_c, M_b, P)
        try:
            fit, idata = fit_model(data, seed=seed + r, iter_warmup=500,
                                   iter_sampling=500, chains=4)
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
            # exclude this replication's ranks entirely: draws the check judged
            # unreliable must not enter the uniformity evidence; record where
            # in prior space the failures concentrate instead
            n_diag_failed += 1
            failed_truths.append({"tau_constr": round(float(pr["tau_constr"]), 2),
                                  "tau_item": round(float(pr["tau_item"]), 2),
                                  "sigma_u": round(float(pr["sigma_u"]), 2)})
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
    report["failed_fit_truths"] = failed_truths
    from ..spec import load_spec
    cap = load_spec()["sbc"]["failure_frac_max"]
    if (n_failed + n_diag_failed) > cap * R:
        passed = False
        report["failure_note"] = (f"more than {cap:.0%} of replications failed "
                                  "or failed diagnostics")
    report["passed"] = passed
    from .fit import STAN_FILE, sha256_file
    report["stan_sha256"] = sha256_file(STAN_FILE)
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
        # nuisance covariates: training-family constants only, applied
        # unchanged to the held family (inductive, not transductive)
        tr_rows = [j for j, it in enumerate(items) if it.item_id in train_items]
        Xtr = np.asarray(X, dtype=float)[tr_rows]
        X_stats = (Xtr.mean(axis=0), Xtr.std(axis=0))
        data, maps = build_stan_data(items, X, train_human, cont_std, binary,
                                     K=K, standardize_scores=False,
                                     new_families={fam}, X_stats=X_stats)
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
    # the pooled statistic conflates between-family separation with
    # within-family ordering (well-separated family means can carry a high
    # pooled rho over random within-family ranks), so the claim-relevant
    # quantities are reported separately:
    fam_pred = pd.Series(pooled_pred).groupby(pd.Series(pooled_fam)).mean()
    fam_obs = pd.Series(pooled_obs).groupby(pd.Series(pooled_fam)).mean()
    between_rho = _spearman(fam_pred.to_numpy(), fam_obs.to_numpy())
    within = [v["spearman"] for v in per_family.values()
              if v["spearman"] is not None]
    frac_within_gt_03 = (float(np.mean([w > 0.3 for w in within]))
                         if within else None)
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
        "pooled_spearman_descriptive": (round(pooled_rho, 3)
                                        if pooled_rho == pooled_rho else None),
        "pooled_spearman_cluster_boot_lower90": (round(lower90, 3)
                                                 if lower90 is not None else None),
        "between_family_spearman": (round(between_rho, 3)
                                    if between_rho == between_rho else None),
        "within_family_spearman_min": (round(min(within), 3) if within else None),
        "frac_families_within_spearman_gt_0.3": frac_within_gt_03,
        "sd_observed_item_means": round(float(np.std(pooled_obs)), 3),
        "mean_spearman": round(float(np.mean(
            [v["spearman"] for v in fams_ok if v["spearman"] is not None])), 3) if fams_ok else None,
        "mean_rmse": round(float(np.mean([v["rmse"] for v in fams_ok])), 3) if fams_ok else None,
        "mean_coverage90": round(float(np.mean([v["coverage90"] for v in fams_ok])), 3) if fams_ok else None,
        "all_diagnostics_passed": bool(fams_ok) and all(
            v["diagnostics_passed"] for v in per_family.values()),
        "input_hashes": input_hashes or {},
    }
    # LOCO refits the model, so it binds by BOTH input hashes and Stan source:
    # a report generated under an older model with the same inputs must refuse
    from .fit import STAN_FILE
    from .fit import sha256_file as _sha
    report["stan_sha256"] = _sha(STAN_FILE)
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
    X_stats: tuple | None = None,
) -> tuple[dict, dict]:
    """Returns (stan_data, index_maps). index_maps records the id->index
    mappings and per-cell standardization constants so posteriors can be
    mapped back to names and raw scales."""
    item_ids = [it.item_id for it in items]
    item_ix = {iid: i + 1 for i, iid in enumerate(item_ids)}
    constrs = sorted({it.construction for it in items})
    constr_ix = {c: i + 1 for i, c in enumerate(constrs)}

    X = np.asarray(X, dtype=float)
    if X_stats is not None:
        X_mean, X_sd = (np.asarray(X_stats[0], dtype=float),
                        np.asarray(X_stats[1], dtype=float))
    else:
        X_mean, X_sd = X.mean(axis=0), X.std(axis=0)
    X_sd = np.where(X_sd == 0, 1.0, X_sd)
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
    """Hard gate on core parameters; thresholds come from the frozen spec
    (thresholds.yaml). Divergence tolerance is deliberately nonzero: Stan's
    strict reading is that any divergence is suspect, and the rate is always
    reported so a stricter reader can refuse. Returns report with `passed`."""
    import arviz as az

    from ..spec import load_spec
    g = load_spec()["diagnostics"]

    div = int(np.sum(fit.method_variables()["divergent__"]))
    n_draws = int(np.prod(fit.method_variables()["divergent__"].shape))
    core = [v for v in ["beta", "sigma_s", "tau_constr", "tau_item", "tau_a",
                        "tau_b", "omega", "kappa", "sigma_u", "b_b",
                        "a_dev", "b_dev", "theta"]
            if v in idata.posterior]
    if "mu_new_raw" in idata.posterior and idata.posterior["mu_new_raw"].shape[-1] > 0:
        core.append("mu_new_raw")
    summ = az.summary(idata, var_names=core)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    report = {
        "divergences": div,
        "divergence_rate": round(div / n_draws, 4),
        "rhat_max": round(rhat_max, 4),
        "ess_bulk_min": int(ess_min),
        "passed": bool(div / n_draws < g["divergence_rate_max"]
                       and rhat_max < g["rhat_max"]
                       and ess_min > g["ess_bulk_min"]),
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


def mode_audit(data: dict, seed: int = 3, iter_warmup: int = 750,
               iter_sampling: int = 750) -> dict:
    """Deliberately overdispersed-start check for the known reflection basin.

    Data-informed inits place all production chains in one basin; agreement
    among them cannot show the other basin has negligible mass. This audit
    runs extra chains from random inits and compares the best log posterior
    density found in each orientation (sign of the first instrument slope).
    A reflected basin within ~10 lp of the main one is a hard warning: the
    posterior is genuinely bimodal and single-basin results are truncated."""
    import arviz as az
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(data=data, chains=4, parallel_chains=4,
                       iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                       adapt_delta=0.95, seed=seed, show_progress=False)
    idata = az.from_cmdstanpy(fit)
    lp = fit.method_variables()["lp__"]                    # (draws, chains)
    beta0 = idata.posterior["beta"].values[..., 0]          # (chains, draws)
    lp_by_chain = lp.T                                      # (chains, draws)
    pos_mask = beta0.mean(axis=1) > 0
    out = {
        "chains_positive_orientation": int(pos_mask.sum()),
        "chains_negative_orientation": int((~pos_mask).sum()),
        "max_lp_positive": (round(float(lp_by_chain[pos_mask].max()), 1)
                            if pos_mask.any() else None),
        "max_lp_negative": (round(float(lp_by_chain[~pos_mask].max()), 1)
                            if (~pos_mask).any() else None),
    }
    if out["max_lp_positive"] is not None and out["max_lp_negative"] is not None:
        gap = out["max_lp_positive"] - out["max_lp_negative"]
        out["lp_gap_positive_minus_negative"] = round(float(gap), 1)
        out["bimodality_warning"] = bool(abs(gap) < 10)
    else:
        out["lp_gap_positive_minus_negative"] = None
        out["bimodality_warning"] = False
    return out

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

- participant response style: mean per-participant category-usage entropy
  and mean per-participant response range, against the model's additive
  normal-intercept account of rater differences.

Plus a GATED instrument-arm posterior predictive check on the fitted
(standardized) score scale, per (cell, family), with every model term
including the item effect: this is the measurement model that produces every
instrument-based claim, so its failure blocks them.

Gates (pre-committed): global ppps >= 0.01, min per-family ppp >= 0.005
(Bonferroni-flavored), category ppp >= 0.01, participant-style ppps >= 0.01,
in BOTH participant modes; instrument min ppp >= 0.005.
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

    part_col = human["participant_id"].to_numpy()

    def stats(y):
        df = pd.DataFrame({"y": y, "item": human["item_id"].to_numpy(),
                           "fam": fam_col, "part": part_col})
        item_means = df.groupby("item")["y"].mean()
        fam_of_item = df.groupby("item")["fam"].first()
        spread_by_fam = item_means.groupby(fam_of_item).std()
        within = df.groupby("item")["y"].std()
        within_by_fam = within.groupby(fam_of_item).mean()
        cats = np.bincount(y, minlength=K + 1)[1:] / len(y)

        def _entropy(v):
            pr = np.bincount(v, minlength=K + 1)[1:] / len(v)
            pr = pr[pr > 0]
            return float(-(pr * np.log(pr)).sum())
        by_part = df.groupby("part")["y"]
        part_entropy = float(by_part.apply(
            lambda v: _entropy(v.to_numpy())).mean())
        part_range = float((by_part.max() - by_part.min()).mean())
        return (spread_by_fam.reindex(fams).to_numpy(),
                float(within.mean()),
                within_by_fam.reindex(fams).to_numpy(),
                cats, part_entropy, part_range)

    (obs_spread, obs_within, obs_within_fam, obs_cats,
     obs_pent, obs_prange) = stats(y_obs)

    draws = rng.choice(L, size=min(n_sims, L), replace=True)
    report: dict = {"n_sims": len(draws), "families": fams}
    if posterior_sha256:
        report["posterior_sha256"] = posterior_sha256
    all_pass = True

    for mode in ("conditional", "marginal"):
        sp, wi, wif, cats_list, pent, prange = [], [], [], [], [], []
        for d in draws:
            if mode == "conditional":
                u_d = u[:, d]
            else:
                u_d = rng.normal(0.0, sigma_u[d], u.shape[0])
            y_sim = _simulate_ratings(theta[:, d], u_d, kappa[:, d],
                                      item_idx0, part_idx0, rng)
            a, b, c, k, pe, pr = stats(y_sim)
            sp.append(a); wi.append(b); wif.append(c); cats_list.append(k)
            pent.append(pe); prange.append(pr)
        sp = np.array(sp); wi = np.array(wi); wif = np.array(wif)
        cats_arr = np.array(cats_list)
        pent = np.array(pent); prange = np.array(prange)

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
            "participant_entropy_ppp": _ppp(obs_pent, pent),
            "participant_range_ppp": _ppp(obs_prange, prange),
        }
        from ..spec import load_spec
        gp = load_spec()["ppc"]
        mode_pass = (mode_report["within_item_sd_global_ppp"] >= gp["global_ppp_min"]
                     and mode_report["family_spread_ppp_min"] >= gp["family_ppp_min"]
                     and mode_report["within_item_sd_family_ppp_min"] >= gp["family_ppp_min"]
                     and mode_report["category_usage_ppp"] >= gp["global_ppp_min"]
                     and mode_report["participant_entropy_ppp"] >= gp["global_ppp_min"]
                     and mode_report["participant_range_ppp"] >= gp["global_ppp_min"])
        mode_report["passed"] = bool(mode_pass)
        report[mode] = mode_report
        all_pass = all_pass and mode_pass

    # instrument-arm posterior-predictive check, on the FITTED (standardized)
    # score scale, with every model term including the item effect and its
    # uncertainty: per (cell, family), ppp of the observed mean standardized
    # score against replicates simulated from the full posterior. Gated: this
    # is the measurement model that produces every instrument-based claim.
    if cont is not None and len(cont) and "beta" in post:
        std = maps.get("cell_standardization", {})
        cs = cont.copy()
        for cell, const in std.items():
            mrow = cs["cell_id"] == cell
            cs.loc[mrow, "value"] = (cs.loc[mrow, "value"] - const["mean"]) / const["sd"]
        beta_d = post["beta"].stack(d=("chain", "draw")).values
        alpha_d = post["alpha"].stack(d=("chain", "draw")).values
        a_dev_d = post["a_dev"].stack(d=("chain", "draw")).values
        b_dev_d = post["b_dev"].stack(d=("chain", "draw")).values
        gamma_d = post["gamma"].stack(d=("chain", "draw")).values
        sigma_d = post["sigma_s"].stack(d=("chain", "draw")).values
        omega_d = post["omega"].stack(d=("chain", "draw")).values
        e_d = post["e_raw"].stack(d=("chain", "draw")).values
        Xs = (np.asarray(maps["X_standardized"])
              if maps.get("X_standardized") else None)
        cix = {c: j for j, c in enumerate(maps["cont_cells"])}
        constr_of = {iid: fam_of[iid] for iid in maps["item_ids"]}
        fam_ix = {f: j for j, f in enumerate(maps["constructions"])}
        sub = rng.choice(theta.shape[1], size=min(400, theta.shape[1]),
                         replace=False)
        inst_ppps = {}
        for (cell, fam), grp in cs.assign(
                fam=cs["item_id"].map(constr_of)).groupby(["cell_id", "fam"]):
            m = cix.get(cell)
            if m is None:
                continue
            c = fam_ix[fam]
            ii = grp["item_id"].map(item_pos).to_numpy()
            obs_mean = float(grp["value"].mean())
            reps = np.empty(len(sub))
            for jj, d in enumerate(sub):
                mu = (alpha_d[m, d] + a_dev_d[m, c, d]
                      + (beta_d[m, d] + b_dev_d[m, c, d]) * theta[ii, d]
                      + omega_d[m, d] * e_d[m, ii, d])
                if Xs is not None:
                    mu = mu + Xs[ii] @ gamma_d[m, :, d]
                reps[jj] = float(np.mean(
                    mu + rng.normal(0.0, sigma_d[m, d], len(ii))))
            inst_ppps[f"{cell}|{fam}"] = _ppp(obs_mean, reps)
        report["instrument_ppc"] = {
            "cell_family_ppp": inst_ppps,
            "min_ppp": min(inst_ppps.values()) if inst_ppps else None,
            "passed": bool(inst_ppps and min(inst_ppps.values())
                           >= load_spec()["ppc"]["instrument_ppp_min"]),
        }
        all_pass = all_pass and report["instrument_ppc"]["passed"]
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
which claims does the evidence on disk license? Principles, enforced in code:

- Every number comes from a file actually read; missing or failed evidence
  refuses dependent claims, never grants them.
- Evidence binds by RECOMPUTED hashes (posterior, Stan source, exact
  input-hash maps); a swapped posterior, edited manifest, or stale report
  invalidates.
- Contamination vitiates every deployment tier, screening included.
- Claims are a matrix, not a ladder; aggregate estimation does not presuppose
  ranking and carries a sharpness gate.
- New-family quantities are sign-aware and labeled as model-based
  extrapolations conditional on family exchangeability.
- Absolute location of an unanchored family is prior-identified, so those
  claims are refused structurally.

Assurance structure adopted from the portfolio's AI-safety papers:

- A grant is a LICENCE WITH A LIFE-CYCLE (ai-evaluation-kinds): the
  certificate states what expires it, what erodes it (optimization against
  its own gate statistics), and what defeats or supersedes it, instead of
  pretending issuance is the end of the story.
- Each licensed claim carries the four projective-claim blocks (declaration,
  projectibility profile, warrant, defeaters), and the profile is the
  analyst's stated hypothesis, never fused with the warrant.
- Refusals are TYPED with remedies (evidentiary-assurance): an unevaluable
  record, an adequate record whose showing falls short, affirmative evidence
  of failure, a vitiated source, and a structural impossibility call for
  different responses, so the certificate distinguishes them.
- The certificate names an answerable issuer and a concrete contestation
  route (evidentiary-assurance's answerability and forum questions, scaled
  to a research artifact).
- Thresholds come from a frozen, versioned spec (thresholds.yaml), cited by
  version and hash (delegation-assurance's frozen-specification practice).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model.fit import STAN_FILE, sha256_file
from .spec import load_spec, spec_identity

TIERS = (
    "screening",
    "ranking_within_family",
    "family_location_unanchored",
    "aggregate_estimation",
    "effect_reproduction",
    "distributional_claims",
    "population_transfer",
    "individual_simulation",
    "mechanism_claims",
)

# refusal types, per the evidentiary-assurance verdict separation
UNEVALUABLE = "unevaluable"            # evidence missing or unbound
SHORTFALL = "shortfall"                # adequate evidence, gate not met
AFFIRMATIVE = "affirmative_failure"    # evidence of failure
VITIATED = "vitiated"                  # evidence bearing undermined (contamination)
STRUCTURAL = "structural"              # unsupportable in this design


def _refuse(reason: str, rtype: str, remedy: str) -> dict:
    return {"type": rtype, "reason": reason, "remedy": remedy}


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


def build_warrant(run_dir: str | Path, estimand: dict,
                  out_path: str | Path | None = None) -> dict:
    import arviz as az

    spec = load_spec()
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
    newfam = _read_json(run_dir / "newfam.json")
    loco = _read_json(run_dir / "loco.json")
    ppc = _read_json(run_dir / "ppc.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None
    provenance = _read_json(run_dir / "instruments.json")
    response_style = _read_json(run_dir / "response_style.json")

    # ---- recomputed binding: trust nothing declared, rehash what exists
    binding: dict[str, str] = {}
    posterior_sha = sha256_file(post_path)
    stan_sha = sha256_file(STAN_FILE)
    if run is None:
        binding["run"] = "run.json missing: nothing can be bound"
        bound_core = False
    else:
        checks = {
            "posterior": run.get("posterior_sha256") == posterior_sha,
            "stan": run.get("stan_sha256") == stan_sha,
        }
        binding["run"] = ("bound" if all(checks.values()) else
                          "MISMATCH: " + ", ".join(k for k, v in checks.items() if not v))
        bound_core = all(checks.values())

    def _bound_posterior(child: dict | None, name: str) -> bool:
        if child is None:
            binding[name] = "not produced"
            return False
        got = child.get("posterior_sha256")
        if got is None:
            binding[name] = "no posterior stamp"
            return False
        ok = got == posterior_sha
        binding[name] = "bound" if ok else "stamped for a different posterior"
        return ok

    def _bound_inputs(child: dict | None, name: str) -> bool:
        if child is None:
            binding[name] = "not produced"
            return False
        if run is None:
            binding[name] = "run.json missing"
            return False
        want = run.get("input_hashes") or {}
        got = child.get("input_hashes") or {}
        ok = bool(want) and got == want          # exact map equality
        if ok and child.get("stan_sha256") != stan_sha:
            binding[name] = "generated under a different model (stan hash differs)"
            return False
        binding[name] = ("bound" if ok else
                         "input-hash map differs from run.json (exact equality required)")
        return ok

    def _bound_stan(child: dict | None, name: str) -> bool:
        """Simulation-based evidence certifies the MODEL, not a posterior:
        it binds through the Stan source hash."""
        if child is None:
            binding[name] = "not produced"
            return False
        ok = child.get("stan_sha256") == stan_sha
        binding[name] = ("bound" if ok else
                         "generated for a different model (stan hash differs)")
        return ok

    ppc_bound = _bound_posterior(ppc, "ppc")
    loco_bound = _bound_inputs(loco, "loco")
    rec_bound = _bound_stan(recovery, "recovery")
    sbc_bound = _bound_stan(sbc, "sbc")
    newfam_bound = _bound_stan(newfam, "newfam")

    # ---- instruments: sign-aware new-family quantities gate; provenance merged
    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    gate_stats: dict[str, dict] = {}
    if cont_cells and "reliability_new_directional" in idata.posterior:
        rel_dir = np.asarray(idata.posterior["reliability_new_directional"].values)
        rel_dir = rel_dir.reshape(-1, rel_dir.shape[-1])
        p_pos = np.asarray(idata.posterior["p_pos_new"].values)
        p_pos = p_pos.reshape(-1, p_pos.shape[-1])
        rel_fam = np.asarray(idata.posterior["reliability_family"].values)
        rel_fam = rel_fam.reshape(-1, rel_fam.shape[-2], rel_fam.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel_dir[:, j]))
            q10, q90 = np.percentile(rel_dir[:, j], [10, 90])
            ppos = float(p_pos[:, j].mean())
            fam_meds = np.median(rel_fam[:, j, :], axis=0)
            gate_stats[name] = {"median": med, "q10": float(q10), "p_pos": ppos}
            entry = {
                "cell": name,
                "p_positive_slope_new_family": round(ppos, 3),
                "reliability_new_directional_median": round(med, 3),
                "reliability_new_directional_80": [round(float(q10), 3),
                                                   round(float(q90), 3)],
                "reliability_by_family_within": {
                    f: round(float(v), 3)
                    for f, v in zip(maps.get("constructions", []), fam_meds)
                },
                "alpha_standardized_median": round(float(np.median(alpha[:, j])), 3),
                "standardization": maps.get("cell_standardization", {}).get(
                    name, "none recorded"),
                "interpretation": ("model-based extrapolation conditional on "
                                   "exchangeability of construction families "
                                   "(purposive sample); within-family signal "
                                   "variance; reversed-slope draws count as "
                                   "zero signal"),
            }
            if provenance and name in provenance:
                entry["provenance"] = provenance[name]
            else:
                entry["provenance"] = "not recorded in run dir"
            instruments.append(entry)

    estimand = dict(estimand)
    split_half = estimand.pop("human_split_half", "not provided")
    issued_by = estimand.pop("issued_by", "not named")
    profiles = estimand.pop("profiles", {}) or {}
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
    flag_rate = spec["nonresponse_flag_rate"]
    flagged_cells = sorted(
        c for c, r in (nonresponse or {}).items()
        if isinstance(r, (int, float)) and r > flag_rate)

    inst_ppc = (ppc or {}).get("instrument_ppc") if ppc else None

    evidence = {
        "run": run if run is not None else "not produced",
        "binding": binding,
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "sbc": sbc if sbc is not None else "not produced",
        "new_family_recovery": newfam if newfam is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "ppc": ppc if ppc is not None else "not produced",
        "multiverse_spread": _multiverse_spread(run_dir),
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
    }

    licensed: dict[str, dict] = {}
    refused: dict[str, dict] = {}

    # ---- shared prerequisites for ANY instrument-based deployment claim
    diag_ok = bool(diagnostics and diagnostics.get("passed") is True)
    prereq: tuple[str, str, str] | None = None   # (reason, type, remedy)
    if not bound_core:
        prereq = (f"run binding failed ({binding['run']})", UNEVALUABLE,
                  "regenerate run.json alongside the posterior it describes")
    elif not diag_ok:
        prereq = ("diagnostics gate not passed", AFFIRMATIVE,
                  "fix sampling (iterations, adapt_delta, model) and refit")
    elif recovery is None or not rec_bound:
        prereq = (f"fake-data recovery: {binding['recovery']}", UNEVALUABLE,
                  "run recovery against the current model")
    elif recovery.get("passed") is not True:
        prereq = ("fake-data recovery failed", AFFIRMATIVE,
                  "the pipeline cannot recover known truth; fix before any claim")
    elif sbc is None or not sbc_bound:
        prereq = (f"SBC: {binding['sbc']}", UNEVALUABLE,
                  "run SBC against the current model")
    elif sbc.get("passed") is not True:
        prereq = ("SBC failed", AFFIRMATIVE,
                  "rank calibration or fit stability is broken; fix before any claim")
    elif newfam is None or not newfam_bound:
        prereq = (f"new-family simulated recovery: {binding['newfam']}",
                  UNEVALUABLE, "run newfam_check against the current model")
    elif newfam.get("passed") is not True:
        prereq = ("new-family simulated recovery failed", AFFIRMATIVE,
                  "the branch every new-family claim uses is miscalibrated")
    elif ppc is None or not ppc_bound:
        prereq = (f"PPC: {binding['ppc']}", UNEVALUABLE,
                  "run the PPC against this posterior")
    elif not ppc.get("passed"):
        prereq = ("PPC failed (human or instrument arm)", AFFIRMATIVE,
                  "the fitted model misfits the criterion data; revise the model")
    elif inst_ppc is None or not inst_ppc.get("passed"):
        prereq = ("instrument-arm posterior predictive check absent or failed",
                  AFFIRMATIVE if inst_ppc else UNEVALUABLE,
                  "the measurement model must fit before instrument claims")

    vitiation = _refuse(
        "contamination cap: item source not assessed clean; exposure to "
        "published items can inflate the fitted slope and stable instrument "
        "opinion exactly where this claim's statistic looks",
        VITIATED,
        "validate on a post-cutoff, unpublished item set with fresh norms")

    def _declaration(cell: str | None) -> dict:
        prov = "not recorded"
        if cell:
            for inst in instruments:
                if inst["cell"] == cell:
                    prov = inst.get("provenance", "not recorded")
        return {
            "bearer": cell or "instrument set",
            "unit": "item (decontextualized sentence)",
            "population": estimand.get("population"),
            "range": {"families": domain["construction_families"],
                      "item_source": domain.get("item_source", "see domain")},
            "time_version": {"posterior_sha256": posterior_sha,
                             "instrument_provenance": prov},
        }

    DEFEATERS = [
        "contamination discovered for the item source (vitiates retroactively)",
        "instrument version change (weights, digest, or serving stack)",
        "item-pool or criterion-data change (hashes in run.json)",
        "a failed re-validation or stronger contrary result (defeats)",
        "optimization of prompts, models, or items against this certificate's "
        "gate statistics (Goodhart erosion; vitiates the statistic)",
    ]

    def _grant(claim: str, cell: str | None, basis: str) -> None:
        licensed[claim] = {
            "basis": basis + f" [spec {spec['spec_version']}]",
            "declaration": _declaration(cell),
            "projectibility_profile": profiles.get(
                claim,
                "not stated by the analyst: without a stated profile (the "
                "worldly pattern that would make this inference "
                "non-accidental) the grant is evidence-only"),
            "defeaters": DEFEATERS,
        }

    # ---- screening
    g = spec["screening"]
    candidates = {c: v for c, v in gate_stats.items() if c not in flagged_cells}
    best = max(candidates.items(), key=lambda kv: kv[1]["median"]) if candidates else None
    if prereq:
        refused["screening"] = _refuse(*prereq)
    elif not contamination_clean:
        refused["screening"] = vitiation
    elif best is None:
        refused["screening"] = _refuse(
            "no unflagged continuous cell carries the directional new-family "
            "quantities", UNEVALUABLE, "fit at least one unflagged instrument cell")
    elif (best[1]["p_pos"] >= g["p_pos_min"] and best[1]["median"] > g["rel_median_min"]
          and best[1]["q10"] > g["rel_q10_min"]):
        _grant("screening", best[0],
               f"prerequisites passed; cell {best[0]}: P(new-family slope "
               f"positive) {best[1]['p_pos']:.2f} >= {g['p_pos_min']}, "
               f"directional reliability median {best[1]['median']:.2f} > "
               f"{g['rel_median_min']}, q10 {best[1]['q10']:.2f} > {g['rel_q10_min']}")
    else:
        refused["screening"] = _refuse(
            f"best unflagged cell {best[0]}: P(positive slope) "
            f"{best[1]['p_pos']:.2f}, directional reliability median "
            f"{best[1]['median']:.2f}, q10 {best[1]['q10']:.2f} (need >= "
            f"{g['p_pos_min']}, > {g['rel_median_min']}, > {g['rel_q10_min']})",
            SHORTFALL, "stronger instrument or more criterion data")

    # ---- within-family ranking
    g = spec["ranking_within_family"]
    all_fams = set(maps.get("constructions", []))
    if prereq:
        refused["ranking_within_family"] = _refuse(*prereq)
    elif loco is None:
        refused["ranking_within_family"] = _refuse(
            "loco.json missing", UNEVALUABLE, "run LOCO for this run's inputs")
    elif not loco_bound:
        refused["ranking_within_family"] = _refuse(
            f"LOCO not bound: {binding['loco']}", UNEVALUABLE,
            "re-run LOCO against the current inputs")
    elif not loco.get("all_diagnostics_passed"):
        refused["ranking_within_family"] = _refuse(
            "one or more LOCO fold fits failed diagnostics", AFFIRMATIVE,
            "raise fold iterations or fix the model, then re-run LOCO")
    elif set(loco.get("families_tested", [])) != all_fams:
        missing = sorted(all_fams - set(loco.get("families_tested", [])))
        refused["ranking_within_family"] = _refuse(
            f"LOCO did not cover every family (missing: {missing})",
            UNEVALUABLE, "run the missing folds")
    elif not contamination_clean:
        refused["ranking_within_family"] = vitiation
    elif ((loco.get("mean_spearman") or -1) > g["mean_spearman_min"]
          and (loco.get("frac_families_within_spearman_gt_0.3") or 0)
          >= g["frac_families_gt_0p3_min"]):
        _grant("ranking_within_family", None,
               f"mean within-family held-out Spearman "
               f"{loco['mean_spearman']:.2f} > {g['mean_spearman_min']} and "
               f"{loco['frac_families_within_spearman_gt_0.3']:.0%} of "
               f"families > 0.3 (min {loco.get('within_family_spearman_min')})")
    else:
        refused["ranking_within_family"] = _refuse(
            f"within-family transfer insufficient: mean Spearman "
            f"{loco.get('mean_spearman')}, "
            f"{(loco.get('frac_families_within_spearman_gt_0.3') or 0):.0%} "
            f"of families > 0.3, min {loco.get('within_family_spearman_min')}",
            SHORTFALL, "stronger instrument, more families, or more items per family")

    # ---- family location for unanchored families: structurally refused
    refused["family_location_unanchored"] = _refuse(
        "absolute location of a family with no human anchor is "
        "prior-identified, not likelihood-identified (delta-invariance "
        "against per-cell family intercepts)",
        STRUCTURAL, "collect human anchor items in each new family, or accept "
        "within-family claims only")

    # ---- aggregate estimation (independent of ranking; sharpness-aware)
    g = spec["aggregate_estimation"]
    coverage = loco.get("mean_coverage90") if loco else None
    rmse = loco.get("mean_rmse") if loco else None
    sd_obs = loco.get("sd_observed_item_means") if loco else None
    lo_band, hi_band = g["coverage_band"]
    if prereq:
        refused["aggregate_estimation"] = _refuse(*prereq)
    elif loco is None or not loco_bound or not loco.get("all_diagnostics_passed"):
        refused["aggregate_estimation"] = _refuse(
            "LOCO evidence missing, unbound, or diagnostics-failed",
            UNEVALUABLE, "produce clean, bound LOCO evidence")
    elif not contamination_clean:
        refused["aggregate_estimation"] = vitiation
    elif coverage is None or rmse is None or sd_obs is None:
        refused["aggregate_estimation"] = _refuse(
            "LOCO carries no coverage/RMSE/spread for the sharpness-aware gate",
            UNEVALUABLE, "re-run LOCO with the current report schema")
    elif lo_band <= coverage <= hi_band and rmse <= g["rmse_max_frac_of_observed_sd"] * sd_obs:
        _grant("aggregate_estimation", None,
               f"coverage {coverage:.2f} in [{lo_band}, {hi_band}] and RMSE "
               f"{rmse:.2f} <= {g['rmse_max_frac_of_observed_sd']} x "
               f"sd(observed item means) = "
               f"{g['rmse_max_frac_of_observed_sd'] * sd_obs:.2f}; "
               "within-family interpretation only")
    else:
        refused["aggregate_estimation"] = _refuse(
            f"coverage {coverage:.2f} (need [{lo_band}, {hi_band}]) with RMSE "
            f"{rmse:.2f} vs sharpness bound "
            f"{g['rmse_max_frac_of_observed_sd'] * sd_obs:.2f}: wide-interval "
            "coverage without accuracy does not license aggregate use",
            SHORTFALL, "sharper instrument or more criterion data")

    refused["effect_reproduction"] = _refuse(
        "not yet tested: requires matched experimental contrasts",
        UNEVALUABLE, "add a matched-contrast validation stage (v2)")
    refused["distributional_claims"] = _refuse(
        "no participant-level validation of variance structure"
        + ("" if ppc is None or ppc.get("passed")
           else "; posterior predictive checks failed"),
        UNEVALUABLE if ppc is None or ppc.get("passed") else AFFIRMATIVE,
        "model and validate rater heterogeneity (v2)")
    refused["population_transfer"] = _refuse(
        "the v1 ladder contains no population-transfer test; the estimand "
        "population defaults to the criterion sample's own",
        UNEVALUABLE, "add stratified human anchors from the target population (v2)")
    refused["individual_simulation"] = _refuse(
        "an item-level instrument licenses no individual-level human simulation",
        STRUCTURAL, "none: outside this design's claim space")
    refused["mechanism_claims"] = _refuse(
        "the model estimates a linking function, not a mechanism",
        STRUCTURAL, "none: outside this design's claim space")

    # ---- descriptive findings: always emitted, never a license
    descriptive = {
        "note": ("in-source descriptive associations on this item set; not "
                 "deployment evidence"),
        "pooled_heldout_spearman": (loco or {}).get("pooled_spearman_descriptive"),
        "between_family_spearman": (loco or {}).get("between_family_spearman"),
        "per_family_within_spearman": {
            f: v.get("spearman") for f, v in ((loco or {}).get("per_family") or {}).items()
        } or "no LOCO evidence",
        "human_split_half": split_half,
        "response_style": (response_style if response_style is not None
                           else "not assessed (no response_style.json)"),
    }

    # ---- the licence life-cycle block (ai-evaluation-kinds)
    licence = {
        "status": "issued" if licensed else "descriptive_only",
        "issued_by": issued_by,
        "issued": datetime.date.today().isoformat(),
        "threshold_spec": spec_identity(),
        "expiry_conditions": [
            "any instrument version change (checkpoint commit, digest, or "
            "serving stack) relative to the recorded provenance",
            "any change to the item pool, criterion data, or measurements "
            "(hashes in run.json)",
            "any change to the Stan model (stan_sha256)",
        ],
        "erosion_conditions": [
            "selecting prompts, models, or items against this certificate's "
            "gate statistics vitiates those statistics (Goodhart)",
            "repeated selective re-running until a gate passes",
        ],
        "defeat_and_supersession": (
            "a later certificate built for this run_dir supersedes this one; "
            "a failed re-validation defeats all outstanding grants"),
        "contestation": (
            "re-run scripts/pilot.py and rebuild the warrant, or submit the "
            "current validation packet (reviews/) to an independent reviewer; "
            "disagreement with a gate is a spec revision, not an edit"),
    }

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_id": (run or {}).get("run_id", "not recorded"),
        "posterior_sha256_recomputed": posterior_sha,
        "stan_sha256_recomputed": stan_sha,
        "licence": licence,
        "estimand": estimand,
        "domain": domain,
        "instruments": instruments,
        "evidence": evidence,
        "licensed_claims": licensed,
        "refused_claims": refused,
        "descriptive_findings": descriptive,
        "residual_risks": [
            "binding integrity is not construct validity: a pristine, fully "
            "bound record can still document the wrong predicate; hash checks "
            "discharge staleness, not meaning",
            "shared pretraining bias: local instruments share web-scale "
            "training data and can share construction-specific error; a tight "
            "multiverse fan does not rule this out",
            "reliability quantities are fit- and item-set-specific and "
            "conditional on a family-exchangeability extrapolation model",
            "thresholds are pre-registered defaults, not estimand-specific "
            "loss analyses",
            "prompt-level dependence within pooled paraphrase cells is not "
            "yet modeled (nested prompt effects are v2)",
        ],
    })

    out_path = Path(out_path) if out_path else run_dir / "warrant.yaml"
    out_path.write_text(yaml.safe_dump(cert, sort_keys=False, allow_unicode=True))
    return cert

```


---

## FILE: scripts/pilot.py

```
"""Run the Sprouse-LI pilot end to end against the current model.

Usage: uv run python scripts/pilot.py [--skip-loco]

Every warrant prerequisite is produced (or verified) here, and the script
STOPS at the first failed gate rather than proceeding to later stages: a
partially-run directory must not look like a validated one. --skip-loco
deletes any stale loco.json for the same reason.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.items import load_items, nuisance_covariates
from acceptometer.elicit.base import load_measurements, CONTINUOUS
from acceptometer.model.fit import (build_stan_data, fit_model, diagnostics_gate,
                                    save_fit, sha256_file, STAN_FILE)
from acceptometer.model.simulate import (simulate, recovery_check, write_report,
                                         newfam_check)
from acceptometer.model.sbc import sbc_run
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


def write_provenance(ms, out: Path) -> None:
    """Collapse measurement metadata into per-fitted-cell provenance so the
    certificate can name the model installation it certifies."""
    prov: dict = {}
    for m in ms:
        cell = m.cell_id
        cell = cell.rsplit("/p", 1)[0] if "/prompt_" in cell else cell
        entry = prov.setdefault(cell, {"revisions": set(), "digests": set(),
                                       "dates": set()})
        if m.meta.get("revision"):
            entry["revisions"].add(m.meta["revision"])
        if m.meta.get("model_digest"):
            entry["digests"].add(str(m.meta["model_digest"])[:200])
        if m.meta.get("date"):
            entry["dates"].add(m.meta["date"][:10])
    out.write_text(json.dumps(
        {c: {k: sorted(v) for k, v in e.items()} for c, e in prov.items()},
        indent=2))


def fail(stage: str) -> int:
    print(f"STOP: {stage} failed; later stages not run")
    return 1


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
    write_provenance(ms, RUN / "instruments.json")

    # ladder step 1: fake-data recovery against the CURRENT model
    sdata, truth = simulate()
    sfit, sidata = fit_model(sdata, seed=7)
    srec = recovery_check(sidata, truth)
    srec["diagnostics"] = diagnostics_gate(sfit, sidata)
    srec["stan_sha256"] = sha256_file(STAN_FILE)
    write_report(srec, RUN / "recovery.json")
    print("recovery passed:", srec["passed"])
    if not srec["passed"]:
        return fail("recovery")

    # ladder step 2: SBC (reuse a current run only if it matches this model)
    sbc_path = RUN / "sbc.json"
    sbc = json.loads(sbc_path.read_text()) if sbc_path.exists() else None
    if not (sbc and sbc.get("stan_sha256") == srec["stan_sha256"]):
        print("running SBC (no current-model report found)...")
        sbc = sbc_run(R=100, seed=17, out_path=sbc_path)
    print("SBC passed:", sbc["passed"], "| failed fits:",
          sbc.get("n_failed_fits"), "+", sbc.get("n_diag_failed"), "diag")
    if not sbc["passed"]:
        return fail("SBC")

    # ladder step 2b: simulated new-family recovery (the branch the warrant uses)
    nf = newfam_check(iter_warmup=2000, iter_sampling=2000)
    (RUN / "newfam.json").write_text(json.dumps(nf, indent=2))
    print("new-family recovery passed:", nf["passed"])
    if not nf["passed"]:
        return fail("new-family recovery")

    # real fit
    data, maps = build_stan_data(items, X, human, cont, None, K=7)
    fit, idata = fit_model(data, seed=42, iter_warmup=1500, iter_sampling=1500)
    diag = diagnostics_gate(fit, idata)
    save_fit(idata, maps, diag, RUN, data=data, input_hashes=input_hashes)
    cont.to_csv(RUN / "cont.csv", index=False)
    print("fit diag:", json.dumps(diag))
    if not diag["passed"]:
        return fail("fit diagnostics")

    # auditability: auxiliary posterior summaries the report quotes
    post = idata.posterior
    hm = human.groupby("item_id").rating.mean()
    q = cont[cont.cell_id == "qwen3:8b/prompt_scalar"].groupby("item_id").value.mean()
    common = q.index.intersection(hm.index)
    summary = {
        "tau_item_median": round(float(post.tau_item.median()), 3),
        "omega_median_by_cell": {
            c: round(float(post.omega.median(dim=("chain", "draw")).values[j]), 3)
            for j, c in enumerate(maps["cont_cells"])},
        "qwen_pooled_scalar_item_mean_r_with_human": round(
            float(np.corrcoef(q[common], hm[common])[0, 1]), 3),
        "posterior_sha256": json.loads((RUN / "run.json").read_text())["posterior_sha256"],
    }
    (RUN / "posterior_summary.json").write_text(json.dumps(summary, indent=2))

    run = json.loads((RUN / "run.json").read_text())
    rep = ppc_human(idata, maps, human, items, out_path=RUN / "ppc.json",
                    cont=cont, posterior_sha256=run["posterior_sha256"])
    print("PPC passed:", rep["passed"],
          "| marginal:", rep["marginal"]["passed"],
          "| conditional:", rep["conditional"]["passed"],
          "| instrument:", rep.get("instrument_ppc", {}).get("passed"))
    if not rep["passed"]:
        return fail("PPC")

    if args.skip_loco:
        stale = RUN / "loco.json"
        if stale.exists():
            stale.unlink()
            print("removed stale loco.json (--skip-loco)")
        return 0

    # held-out-family geometry mixes slowly; 2000/2000 keeps fold fits
    # inside the production diagnostics gate
    lrep = loco(items, X, human, cont, None, K=7,
                iter_warmup=2000, iter_sampling=2000,
                out_path=RUN / "loco.json", input_hashes=input_hashes)
    print("LOCO:", json.dumps({k: v for k, v in lrep.items()
                               if k not in ("per_family", "input_hashes")}))
    return 0 if lrep["all_diagnostics_passed"] else fail("LOCO fold diagnostics")


if __name__ == "__main__":
    raise SystemExit(main())

```


---

## FILE: scripts/agy_judge.py

```
"""Batched acceptability judgments from Claude Opus 4.6 via the agy CLI.

The agy quota is account-wide with multi-day resets, so this instrument uses
a BATCHED protocol: 20 numbered sentences per call, strict-JSON ratings, 3
passes with per-pass shuffling (so batch context varies across repeats).
Batching is a protocol difference from the single-item Ollama cells and is
recorded in the cell id and metadata: the protocol is part of the instrument.

Writes into runs/multi/measurements.jsonl with (item, cell, repeat) dedup.
Unparseable chunks are retried once, then recorded as failures; no values are
ever fabricated.
"""

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from acceptometer.items import load_items
from acceptometer.elicit.base import CONTINUOUS, Measurement, load_measurements

MODEL = "Claude Opus 4.6 (Thinking)"
CELL = "claude-opus-4.6-agy/prompt_scalar_batch20"
OUT = Path("runs/multi/measurements.jsonl")
BATCH = 20
REPEATS = 3

PROMPT = """You are rating English sentences for acceptability to a native speaker, \
on a scale from 1 (completely unacceptable) to 7 (completely natural). Rate each \
sentence independently.

Reply with ONLY a JSON object mapping each sentence number to an integer rating, \
like {{"1": 5, "2": 2}}. No other text.

Sentences:
{sentences}"""


def call_agy(prompt: str) -> str:
    r = subprocess.run(
        ["agy", "--model", MODEL, "--dangerously-skip-permissions", "-p", prompt],
        capture_output=True, text=True, timeout=420)
    return r.stdout + "\n" + r.stderr


def parse_ratings(text: str, n: int) -> dict | None:
    for m in reversed(re.findall(r"\{[^{}]+\}", text, flags=re.S)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        out = {}
        for k, v in d.items():
            try:
                k, v = int(k), int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= n and 1 <= v <= 7:
                out[k] = float(v)
        if out:
            return out
    return None


def main() -> int:
    items = load_items("data/pilot_items.jsonl")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if OUT.exists():
        for m in load_measurements(OUT):
            have.add((m.item_id, m.cell_id, m.repeat))

    now = datetime.now(timezone.utc).isoformat()
    prompt_sha = sha256(PROMPT.encode()).hexdigest()[:16]
    n_ok, failures = 0, []
    for rep in range(REPEATS):
        order = np.random.default_rng(rep).permutation(len(items))
        chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        for ci, chunk in enumerate(chunks):
            todo = [items[j] for j in chunk
                    if (items[j].item_id, CELL, rep) not in have]
            if not todo:
                continue
            listing = "\n".join(f"{k + 1}. {it.text}" for k, it in enumerate(todo))
            prompt = PROMPT.format(sentences=listing)
            ratings = None
            for attempt in range(2):
                try:
                    raw = call_agy(prompt)
                except subprocess.TimeoutExpired:
                    raw = ""
                ratings = parse_ratings(raw, len(todo))
                if ratings:
                    break
            if not ratings:
                failures.append({"repeat": rep, "chunk": ci, "n": len(todo)})
                print(f"rep {rep} chunk {ci}: PARSE FAILURE", flush=True)
                continue
            ms = []
            for k, it in enumerate(todo):
                if (k + 1) in ratings:
                    ms.append(Measurement(
                        item_id=it.item_id, cell_id=CELL, kind=CONTINUOUS,
                        value=ratings[k + 1], repeat=rep,
                        meta={"provider": "antigravity/agy", "model": MODEL,
                              "batch_size": BATCH, "protocol": "batched",
                              "prompt_sha": prompt_sha, "date": now}))
            with open(OUT, "a", encoding="utf-8") as fh:
                for m in ms:
                    fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
            n_ok += len(ms)
            print(f"rep {rep} chunk {ci}: {len(ms)}/{len(todo)} ratings",
                  flush=True)
            time.sleep(2)
    print(f"TOTAL: {n_ok} measurements, {len(failures)} failed chunks")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

```


---

## FILE: scripts/response_evidence.py

```
"""Response-style and marginal-stratum evidence (adversarial-pragmatics
lessons applied to acceptability).

Two failure modes that aggregate correlations conceal:

- MARGINAL-STRATUM performance: the contested middle band of items (human
  mean in [3, 5] on a 7-point scale) is where theoretical decisions live and
  where restriction-of-range inflation does not reach. Per-cell correlations
  are reported inside and outside the band.
- CATEGORY OMISSION: an instrument can reach an aggregate advantage partly by
  never emitting certain response categories (the adversarial-pragmatics
  judge study's label-omission finding). Human vs instrument category usage
  is compared directly for prompted scalar cells.

Writes runs/pilot/response_style.json; the warrant surfaces it under
descriptive_findings.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.elicit.base import load_measurements, CONTINUOUS

RUN = Path("runs/pilot")


def main() -> int:
    human = pd.read_csv("data/pilot_human.csv")
    hm = human.groupby("item_id").rating.mean()
    ms = load_measurements(RUN / "measurements.jsonl")
    multi = Path("runs/multi/measurements.jsonl")
    if multi.exists():
        ms += load_measurements(multi)
    df = pd.DataFrame([{"item_id": m.item_id, "cell_id": m.cell_id,
                        "kind": m.kind, "value": m.value} for m in ms])

    out: dict = {"marginal_band": [3.0, 5.0], "cells": {}}
    cont = df[df.kind == CONTINUOUS].copy()
    cont["cell_id"] = cont.cell_id.str.replace(r"/p\d$", "", regex=True)
    for cell, grp in cont.groupby("cell_id"):
        v = grp.groupby("item_id").value.mean()
        common = v.index.intersection(hm.index)
        h = hm[common]
        v = v[common]
        marg = (h >= 3.0) & (h <= 5.0)
        entry = {"n_items": int(len(common)),
                 "r_all": round(float(np.corrcoef(v, h)[0, 1]), 3)}
        for label, mask in (("r_marginal_band", marg), ("r_outside_band", ~marg)):
            if mask.sum() >= 5 and h[mask].std() > 0 and v[mask].std() > 0:
                entry[label] = round(float(np.corrcoef(v[mask], h[mask])[0, 1]), 3)
                entry[label + "_n"] = int(mask.sum())
            else:
                entry[label] = "not computable (too few items or no variance)"
        out["cells"][cell] = entry

    # category usage: human vs prompted scalar (raw 1-7 draws, all repeats)
    hu_usage = (human.rating.value_counts(normalize=True)
                .reindex(range(1, 8), fill_value=0.0))
    out["category_usage"] = {
        "human": {str(k): round(float(p), 3) for k, p in hu_usage.items()}}
    scal = df[df.cell_id.str.contains("prompt_scalar")]  # includes batch cells
    for cell, grp in scal.groupby(scal.cell_id.str.replace(r"/p\d$", "", regex=True)):
        usage = (grp.value.astype(int).value_counts(normalize=True)
                 .reindex(range(1, 8), fill_value=0.0))
        omitted = [k for k, p in usage.items() if p == 0.0]
        out["category_usage"][cell] = {
            "usage": {str(k): round(float(p), 3) for k, p in usage.items()},
            "categories_never_emitted": omitted or "none",
        }

    (RUN / "response_style.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```


---

## FILE: PILOT-REPORT.md

```
# Acceptometer pilot report: Sprouse LI items, six judge models
<!-- SUMMARY: Final pilot: certificate descriptive_only (contamination + rater-entropy PPC), cross-model contested-band deficit measured across five families incl. Opus 4.6 · status: complete · updated: 2026-08-27 -->

## What ran

120 items from Sprouse, Schütze & Almeida (2013) (10 source-paper families,
6 starred / 6 good each) with all 1,519 participant-level 7-point ratings
(304 participants). Judges: Pythia-160m (exact log-probabilities), qwen3:8b,
gemma3:12b, mistral-small:24b (local, single-item protocol, 3 paraphrases x 3
repeats), and Claude Opus 4.6 via agy (batched-20 protocol, 3 passes, 18
calls, 0 parse failures); glm-4.7-flash and qwen3.8:27b in flight. The joint
Bayesian fit uses qwen3:8b + Pythia SLOR; the others are descriptive cells.

The full ladder ran gated, stop-at-first-failure: fake-data recovery PASSED;
SBC (R=100, 4-chain, diagnostics-aware, failures excluded from ranks) PASSED;
simulated new-family recovery PASSED (within-family coverage .83, rank
recovery +.84/+.85, family-location bias reported as the empirical face of
the prior-identification limit); fit diagnostics PASSED; posterior predictive
checks FAILED on exactly one discrepancy: participant-level category-usage
entropy (ppp .000/.006), the rater-style misfit review 3 predicted, caught by
the check built for it. LOCO ran under the current model (held-out families
outside the sum-to-zero vector): mean within-family Spearman .687, 90%
coverage .926, between-family Spearman .41, fox still failing outright
(-.03), one fold marginal (R-hat 1.013).

## The certificate (runs/pilot/warrant.yaml)

**Licence status: `descriptive_only`.** All six evidence artifacts recompute-
hash-bound. Every deployment claim refused, typed, with a remedy:

| claim | type | remedy |
|---|---|---|
| screening | affirmative_failure (PPC) | model rater-style heterogeneity; then the contamination cap still requires post-cutoff items |
| ranking_within_family | affirmative_failure (PPC) | same |
| aggregate_estimation | affirmative_failure (PPC) | same |
| family_location_unanchored | structural | anchor items per new family, or within-family claims only |
| effect_reproduction, population_transfer | unevaluable | build the v2 tests |
| distributional_claims | affirmative_failure | rater-style modeling (v2) |
| individual_simulation, mechanism_claims | structural | none: outside the design's claim space |

The certificate carries the licence life-cycle (expiry on instrument/item/
model change; Goodhart-erosion clause; defeat and supersession; contestation
route), the analyst's projectibility profiles as stated hypotheses, and the
frozen threshold spec v1.0.0 by hash.

## The headline scientific finding: the contested-band deficit

Correlations of judge item-means with human item-means, split at the
contested band (human mean in [3, 5]; 47 of 120 items):

| judge | r all | r band | r outside |
|---|---|---|---|
| Claude Opus 4.6 (batched) | **.834** | **.397** | .935 |
| mistral-small 24B | .79 | .24 | .87 |
| gemma3 12B | .74 | .28 | .85 |
| qwen3 8B | .72 | .18 | .85 |
| Pythia-160m SLOR | .35 | -.18 | .51 |
| *human split-half ceiling* | *.857* | | |

Aggregate performance rises with capability to ceiling-adjacent (Opus .834
vs .857), and the band lags everywhere: the frontier model carries less than
half its outside-band signal in the region where acceptability theory
actually needs judgment. Opus also avoids the middle category harder than
humans do (uses "4" 5% vs 13.5%) while hitting the extremes. The deficit is
class-wide and scale-mitigated, not solved. Published aggregate correlations
(r~.8) are real and are carried by the items nobody needed an instrument for.

Consequent use profile: triage. Screen the clear cases, spend the human
budget on the middle band (which is what design.py allocates); no marginal-
item claim is supported for any tested judge.

## What the day's three external reviews and the gates changed

Review 1 (glm-4.7-flash): family-varying linking, PPC gate, contamination
caps, observed-mean LOCO target. Real data then forced: freed latent scale
(tau_item ~ 2.0), one-instrument-per-model+method, reflection-mode handling,
instrument-by-item error (reliability .90 -> .54 honest). Review 2
(GPT-family): new-family effects outside the zero-sum vector, hash binding,
predictive reliability, tie-aware pooled statistics. Review 3 (GPT-family):
sign-aware directional reliability, contamination vitiates screening too,
recomputed binding with model hashes, simulated new-family recovery,
rebuilt gated instrument PPC, claim matrix with sharpness gate, structural
refusal of unanchored family location. Assurance layer folded in from the
AI-safety papers: licence life-cycle, projectibility profiles, typed
refusals, frozen threshold spec, answerability. Full trail: DECISIONS.md.

## Caveats

One language, one contamination-suspect item source, constructed sentences,
one register, one date; Opus used a batched protocol (recorded in its cell
identity). The certificate lists untested axes. v2 priorities: post-cutoff
item set (the only route to any deployment grant), rater-style modeling
(clears the PPC), ordered-logistic scalar arm, binary overdispersion,
per-tier PPC consequence map, population-transfer test.

## Artifacts

`runs/pilot/`: warrant.yaml, run.json, recovery.json, sbc.json, newfam.json,
ppc.json, loco.json, posterior.nc, posterior_summary.json,
response_style.json, instruments.json, estimand.yaml, plots.
`runs/multi/measurements.jsonl`: cross-model judgments. Provenance:
`data/MANIFEST.yaml`. Pipeline: `scripts/pilot.py`; cross-model evidence:
`scripts/response_evidence.py`, `scripts/agy_judge.py`.

```


---

## EVIDENCE

### warrant.yaml (the emitted certificate)

```yaml
generated: '2026-08-27T21:53:04'
run_dir: runs/pilot
run_id: 6068b87c-5576-4a69-bebe-e6cc8f3cb75a
posterior_sha256_recomputed: 06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723
stan_sha256_recomputed: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
licence:
  status: descriptive_only
  issued_by: Brett Reynolds (analysis executed by Claude Fable 5, 2026-08-27 session;
    contestation route in the licence block)
  issued: '2026-08-27'
  threshold_spec:
    spec_version: 1.0.0
    spec_sha256: 2388075641560db3f33f509ece14f4ca31a1875933673518ff15b5cdc14640d3
  expiry_conditions:
  - any instrument version change (checkpoint commit, digest, or serving stack) relative
    to the recorded provenance
  - any change to the item pool, criterion data, or measurements (hashes in run.json)
  - any change to the Stan model (stan_sha256)
  erosion_conditions:
  - selecting prompts, models, or items against this certificate's gate statistics
    vitiates those statistics (Goodhart)
  - repeated selective re-running until a gate passes
  defeat_and_supersession: a later certificate built for this run_dir supersedes this
    one; a failed re-validation defeats all outstanding grants
  contestation: re-run scripts/pilot.py and rebuild the warrant, or submit the current
    validation packet (reviews/) to an independent reviewer; disagreement with a gate
    is a spec revision, not an edit
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
instruments: []
evidence:
  run:
    run_id: 6068b87c-5576-4a69-bebe-e6cc8f3cb75a
    posterior_sha256: 06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
    code_commit: 62191e84ff1b9e468dad770a428efcdc3d24d235
    n_items_criterion: 120
    n_items_total: 120
    input_hashes:
      items: f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c
      human: 5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec
      measurements: 1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a
  binding:
    run: bound
    ppc: bound
    loco: bound
    recovery: bound
    sbc: bound
    newfam: bound
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
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
  sbc:
    R: 100
    n_failed_fits: 0
    n_diag_failed: 8
    n_thin: 63
    params:
      tau_constr:
        chi2: 3.72
        p_uniform_approx: 0.717
        rank_hist:
        - 18
        - 11
        - 11
        - 11
        - 15
        - 15
        - 11
      tau_item:
        chi2: 3.11
        p_uniform_approx: 0.7968
        rank_hist:
        - 12
        - 15
        - 10
        - 18
        - 12
        - 13
        - 12
      sigma_u:
        chi2: 3.11
        p_uniform_approx: 0.7968
        rank_hist:
        - 11
        - 9
        - 17
        - 15
        - 14
        - 13
        - 13
      beta[1]:
        chi2: 5.24
        p_uniform_approx: 0.5148
        rank_hist:
        - 14
        - 16
        - 14
        - 9
        - 8
        - 17
        - 14
      beta[2]:
        chi2: 2.5
        p_uniform_approx: 0.8692
        rank_hist:
        - 13
        - 11
        - 13
        - 15
        - 10
        - 17
        - 13
      sigma_s[1]:
        chi2: 13.3
        p_uniform_approx: 0.0382
        rank_hist:
        - 9
        - 12
        - 13
        - 8
        - 19
        - 22
        - 9
      sigma_s[2]:
        chi2: 10.41
        p_uniform_approx: 0.1074
        rank_hist:
        - 21
        - 10
        - 19
        - 11
        - 11
        - 9
        - 11
      tau_b[1]:
        chi2: 3.87
        p_uniform_approx: 0.6965
        rank_hist:
        - 15
        - 9
        - 15
        - 14
        - 14
        - 16
        - 9
      tau_b[2]:
        chi2: 2.96
        p_uniform_approx: 0.8158
        rank_hist:
        - 12
        - 18
        - 14
        - 10
        - 14
        - 12
        - 12
      omega[1]:
        chi2: 3.57
        p_uniform_approx: 0.7374
        rank_hist:
        - 18
        - 11
        - 13
        - 14
        - 11
        - 10
        - 15
      omega[2]:
        chi2: 6.0
        p_uniform_approx: 0.4237
        rank_hist:
        - 17
        - 10
        - 9
        - 9
        - 15
        - 16
        - 16
    failed_fit_truths:
    - tau_constr: 0.81
      tau_item: 3.49
      sigma_u: 0.64
    - tau_constr: 0.09
      tau_item: 1.44
      sigma_u: 0.92
    - tau_constr: 1.18
      tau_item: 3.01
      sigma_u: 0.32
    - tau_constr: 0.65
      tau_item: 0.08
      sigma_u: 0.62
    - tau_constr: 1.15
      tau_item: 3.56
      sigma_u: 0.8
    - tau_constr: 1.09
      tau_item: 0.61
      sigma_u: 0.34
    - tau_constr: 1.96
      tau_item: 0.31
      sigma_u: 2.02
    - tau_constr: 0.3
      tau_item: 0.17
      sigma_u: 1.07
    passed: true
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
  new_family_recovery:
    n_new_families: 2
    diagnostics:
      divergences: 0
      divergence_rate: 0.0
      rhat_max: 1.0039
      ess_bulk_min: 1054
      passed: true
    per_family:
      fam7:
        theta_within_coverage90: 0.833
        within_rank_corr_truth: 0.839
        family_location_bias: 0.175
      fam8:
        theta_within_coverage90: 0.833
        within_rank_corr_truth: 0.853
        family_location_bias: -0.346
    mean_within_coverage90: 0.833
    passed: true
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
    note: 'family_location_bias is reported unGated: absolute location of an unanchored
      family is prior-identified (delta-invariance with per-cell family intercepts)'
  loco_transfer:
    target: same participants, new items
    per_family:
      32.1.martin:
        n_items: 12
        rmse: 0.849
        mean_signed_error: -0.462
        spearman: 0.884
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0053
          ess_bulk_min: 941
          passed: true
      32.3.Culicover:
        n_items: 12
        rmse: 0.866
        mean_signed_error: -0.008
        spearman: 0.546
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0037
          ess_bulk_min: 1235
          passed: true
      32.3.fanselow:
        n_items: 12
        rmse: 1.14
        mean_signed_error: 0.466
        spearman: 0.951
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0053
          ess_bulk_min: 1096
          passed: true
      33.4.neeleman:
        n_items: 12
        rmse: 0.997
        mean_signed_error: 0.338
        spearman: 0.789
        coverage90: 0.917
        diagnostics_passed: false
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0127
          ess_bulk_min: 741
          passed: false
      34.1.fox:
        n_items: 12
        rmse: 1.453
        mean_signed_error: -0.19
        spearman: -0.028
        coverage90: 0.75
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0055
          ess_bulk_min: 988
          passed: true
      34.1.phillips:
        n_items: 12
        rmse: 0.736
        mean_signed_error: -0.104
        spearman: 0.83
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 6
          divergence_rate: 0.0008
          rhat_max: 1.006
          ess_bulk_min: 1014
          passed: true
      34.3.heycock:
        n_items: 12
        rmse: 1.309
        mean_signed_error: -0.211
        spearman: 0.657
        coverage90: 0.833
        diagnostics_passed: true
        diagnostics:
          divergences: 1
          divergence_rate: 0.0001
          rhat_max: 1.0097
          ess_bulk_min: 951
          passed: true
      34.4.boskovic:
        n_items: 12
        rmse: 0.736
        mean_signed_error: 0.363
        spearman: 0.846
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.005
          ess_bulk_min: 910
          passed: true
      41.3.Landau:
        n_items: 12
        rmse: 1.152
        mean_signed_error: 0.157
        spearman: 0.613
        coverage90: 0.917
        diagnostics_passed: true
        diagnostics:
          divergences: 0
          divergence_rate: 0.0
          rhat_max: 1.0039
          ess_bulk_min: 1082
          passed: true
      41.3.Vicente:
        n_items: 12
        rmse: 0.685
        mean_signed_error: -0.128
        spearman: 0.881
        coverage90: 1.0
        diagnostics_passed: true
        diagnostics:
          divergences: 2
          divergence_rate: 0.0003
          rhat_max: 1.0039
          ess_bulk_min: 947
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
    pooled_spearman_descriptive: 0.742
    pooled_spearman_cluster_boot_lower90: 0.654
    between_family_spearman: 0.406
    within_family_spearman_min: -0.028
    frac_families_within_spearman_gt_0.3: 0.9
    sd_observed_item_means: 1.532
    mean_spearman: 0.687
    mean_rmse: 0.992
    mean_coverage90: 0.926
    all_diagnostics_passed: false
    input_hashes:
      items: f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c
      human: 5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec
      measurements: 1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a
    stan_sha256: 969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86
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
    posterior_sha256: 06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723
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
      participant_entropy_ppp: 0.0
      participant_range_ppp: 0.162
      passed: false
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
      participant_entropy_ppp: 0.006
      participant_range_ppp: 0.204
      passed: false
    instrument_ppc:
      cell_family_ppp:
        pythia-160m/slor|32.1.martin: 0.98
        pythia-160m/slor|32.3.Culicover: 0.835
        pythia-160m/slor|32.3.fanselow: 0.735
        pythia-160m/slor|33.4.neeleman: 0.935
        pythia-160m/slor|34.1.fox: 0.755
        pythia-160m/slor|34.1.phillips: 0.835
        pythia-160m/slor|34.3.heycock: 0.9
        pythia-160m/slor|34.4.boskovic: 0.95
        pythia-160m/slor|41.3.Landau: 0.915
        pythia-160m/slor|41.3.Vicente: 0.895
        qwen3:8b/prompt_scalar|32.1.martin: 0.93
        qwen3:8b/prompt_scalar|32.3.Culicover: 0.965
        qwen3:8b/prompt_scalar|32.3.fanselow: 0.905
        qwen3:8b/prompt_scalar|33.4.neeleman: 0.95
        qwen3:8b/prompt_scalar|34.1.fox: 0.975
        qwen3:8b/prompt_scalar|34.1.phillips: 0.99
        qwen3:8b/prompt_scalar|34.3.heycock: 0.98
        qwen3:8b/prompt_scalar|34.4.boskovic: 0.915
        qwen3:8b/prompt_scalar|41.3.Landau: 0.94
        qwen3:8b/prompt_scalar|41.3.Vicente: 0.97
      min_ppp: 0.735
      passed: true
    passed: false
  multiverse_spread:
    mean_per_item_sd_across_cells: 0.627
    n_items: 120
    n_cells: 2
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
licensed_claims: {}
refused_claims:
  screening:
    type: affirmative_failure
    reason: PPC failed (human or instrument arm)
    remedy: the fitted model misfits the criterion data; revise the model
  ranking_within_family:
    type: affirmative_failure
    reason: PPC failed (human or instrument arm)
    remedy: the fitted model misfits the criterion data; revise the model
  family_location_unanchored:
    type: structural
    reason: absolute location of a family with no human anchor is prior-identified,
      not likelihood-identified (delta-invariance against per-cell family intercepts)
    remedy: collect human anchor items in each new family, or accept within-family
      claims only
  aggregate_estimation:
    type: affirmative_failure
    reason: PPC failed (human or instrument arm)
    remedy: the fitted model misfits the criterion data; revise the model
  effect_reproduction:
    type: unevaluable
    reason: 'not yet tested: requires matched experimental contrasts'
    remedy: add a matched-contrast validation stage (v2)
  distributional_claims:
    type: affirmative_failure
    reason: no participant-level validation of variance structure; posterior predictive
      checks failed
    remedy: model and validate rater heterogeneity (v2)
  population_transfer:
    type: unevaluable
    reason: the v1 ladder contains no population-transfer test; the estimand population
      defaults to the criterion sample's own
    remedy: add stratified human anchors from the target population (v2)
  individual_simulation:
    type: structural
    reason: an item-level instrument licenses no individual-level human simulation
    remedy: 'none: outside this design''s claim space'
  mechanism_claims:
    type: structural
    reason: the model estimates a linking function, not a mechanism
    remedy: 'none: outside this design''s claim space'
descriptive_findings:
  note: in-source descriptive associations on this item set; not deployment evidence
  pooled_heldout_spearman: 0.742
  between_family_spearman: 0.406
  per_family_within_spearman:
    32.1.martin: 0.884
    32.3.Culicover: 0.546
    32.3.fanselow: 0.951
    33.4.neeleman: 0.789
    34.1.fox: -0.028
    34.1.phillips: 0.83
    34.3.heycock: 0.657
    34.4.boskovic: 0.846
    41.3.Landau: 0.613
    41.3.Vicente: 0.881
  human_split_half: 0.857 (mean of 200 random participant splits, 120 items; Spearman-Brown
    corrected 0.923; computed from data/pilot_human.csv)
  response_style:
    marginal_band:
    - 3.0
    - 5.0
    cells:
      claude-opus-4.6-agy/prompt_scalar_batch20:
        n_items: 120
        r_all: 0.834
        r_marginal_band: 0.397
        r_marginal_band_n: 47
        r_outside_band: 0.935
        r_outside_band_n: 73
      gemma3:12b/prompt_scalar:
        n_items: 120
        r_all: 0.745
        r_marginal_band: 0.275
        r_marginal_band_n: 47
        r_outside_band: 0.852
        r_outside_band_n: 73
      mistral-small:24b-instruct-2501-q4_K_M/prompt_scalar:
        n_items: 120
        r_all: 0.791
        r_marginal_band: 0.236
        r_marginal_band_n: 47
        r_outside_band: 0.866
        r_outside_band_n: 73
      pythia-160m/logprob_mean:
        n_items: 120
        r_all: 0.366
        r_marginal_band: -0.27
        r_marginal_band_n: 47
        r_outside_band: 0.507
        r_outside_band_n: 73
      pythia-160m/logprob_sum:
        n_items: 120
        r_all: 0.156
        r_marginal_band: 0.21
        r_marginal_band_n: 47
        r_outside_band: 0.148
        r_outside_band_n: 73
      pythia-160m/slor:
        n_items: 120
        r_all: 0.35
        r_marginal_band: -0.179
        r_marginal_band_n: 47
        r_outside_band: 0.505
        r_outside_band_n: 73
      qwen3:8b/prompt_scalar:
        n_items: 120
        r_all: 0.724
        r_marginal_band: 0.183
        r_marginal_band_n: 47
        r_outside_band: 0.851
        r_outside_band_n: 73
    category_usage:
      human:
        '1': 0.144
        '2': 0.131
        '3': 0.119
        '4': 0.135
        '5': 0.128
        '6': 0.127
        '7': 0.217
      claude-opus-4.6-agy/prompt_scalar_batch20:
        usage:
          '1': 0.086
          '2': 0.231
          '3': 0.131
          '4': 0.05
          '5': 0.147
          '6': 0.125
          '7': 0.231
        categories_never_emitted: none
      gemma3:12b/prompt_scalar:
        usage:
          '1': 0.078
          '2': 0.068
          '3': 0.274
          '4': 0.151
          '5': 0.067
          '6': 0.18
          '7': 0.183
        categories_never_emitted: none
      mistral-small:24b-instruct-2501-q4_K_M/prompt_scalar:
        usage:
          '1': 0.03
          '2': 0.046
          '3': 0.225
          '4': 0.193
          '5': 0.206
          '6': 0.125
          '7': 0.176
        categories_never_emitted: none
      qwen3:8b/prompt_scalar:
        usage:
          '1': 0.075
          '2': 0.218
          '3': 0.118
          '4': 0.083
          '5': 0.094
          '6': 0.311
          '7': 0.101
        categories_never_emitted: none
residual_risks:
- 'binding integrity is not construct validity: a pristine, fully bound record can
  still document the wrong predicate; hash checks discharge staleness, not meaning'
- 'shared pretraining bias: local instruments share web-scale training data and can
  share construction-specific error; a tight multiverse fan does not rule this out'
- reliability quantities are fit- and item-set-specific and conditional on a family-exchangeability
  extrapolation model
- thresholds are pre-registered defaults, not estimand-specific loss analyses
- prompt-level dependence within pooled paraphrase cells is not yet modeled (nested
  prompt effects are v2)

```

### recovery / sbc / newfam / diagnostics / ppc / loco / response_style

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
  },
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86"
}

{
  "R": 100,
  "n_failed_fits": 0,
  "n_diag_failed": 8,
  "n_thin": 63,
  "params": {
    "tau_constr": {
      "chi2": 3.72,
      "p_uniform_approx": 0.717,
      "rank_hist": [
        18,
        11,
        11,
        11,
        15,
        15,
        11
      ]
    },
    "tau_item": {
      "chi2": 3.11,
      "p_uniform_approx": 0.7968,
      "rank_hist": [
        12,
        15,
        10,
        18,
        12,
        13,
        12
      ]
    },
    "sigma_u": {
      "chi2": 3.11,
      "p_uniform_approx": 0.7968,
      "rank_hist": [
        11,
        9,
        17,
        15,
        14,
        13,
        13
      ]
    },
    "beta[1]": {
      "chi2": 5.24,
      "p_uniform_approx": 0.5148,
      "rank_hist": [
        14,
        16,
        14,
        9,
        8,
        17,
        14
      ]
    },
    "beta[2]": {
      "chi2": 2.5,
      "p_uniform_approx": 0.8692,
      "rank_hist": [
        13,
        11,
        13,
        15,
        10,
        17,
        13
      ]
    },
    "sigma_s[1]": {
      "chi2": 13.3,
      "p_uniform_approx": 0.0382,
      "rank_hist": [
        9,
        12,
        13,
        8,
        19,
        22,
        9
      ]
    },
    "sigma_s[2]": {
      "chi2": 10.41,
      "p_uniform_approx": 0.1074,
      "rank_hist": [
        21,
        10,
        19,
        11,
        11,
        9,
        11
      ]
    },
    "tau_b[1]": {
      "chi2": 3.87,
      "p_uniform_approx": 0.6965,
      "rank_hist": [
        15,
        9,
        15,
        14,
        14,
        16,
        9
      ]
    },
    "tau_b[2]": {
      "chi2": 2.96,
      "p_uniform_approx": 0.8158,
      "rank_hist": [
        12,
        18,
        14,
        10,
        14,
        12,
        12
      ]
    },
    "omega[1]": {
      "chi2": 3.57,
      "p_uniform_approx": 0.7374,
      "rank_hist": [
        18,
        11,
        13,
        14,
        11,
        10,
        15
      ]
    },
    "omega[2]": {
      "chi2": 6.0,
      "p_uniform_approx": 0.4237,
      "rank_hist": [
        17,
        10,
        9,
        9,
        15,
        16,
        16
      ]
    }
  },
  "failed_fit_truths": [
    {
      "tau_constr": 0.81,
      "tau_item": 3.49,
      "sigma_u": 0.64
    },
    {
      "tau_constr": 0.09,
      "tau_item": 1.44,
      "sigma_u": 0.92
    },
    {
      "tau_constr": 1.18,
      "tau_item": 3.01,
      "sigma_u": 0.32
    },
    {
      "tau_constr": 0.65,
      "tau_item": 0.08,
      "sigma_u": 0.62
    },
    {
      "tau_constr": 1.15,
      "tau_item": 3.56,
      "sigma_u": 0.8
    },
    {
      "tau_constr": 1.09,
      "tau_item": 0.61,
      "sigma_u": 0.34
    },
    {
      "tau_constr": 1.96,
      "tau_item": 0.31,
      "sigma_u": 2.02
    },
    {
      "tau_constr": 0.3,
      "tau_item": 0.17,
      "sigma_u": 1.07
    }
  ],
  "passed": true,
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86"
}

{
  "n_new_families": 2,
  "diagnostics": {
    "divergences": 0,
    "divergence_rate": 0.0,
    "rhat_max": 1.0039,
    "ess_bulk_min": 1054,
    "passed": true
  },
  "per_family": {
    "fam7": {
      "theta_within_coverage90": 0.833,
      "within_rank_corr_truth": 0.839,
      "family_location_bias": 0.175
    },
    "fam8": {
      "theta_within_coverage90": 0.833,
      "within_rank_corr_truth": 0.853,
      "family_location_bias": -0.346
    }
  },
  "mean_within_coverage90": 0.833,
  "passed": true,
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86",
  "note": "family_location_bias is reported unGated: absolute location of an unanchored family is prior-identified (delta-invariance with per-cell family intercepts)"
}
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
  "posterior_sha256": "06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723",
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
    "participant_entropy_ppp": 0.0,
    "participant_range_ppp": 0.162,
    "passed": false
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
    "participant_entropy_ppp": 0.006,
    "participant_range_ppp": 0.204,
    "passed": false
  },
  "instrument_ppc": {
    "cell_family_ppp": {
      "pythia-160m/slor|32.1.martin": 0.98,
      "pythia-160m/slor|32.3.Culicover": 0.835,
      "pythia-160m/slor|32.3.fanselow": 0.735,
      "pythia-160m/slor|33.4.neeleman": 0.935,
      "pythia-160m/slor|34.1.fox": 0.755,
      "pythia-160m/slor|34.1.phillips": 0.835,
      "pythia-160m/slor|34.3.heycock": 0.9,
      "pythia-160m/slor|34.4.boskovic": 0.95,
      "pythia-160m/slor|41.3.Landau": 0.915,
      "pythia-160m/slor|41.3.Vicente": 0.895,
      "qwen3:8b/prompt_scalar|32.1.martin": 0.93,
      "qwen3:8b/prompt_scalar|32.3.Culicover": 0.965,
      "qwen3:8b/prompt_scalar|32.3.fanselow": 0.905,
      "qwen3:8b/prompt_scalar|33.4.neeleman": 0.95,
      "qwen3:8b/prompt_scalar|34.1.fox": 0.975,
      "qwen3:8b/prompt_scalar|34.1.phillips": 0.99,
      "qwen3:8b/prompt_scalar|34.3.heycock": 0.98,
      "qwen3:8b/prompt_scalar|34.4.boskovic": 0.915,
      "qwen3:8b/prompt_scalar|41.3.Landau": 0.94,
      "qwen3:8b/prompt_scalar|41.3.Vicente": 0.97
    },
    "min_ppp": 0.735,
    "passed": true
  },
  "passed": false
}

{
  "target": "same participants, new items",
  "per_family": {
    "32.1.martin": {
      "n_items": 12,
      "rmse": 0.849,
      "mean_signed_error": -0.462,
      "spearman": 0.884,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0053,
        "ess_bulk_min": 941,
        "passed": true
      }
    },
    "32.3.Culicover": {
      "n_items": 12,
      "rmse": 0.866,
      "mean_signed_error": -0.008,
      "spearman": 0.546,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0037,
        "ess_bulk_min": 1235,
        "passed": true
      }
    },
    "32.3.fanselow": {
      "n_items": 12,
      "rmse": 1.14,
      "mean_signed_error": 0.466,
      "spearman": 0.951,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0053,
        "ess_bulk_min": 1096,
        "passed": true
      }
    },
    "33.4.neeleman": {
      "n_items": 12,
      "rmse": 0.997,
      "mean_signed_error": 0.338,
      "spearman": 0.789,
      "coverage90": 0.917,
      "diagnostics_passed": false,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0127,
        "ess_bulk_min": 741,
        "passed": false
      }
    },
    "34.1.fox": {
      "n_items": 12,
      "rmse": 1.453,
      "mean_signed_error": -0.19,
      "spearman": -0.028,
      "coverage90": 0.75,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0055,
        "ess_bulk_min": 988,
        "passed": true
      }
    },
    "34.1.phillips": {
      "n_items": 12,
      "rmse": 0.736,
      "mean_signed_error": -0.104,
      "spearman": 0.83,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 6,
        "divergence_rate": 0.0008,
        "rhat_max": 1.006,
        "ess_bulk_min": 1014,
        "passed": true
      }
    },
    "34.3.heycock": {
      "n_items": 12,
      "rmse": 1.309,
      "mean_signed_error": -0.211,
      "spearman": 0.657,
      "coverage90": 0.833,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 1,
        "divergence_rate": 0.0001,
        "rhat_max": 1.0097,
        "ess_bulk_min": 951,
        "passed": true
      }
    },
    "34.4.boskovic": {
      "n_items": 12,
      "rmse": 0.736,
      "mean_signed_error": 0.363,
      "spearman": 0.846,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.005,
        "ess_bulk_min": 910,
        "passed": true
      }
    },
    "41.3.Landau": {
      "n_items": 12,
      "rmse": 1.152,
      "mean_signed_error": 0.157,
      "spearman": 0.613,
      "coverage90": 0.917,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 0,
        "divergence_rate": 0.0,
        "rhat_max": 1.0039,
        "ess_bulk_min": 1082,
        "passed": true
      }
    },
    "41.3.Vicente": {
      "n_items": 12,
      "rmse": 0.685,
      "mean_signed_error": -0.128,
      "spearman": 0.881,
      "coverage90": 1.0,
      "diagnostics_passed": true,
      "diagnostics": {
        "divergences": 2,
        "divergence_rate": 0.0003,
        "rhat_max": 1.0039,
        "ess_bulk_min": 947,
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
  "pooled_spearman_descriptive": 0.742,
  "pooled_spearman_cluster_boot_lower90": 0.654,
  "between_family_spearman": 0.406,
  "within_family_spearman_min": -0.028,
  "frac_families_within_spearman_gt_0.3": 0.9,
  "sd_observed_item_means": 1.532,
  "mean_spearman": 0.687,
  "mean_rmse": 0.992,
  "mean_coverage90": 0.926,
  "all_diagnostics_passed": false,
  "input_hashes": {
    "items": "f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c",
    "human": "5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec",
    "measurements": "1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a"
  },
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86"
}

{
  "marginal_band": [
    3.0,
    5.0
  ],
  "cells": {
    "claude-opus-4.6-agy/prompt_scalar_batch20": {
      "n_items": 120,
      "r_all": 0.834,
      "r_marginal_band": 0.397,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.935,
      "r_outside_band_n": 73
    },
    "gemma3:12b/prompt_scalar": {
      "n_items": 120,
      "r_all": 0.745,
      "r_marginal_band": 0.275,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.852,
      "r_outside_band_n": 73
    },
    "mistral-small:24b-instruct-2501-q4_K_M/prompt_scalar": {
      "n_items": 120,
      "r_all": 0.791,
      "r_marginal_band": 0.236,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.866,
      "r_outside_band_n": 73
    },
    "pythia-160m/logprob_mean": {
      "n_items": 120,
      "r_all": 0.366,
      "r_marginal_band": -0.27,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.507,
      "r_outside_band_n": 73
    },
    "pythia-160m/logprob_sum": {
      "n_items": 120,
      "r_all": 0.156,
      "r_marginal_band": 0.21,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.148,
      "r_outside_band_n": 73
    },
    "pythia-160m/slor": {
      "n_items": 120,
      "r_all": 0.35,
      "r_marginal_band": -0.179,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.505,
      "r_outside_band_n": 73
    },
    "qwen3:8b/prompt_scalar": {
      "n_items": 120,
      "r_all": 0.724,
      "r_marginal_band": 0.183,
      "r_marginal_band_n": 47,
      "r_outside_band": 0.851,
      "r_outside_band_n": 73
    }
  },
  "category_usage": {
    "human": {
      "1": 0.144,
      "2": 0.131,
      "3": 0.119,
      "4": 0.135,
      "5": 0.128,
      "6": 0.127,
      "7": 0.217
    },
    "claude-opus-4.6-agy/prompt_scalar_batch20": {
      "usage": {
        "1": 0.086,
        "2": 0.231,
        "3": 0.131,
        "4": 0.05,
        "5": 0.147,
        "6": 0.125,
        "7": 0.231
      },
      "categories_never_emitted": "none"
    },
    "gemma3:12b/prompt_scalar": {
      "usage": {
        "1": 0.078,
        "2": 0.068,
        "3": 0.274,
        "4": 0.151,
        "5": 0.067,
        "6": 0.18,
        "7": 0.183
      },
      "categories_never_emitted": "none"
    },
    "mistral-small:24b-instruct-2501-q4_K_M/prompt_scalar": {
      "usage": {
        "1": 0.03,
        "2": 0.046,
        "3": 0.225,
        "4": 0.193,
        "5": 0.206,
        "6": 0.125,
        "7": 0.176
      },
      "categories_never_emitted": "none"
    },
    "qwen3:8b/prompt_scalar": {
      "usage": {
        "1": 0.075,
        "2": 0.218,
        "3": 0.118,
        "4": 0.083,
        "5": 0.094,
        "6": 0.311,
        "7": 0.101
      },
      "categories_never_emitted": "none"
    }
  }
}

{
  "tau_item_median": 2.043,
  "omega_median_by_cell": {
    "pythia-160m/slor": 0.34,
    "qwen3:8b/prompt_scalar": 0.564
  },
  "qwen_pooled_scalar_item_mean_r_with_human": 0.724,
  "posterior_sha256": "06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723"
}
{
  "run_id": "6068b87c-5576-4a69-bebe-e6cc8f3cb75a",
  "posterior_sha256": "06726af6d3afb019151aa7531a4d4127080c288906774fb230e01f9889731723",
  "stan_sha256": "969366aab7300808e48e21cdc948513fbebd440ffe8673b20d96ee0bfe083b86",
  "code_commit": "62191e84ff1b9e468dad770a428efcdc3d24d235",
  "n_items_criterion": 120,
  "n_items_total": 120,
  "input_hashes": {
    "items": "f3ad1649032fd98a4525ed05de13880afd29cced8d792af34a3aa73c2fc6520c",
    "human": "5eaf0541ea4c5f15bb7d6142e5e93585db6b71251c35946d4816eb6c9fc64eec",
    "measurements": "1041de8980b9f884c22f04a7b6b5c3112cb2e6afd7a63a1515b1b36b5b5b7f9a"
  }
}

```

---

## PRIOR REVIEWS AND TRIAGE (from DECISIONS.md)

# DECISIONS — llm-acceptability-judgments
<!-- SUMMARY: Decision log for the acceptometer build · status: active · updated: 2026-08-27 -->

2026-08-27 — Build the tool as a joint Bayesian measurement-error model (Stan),
not a correlation pipeline. Origin: Brett asked what Gelman would say about the
LLM-judge literature, then "how would he improve the tool", then "make it so".
The design replaces the survey's validation checklist with one generative model
plus explicit generalization tests.

2026-08-27 — Warrant is a first-class output, co-equal with the posterior.
Brett's mid-build correction: "Gelman's posterior is important, but validity
requires a warrant if we're going to generalize at all." Every fit emits
warrant.yaml; LOCO-CV is the operationalized projectibility test; claim tiers
are granted conservatively and refused by default when evidence is missing.

2026-08-27 — Scale convention in the Stan model: human-arm discrimination fixed
at 1, within-family item sd fixed at 1, construction means sum-to-zero
(sum_to_zero_vector, CmdStan 2.36). Theta is in human-logit units; instrument
slopes are score-units per human logit. Reason: clean identification without
sacrificing the cutpoints or instrument parameters.

2026-08-27 — Minimal-pair deltas (mp_delta) are v1-descriptive only: computed
by the grid, plotted, but not yet a likelihood term (a pair-delta measures
theta_good minus theta_bad and needs its own likelihood block). Deferred, not
forgotten.

2026-08-27 — SLOR's unigram term uses wordfreq Zipf values as a proxy for a
model-based unigram LM, labeled as such in measurement meta. Reason: no corpus
unigram model in v1; the label keeps the proxy honest.

2026-08-27 — The survey document (ChatGPT deep-research) is gitignored and
excluded from any public repo: raw AI output. The tool code is publishable; the
survey is a local resource only.

2026-08-27 — Local-first instruments (Pythia logprobs, Ollama chat judges); API
adapters stubbed and off by default. Reason: free, reproducible, versionable;
API elicitation is a deliberate decision with a drift plan, not a default.

2026-08-27 — agy (Gemini 3.1 Pro High) design-review probe discarded: it
reviewed a different document than the one supplied (described DAG tests and an
`hpc_boundary()` API that appear nowhere in DESIGN.md). Consistent with the
recorded agy fabrication risk; not retried, quota preserved.

2026-08-27 — LOCO fits run at 1000/1000 iterations (vs 750/750 default fits)
after a held-out-family fit showed marginal mixing (R-hat 1.013 > 1.01 gate).
The gate stays; the iterations move. (Superseded later the same day: defaults
raised to 1000/1000, adapt_delta 0.95, after the freed latent scale needed
more adaptation.)

2026-08-27 — Within-family item sd (tau_item) freed; scale anchored by the
ordered-logistic error variance instead. Reason: first real-data PPC (Sprouse
LI pilot, 120 items, 1,519 ratings) failed with observed between-item spread
1.53 vs replicated 1.33 [1.25, 1.41] and observed within-item SD 1.44 vs
replicated 1.63 [1.57, 1.68]: fixing the sd at 1 compressed the latent scale
and forced ordinal noise to compensate. With tau_item free the data estimates
it at ~2.0 and the PPC passes (ppp .73 / .54). Recovery and SBC re-run and
green after the change. This is the PPC gate doing precisely what it was
built for on first contact with real data.

2026-08-27 — Pilot design: 120 Sprouse LI items (10 paper-families x 12,
6 starred / 6 good), all 1,519 available LS ratings from 304 participants;
construction family = source paper (volume.issue.author). Sprouse
participant-level data and derived files stay out of git (header requests
contacting Sprouse for novel research; redistribution unclear).

2026-08-27 — Cells of one model+method enter the fit as ONE instrument.
The first full pilot fit treated three scalar paraphrases (and three
transforms of one Pythia forward pass) as independent witnesses; their
correlated errors outvoted 1,519 human ratings (qwen "reliability" .93, PPC
failure, posterior tension). Paraphrases and repeats are repeated
measurements of one instrument; Pythia enters via SLOR only; binary cells
stay descriptive in v1 (no-overdispersion Bernoulli is misspecified for
near-deterministic repeats, b_b exploded to |11| under a N(0,1) prior).

2026-08-27 — Reflection mode handled by data-informed initialization plus a
zero-avoiding gamma(2,1) prior on tau_item. The joint posterior has a
locally-stable mirror (slopes and latent orientation jointly flipped) that
captured 1-in-4 randomly-initialized chains; escaping requires all thetas to
cross the valley at once, which HMC will not do. Chains now start at the
standardized human item means (basin selection, no posterior bias, no sign
constraints anywhere; an anti-correlated instrument can still reach negative
beta).

2026-08-27 — Second external review (GPT-family, via Brett) triaged and
largely adopted; it reviewed packet rev 2 but most findings held against rev
3 code. Adopted: (1) DESIGN scale-convention text rewritten to match the
implemented identification (link-anchored scale; sum-to-zero family location;
prior-anchored absolute location; design-dependent variance separation);
(2) held-out/new families excluded from the sum-to-zero vector, independent
normal(0, tau_constr) predictive effects (the constrained vector leaks
finite-set centering information to a "new" family); (3) tau_item, tau_a,
tau_b, omega added to the diagnostics core and to fake-data recovery truth
and gates; (4) warrant enforces the full ladder literally: recovery + SBC
prerequisites, PPC required (not merely not-failed) for aggregate, nonresponse
flags exclude cells, LOCO must cover all families with fold diagnostics
passed, and all evidence must be hash-bound to the posterior via run.json;
(5) reliability gate switched to new-family predictive reliability (fresh
slope-deviation draw per posterior draw), per-family reliabilities reported,
global ratio demoted to non-warrant status; (6) contamination cap extended to
ranking (LOCO rewards contamination); uncertainty-aware gates (median + q10;
pooled Spearman + family-cluster bootstrap lower bound); (7) tie-aware
Spearman (average ranks) + pooled out-of-fold statistic; (8) LOCO target
stated as same-participants-new-items, predictions use raters' posterior
effects; (9) PPC: n_sims 1000, conditional AND marginal participant modes,
per-family ppp vectors gated on minima, proper predictive reference for
category usage, instrument-arm residual flags (warn-only); (10) LOCO
standardization training-only (was transductive); (11) SBC diagnostics-aware
with a 20% failure cap, rerun at R=100; (12) SLOR made unit-consistent
(per-word, first-word skip aligned with no-BOS path; cached pilot cell
re-scored) and HF provenance now records the checkpoint commit hash;
(13) alpha renamed alpha_standardized_median. Adapted rather than adopted:
full decision-theoretic thresholds (deferred: needs estimand-specific loss),
zero-divergence strictness (0.5% gate kept, documented). Rejected in part:
"no real-data LOCO/warrant exists" — the reviewer saw rev 2; both exist in
rev 3. Deferred to v2 with reasons: second (post-cutoff) item source,
population-transfer test, temporal drift evidence.

2026-08-27 — Cross-model marginal-band result (five judge families; Opus 4.6
elicited via agy with a quota-safe batched protocol, 18 calls, 0 parse
failures; batch protocol recorded in the cell identity): aggregate r with
human item means rises with capability (.35 pythia -> .72 qwen8b -> .74
gemma12b -> .79 mistral24b -> .83 opus4.6, human split-half ceiling .857) and
the contested-band r rises too but lags everywhere (-.18 -> .18 -> .28 ->
.24 -> .40, vs .85-.94 outside the band). The middle-band deficit is
class-wide and scale-mitigated, not solved; Opus also avoids category 4
(5% vs humans' 13.5%). Copilot CLI probed: accepts claude-sonnet-4.6, no
Opus tier, so agy is the Opus route.

2026-08-27 — Final pipeline PPC verdict (post participant-style checks): the
single failing discrepancy is participant-level category-usage entropy
(ppp .000 conditional / .006 marginal); all family-level, item-level,
category, and instrument-arm checks pass. Diagnosis: real raters have
idiosyncratic scale-use styles the additive intercept cannot compress or
expand — review 3's predicted misfit, caught by the check built for it.
v1 keeps the conservative pre-commitment (any PPC failure blocks all
deployment claims); rater-style modeling and a per-tier PPC consequence map
are v2. LOCO binding hole closed the same hour: LOCO now stamps the Stan
hash so a report generated under an older model refuses even with matching
input hashes (the review-3 stale-evidence scenario, found live when the
pre-is_new loco.json would have bound).

2026-08-27 — Assurance machinery folded in from Brett's AI-safety papers
(adversarial-pragmatics / delegation-assurance / evidentiary-assurance /
ai-evaluation-kinds), at Brett's direction after the cross-reading: (1)
frozen versioned threshold spec (src/acceptometer/thresholds.yaml + spec.py);
diagnostics, SBC, PPC, and warrant gates all read it, and certificates cite
spec_version + SHA (delegation-assurance's frozen-specification practice);
(2) the certificate is now a LICENCE WITH A LIFE-CYCLE (ai-evaluation-kinds):
status, issuer, expiry conditions (instrument/item/model change), erosion
conditions (Goodhart against the gate statistics), defeat/supersession, and
a contestation route; (3) each licensed claim carries the four
projective-claim blocks (declaration / projectibility profile / warrant /
defeaters), with the profile supplied by the analyst in estimand.yaml as a
stated HYPOTHESIS, never fused with the warrant (the shared claim-types
protocol); (4) refusals are typed with remedies per evidentiary-assurance's
verdict separation: unevaluable / shortfall / affirmative_failure / vitiated
/ structural; (5) answerability: issued_by named, contestation route stated;
Robodebt residual risk recorded (binding integrity is not construct
validity); (6) response-style evidence module (scripts/response_evidence.py)
from adversarial-pragmatics' minority-class and label-omission lessons.

2026-08-27 — Marginal-stratum finding (first output of the new evidence
module): qwen's aggregate r = .72 with human item means collapses to r = .18
on the 47 contested-band items (human mean in [3,5]) and rises to .85
outside it; Pythia SLOR is -.18 in the band. The aggregate correlation is
carried by the easy items. Category usage: humans spread across the scale
(peak at 7); qwen hedges (peaks at 2 and 6, avoids extremes and middle).
Consequence for use: the instrument profile fits triage (screen clear cases,
spend humans on the middle band, which is what design.py allocates), and any
marginal-item claim is unsupported. Reported under descriptive_findings.

2026-08-27 — SBC per-fit sanity checks moved to 4 chains (500/500) after the
2-chain run failed the tightened 10% cap with 19 scattered (non-concentrated)
diagnostic failures, consistent with 2-chain R-hat noise rather than a
geometry pathology. The pipeline's stop-at-first-failed-gate behavior worked
as designed.

2026-08-27 — Third external review (GPT-family, via Brett) triaged; verdict
accepted: the screening grant is withdrawn and the pilot certificate licenses
no deployment tier (descriptive findings only). Adopted: (1) reliability_new
was sign-blind (squaring launders reversed slopes into signal) and used
pooled between+within theta variance; replaced with P(new-family slope
positive) plus a DIRECTIONAL within-family-variance reliability in which
negative-slope draws count as zero; family-specific reliabilities likewise
sign-aware and within-family; (2) contamination caps screening too, for the
same reason it caps ranking; (3) hash binding made real: the warrant
RECOMPUTES posterior/Stan hashes and requires exact input-hash-map equality;
recovery/SBC/newfam bind via Stan-source hash; (4) simulated new-family
recovery (newfam_check) added as a ladder stage: under known truth the new
branch shows within-family coverage 0.83 and rank recovery +.84/+.85 while
family-location bias (+0.19/-0.33) empirically displays the
prior-identification limit; (5) instrument-arm PPC was broken (raw scores vs
standardized predictions, omega omitted; z of 106-155 was the signature) —
rebuilt as a posterior-predictive check on the fitted scale with all model
terms, and GATED; (6) pooled LOCO Spearman demoted to descriptive (it
conflates between-family separation with within-family ordering); claims
split into within-family ranking, family location, aggregate; aggregate
decoupled from ranking and given a sharpness gate (RMSE <= .75 x sd of
observed means); (7) family location for unanchored families refused
STRUCTURALLY (delta-invariance: shifting a new family's mean and each cell's
family intercept in compensation leaves the instrument likelihood unchanged);
(8) exchangeability labeling on every new-family quantity (families are
purposive; the fresh-deviation draw is a working extrapolation model);
(9) mode_audit added (overdispersed starts, lp comparison across
orientations) since same-basin agreement cannot rule out the mirror;
diagnostics core extended to a_dev/b_dev/mu_new_raw/theta; (10) SBC excludes
failed-diagnostic replications from rank histograms, records their prior
locations, cap tightened to 10%, tau_b and omega tracked; (11) SLOR no-BOS
path eliminated via EOS-as-BOS fallback; (12) provenance (checkpoint commit,
Ollama digest, dates) propagated into certificate instrument entries;
posterior_summary.json added so report numbers are auditable; pilot script
stops at first failed gate and deletes stale evidence. Adapted: prompt-level
dependence noted as residual risk (nested prompt effects v2); participant
response-style PPCs added (entropy, range) with item-by-participant residual
checks deferred. Rejected: none of substance.

2026-08-27 — Instrument-by-item error term (omega) added for replicated
cells. Repeats average away draw noise but never the instrument's stable
opinion about an item; without the term, replicated cells claim fictitious
precision and drag theta off the human criterion (the PPC caught it).
Identified only against replicate noise, so it is switched off for
single-observation cells (redundant-ridge divergences otherwise). Effect on
real data: qwen scalar conditional reliability .90 -> .54 with omega = .56;
PPC now passes (ppp .86/.78). Recovery recovers omega to +-0.04; full ladder
re-run green.

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
