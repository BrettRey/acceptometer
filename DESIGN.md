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

**Instrument arm** (the LLM scores). Elicitation method `m` (a cell in the
multiverse grid, model x method x prompt paraphrase) produces score `s_im`, and
the linking function varies by construction family `c` (LLM scoring error
clusters by phenomenon; a family-invariant linking would assume that away):

    continuous: s_im ~ normal(alpha_m + a_dev_mc + (beta_m + b_dev_mc) * theta_i + gamma_m' * x_i, sigma_m)
                a_dev_mc ~ normal(0, tau_a_m),  b_dev_mc ~ normal(0, tau_b_m)
    binary:     s_im ~ bernoulli_logit(a_m + b_m * theta_i + g_m' * x_i)

`x_i` = nuisance covariates: log token length, mean unigram log frequency
(wordfreq). `beta_m` is the population information coefficient; `alpha_m`,
`gamma_m`, and the family deviations are the instrument's bias structure;
`sigma_m` its noise, floored at 0.05 on the standardized scale so no cell can
present itself as a noiseless oracle. For an item in a NEW family, the
deviations are drawn from their priors, which is what makes out-of-family
intervals wider than in-family ones by construction rather than by hope.

**Conditional reliability** per continuous cell =
`beta_m^2 * var(theta) / (beta_m^2 * var(theta) + sigma_m^2)`. This is
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
