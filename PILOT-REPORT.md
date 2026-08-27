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
