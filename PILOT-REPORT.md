# Acceptometer pilot report: Sprouse LI items, eleven judge models
<!-- SUMMARY: Final pilot: certificate descriptive_only; contested-band deficit measured across eleven judges incl. Opus 5 and GPT-5.6 Sol/Terra/Luna; local qwen3.8:27b remains competitive · status: complete · updated: 2026-08-28 -->

## What ran

120 items from Sprouse, Schütze & Almeida (2013) (10 source-paper families,
6 starred / 6 good each) with all 1,519 participant-level 7-point ratings
(304 participants). Judges: Pythia-160m (exact log-probabilities), qwen3:8b,
gemma3:12b, mistral-small:24b (local, single-item protocol, 3 paraphrases x 3
repeats), Claude Opus 4.6 via agy, Claude Opus 5 via Claude Code, and GPT-5.6
Sol, Terra, and Luna via Codex (frontier models: batched-20 protocol, 3 passes,
18 calls each, 0 parse failures). Glm-4.7-flash and qwen3.8:27b are also
complete; qwen3.8:27b scalar has 1 repeat per paraphrase because its
elicitation was stopped early. The joint Bayesian fit still uses qwen3:8b +
Pythia SLOR; all other judges, including the four added on 2026-08-28, are
descriptive cells and do not change the fit or warrant.

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

<!-- BEGIN GENERATED JUDGE TABLE -->
| judge | r all | r band | r outside |
|---|---|---|---|
| Claude Opus 4.6 (agy, batched) | **.83** | .40 | .94 |
| Claude Opus 5 (Claude Code, batched) | **.83** | .33 | .94 |
| qwen3.8:27b (local; 3 obs/item) | **.81** | **.42** | .91 |
| GPT-5.6 Sol (Codex, batched) | **.80** | .36 | .91 |
| mistral-small 24B | .79 | .24 | .87 |
| GPT-5.6 Luna (Codex, batched) | .78 | **.42** | .88 |
| gemma3 12B | .74 | .28 | .85 |
| qwen3 8B | .72 | .18 | .85 |
| GPT-5.6 Terra (Codex, batched) | .71 | .34 | .85 |
| glm-4.7-flash | .61 | .07 | .71 |
| Pythia-160m SLOR | .35 | -.18 | .51 |
| *human split-half ceiling* | *.857* | | |
<!-- END GENERATED JUDGE TABLE -->

A local 27B open model (qwen3.8:27b) remains competitive with every frontier
judge: it is within .02 of both Opus versions overall and exceeds them in the
contested band (.42 vs .40/.33), though it is lower outside the band (.91 vs
.94). Within the qwen family, scale lifts both numbers (8B .72/.18 -> 27B
.81/.42).

The GPT-5.6 tiers do not form a monotonic quality ladder on this task. Sol is
best overall (.80), Luna is best in the contested band (.42), and Terra is
weakest overall (.71). Opus 5 is effectively unchanged from Opus 4.6 overall
(.832 vs .834) but lower in the band (.334 vs .397); because the two Opus
versions ran through different harnesses, that contrast is not a clean model-
version effect.

The band lags the outside-band correlation for every prompted judge. All four
new frontier judges also underuse category 4 (2.5%--6.4% vs humans' 13.5%)
while favoring the scale endpoints. The deficit is class-wide and not solved
by a newer frontier model or by moving among GPT-5.6 tiers. Published aggregate
correlations around .8 are real, but much of that signal is carried by the
clear items for which an instrument is least needed.

Study-design implication, not a deployment licence: a fresh validation study
should concentrate its human-rating budget in the middle band (which is what
design.py allocates). The current certificate does not license any tested
judge to screen even the clear cases, and no marginal-item claim is supported.

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
one register, two adjacent access dates. Frontier judges used a batched
protocol, and the harness is part of every cell identity: Opus 4.6 used agy,
Opus 5 used Claude Code, and GPT-5.6 used Codex. Harness-confounded comparisons
therefore remain descriptive. The certificate lists untested axes. v2
priorities: post-cutoff item set (the only route to any deployment grant),
rater-style modeling (clears the PPC), ordered-logistic scalar arm, binary
overdispersion, per-tier PPC consequence map, population-transfer test.

## Artifacts

`runs/pilot/`: warrant.yaml, run.json, recovery.json, sbc.json, newfam.json,
ppc.json, loco.json, posterior.nc, posterior_summary.json,
response_style.json, instruments.json, estimand.yaml, plots.
`runs/multi/measurements.jsonl`: cross-model judgments. Provenance:
`data/MANIFEST.yaml`. Pipeline: `scripts/pilot.py`; cross-model evidence:
`scripts/response_evidence.py`, `scripts/agy_judge.py`,
`scripts/frontier_judge.py`.
