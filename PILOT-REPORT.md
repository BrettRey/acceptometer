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
