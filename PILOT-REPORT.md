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
