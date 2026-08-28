# STATUS — acceptometer
<!-- SUMMARY: Warranted Bayesian LLM-judgment instrument; certificate descriptive_only; eleven-judge table now includes Opus 5 and GPT-5.6 Sol/Terra/Luna; local qwen3.8:27b remains competitive · status: v1 complete, pilot extended · updated: 2026-08-28 -->

## Where things stand (2026-08-28)

v0.1 is built, validated, and has completed a real-data pilot. Full story:
`PILOT-REPORT.md`; decisions: `DECISIONS.md`; design: `DESIGN.md`.

- **Model** (Stan): joint measurement-error model; ordinal human arm with
  participant effects; instrument arm with family-varying linking,
  instrument-by-item error for replicated cells, nuisance covariates, noise
  floor. Three real-data-driven expansions today, each forced by a gate.
- **Validation ladder**: fake-data recovery, SBC-lite, diagnostics, simulated
  new-family recovery, and LOCO-CV ran on the final model. All modeled checks
  passed except participant-level category-usage entropy in the human PPC
  (ppp .000 conditional / .006 marginal), which remains binding.
- **Pilot** (Sprouse LI, 120 items, 1,519 ratings; Pythia-160m + qwen3:8b):
  human split-half .857; qwen pooled scalar r=.72 with humans, conditional
  reliability .54, omega .56; LOCO Spearman .675 with one family (34.1.fox)
  failing outright; coverage .925. The final warrant is descriptive_only:
  contamination and the rater-entropy PPC failure block deployment claims.
- **Descriptive judge extension**: eleven-judge table complete. Opus 5 is
  .83 overall/.33 in the contested band; GPT-5.6 Sol .80/.36, Terra .71/.34,
  Luna .78/.42. Qwen3.8:27b remains competitive at .81/.42. The four new
  cells contain 360 ratings each (120 items x 3 shuffled passes), with zero
  failed batches; they do not alter the fitted model or warrant.
- **External review**: glm-5.3-flash design review adopted in large part
  (triage in DECISIONS.md); validation packet for a second-family review at
  `reviews/validation-packet-2026-08-27.md` (rev 3, sent to Brett). agy probe
  discarded (reviewed a nonexistent document).

## next_action

1. If desired, run the rev-5 validation packet past a fresh reviewer; all
   three prior reviews are triaged in DECISIONS.md and adopted. Final
   certificate: descriptive_only (contamination + rater-entropy PPC).
2. v2 candidates, in rough order of value: clean post-cutoff item set (lifts
   the contamination cap and tests aggregate estimation); ordered-logistic
   instrument arm for scalar cells; binary-arm overdispersion; rater-specific
   cutpoints; prompt invariance as a model statistic; prior-sensitivity sweep;
   population-transfer test.
3. If Sprouse data is to be used beyond piloting, contact Jon Sprouse per the
   data header.
