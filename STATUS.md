# STATUS — acceptometer
<!-- SUMMARY: Warranted Bayesian LLM-judgment instrument; v0.1 complete, real-data pilot done, screening+ranking licensed · status: pilot complete · updated: 2026-08-27 -->

## Where things stand (2026-08-27, end of build day)

v0.1 is built, validated, and has completed a real-data pilot. Full story:
`PILOT-REPORT.md`; decisions: `DECISIONS.md`; design: `DESIGN.md`.

- **Model** (Stan): joint measurement-error model; ordinal human arm with
  participant effects; instrument arm with family-varying linking,
  instrument-by-item error for replicated cells, nuisance covariates, noise
  floor. Three real-data-driven expansions today, each forced by a gate.
- **Validation ladder**: fake-data recovery, SBC-lite, diagnostics gate, PPC
  gate, LOCO-CV — all green on the final model.
- **Pilot** (Sprouse LI, 120 items, 1,519 ratings; Pythia-160m + qwen3:8b):
  human split-half .857; qwen pooled scalar r=.72 with humans, conditional
  reliability .54, omega .56; LOCO Spearman .675 with one family (34.1.fox)
  failing outright; coverage .925. Warrant grants screening + ranking, refuses
  aggregate estimation on the contamination cap.
- **External review**: glm-5.3-flash design review adopted in large part
  (triage in DECISIONS.md); validation packet for a second-family review at
  `reviews/validation-packet-2026-08-27.md` (rev 3, sent to Brett). agy probe
  discarded (reviewed a nonexistent document).

## next_action

1. Run the rev-3 validation packet past an OpenAI-family model (codex one-liner
   in the packet discussion) and triage findings.
2. v2 candidates, in rough order of value: clean post-cutoff item set (lifts
   the contamination cap and tests aggregate estimation); ordered-logistic
   instrument arm for scalar cells; binary-arm overdispersion; rater-specific
   cutpoints; prompt invariance as a model statistic; prior-sensitivity sweep;
   population-transfer test.
3. If Sprouse data is to be used beyond piloting, contact Jon Sprouse per the
   data header.
