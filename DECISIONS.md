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
