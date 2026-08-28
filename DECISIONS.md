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

2026-08-27 — Sweep completed (glm-4.7-flash, qwen3.8:27b; the background job
was stopped twice by hand, so qwen3.8:27b scalar has 1 repeat per paraphrase
— full item coverage, not relaunched). Final seven-judge table: qwen3.8:27b
(.81 all / .42 band) MATCHES Opus 4.6 (.83/.40); glm-4.7-flash is weakest
(.61/.07). Within-family scale contrast qwen 8B->27B: .72/.18 -> .81/.42.
The band stays at roughly half the outside signal for every judge at every
scale: class-wide, scale-mitigated, unsolved.

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
