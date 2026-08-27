# STATUS — acceptometer
<!-- SUMMARY: Gelman-style Bayesian instrument for LLM acceptability judgments; core model built and validated on fake data · status: v0.1 build in progress · updated: 2026-08-27 -->

## Where things stand (2026-08-27)

Built and verified today:

- Stan joint measurement model (`src/acceptometer/model/acceptometer.stan`):
  latent item acceptability, ordinal human arm with participant effects,
  continuous + binary instrument cells with per-cell bias/slope/nuisance/noise.
  Compiles clean on CmdStan 2.36.
- Fake-data recovery: PASSED (0 divergences, R-hat max 1.007, beta recovered to
  ±0.003, reliability to ±0.05, theta 90% CI coverage 0.958, 8s wall).
- LOCO-CV harness (`model/loco.py`): PASSED on simulated data (4 families,
  mean held-out Spearman 0.79, 90% coverage 0.84, all diagnostics green).
- SBC-lite (`model/sbc.py`): running.

In flight (parallel dispatch):

- codex: elicitation layer (HF logprob scorer, Ollama chat judges, multiverse
  grid runner, prompt registry, tests).
- Claude subagent: plots (secret weapon, reliability forest, multiverse fan,
  calibration), CLI, warrant builder, drift monitor, budget allocator.
- Claude subagent: data acquisition with provenance (MORCELA human ratings,
  Sprouse-Schütze-Almeida LI data, BLiMP sample) into data/MANIFEST.yaml.
- ocx (glm-5.3-flash): adversarial design review of DESIGN.md.
- agy probe: discarded (reviewed the wrong document; see DECISIONS.md).

## next_action

Integrate subagent output, run the full ladder end-to-end (simulate --check,
SBC, LOCO), then a real elicitation pilot with local instruments against
whatever human criterion data cleared provenance.
