# CLAUDE.md — llm-acceptability-judgments / acceptometer

Role here: **developer/statistician**, not PM. This is a software tool project.

## What lives here

- `acceptometer`: a Bayesian measurement tool treating LLMs as biased
  instruments for human acceptability judgments. Architecture and statistical
  model: `DESIGN.md`. Every fit emits a posterior AND a warrant certificate;
  neither substitutes for the other.
- `llm-grammaticality-judgments-survey.md`: the literature survey that motivated
  the design (ChatGPT deep-research, edited). **Gitignored, do not commit or
  publish**: raw AI output attributed to no author. The cited PDFs are in the
  portfolio's `literature/`.

## Hard rules

- **The validation ladder is ordered and gated** (DESIGN.md). Nothing
  downstream accepts a fit that fails `diagnostics_gate`; no real-data claim
  before `simulate --check` and SBC pass on the current model version. If you
  change `acceptometer.stan`, rerun both before anything else.
- **No fabricated data.** Real data enters only through `data/MANIFEST.yaml`
  with URL, access date, SHA-256, license. If a needed dataset is missing, the
  tool waits and says so.
- **Repeated LLM draws are repeated measurements of one instrument, never
  synthetic participants.** Don't let any code path or doc blur this.
- **The warrant logic errs conservative.** Missing evidence refuses a claim
  tier; it never grants by default. Don't "fix" a refused tier by weakening the
  rule; produce the evidence.
- Python via `uv` (pinned 3.12; system 3.14 breaks torch). Run things with
  `uv run ...` or `.venv/bin/python`. Stan via cmdstanpy + CmdStan 2.36
  (`~/.cmdstan`).

## Working notes

- Ollama serves the local prompted judges (`ollama serve` must be up for
  elicitation; `qwen3:8b`, `gemma3:12b`, `mistral-small:24b` installed).
- HF logprob scorer defaults to Pythia-160m on MPS; first call downloads the
  checkpoint.
- API-model adapters stay off by default. Enabling one is a decision: log it in
  DECISIONS.md with the version pin and the drift plan.
