# Acceptometer

A warranted Bayesian measurement instrument for LLM acceptability judgments.

The premise: an LLM is not a judge, it's a **biased measurement instrument** for
human acceptability judgments. This tool elicits judgments from LLMs across an
explicit multiverse of elicitation methods, fits a joint Bayesian
measurement-error model (Stan) linking those scores to participant-level human
ratings, and emits two artifacts per fit:

1. **A posterior**: for each item, the predicted human response distribution and
   its uncertainty, with the LLM contributing exactly as much as its modeled
   reliability earns.
2. **A warrant** (`warrant.yaml`): a validity certificate stating the domain
   actually validated, the generalization evidence (leave-one-construction-out
   transfer, multiverse spread, prompt invariance, drift epoch), and which claim
   tiers are licensed or refused.

The posterior answers "what does this instrument say?"; the warrant answers
"where are you entitled to believe it?". Neither substitutes for the other.

Design rationale and statistical model: [DESIGN.md](DESIGN.md).

## Quick start

```bash
uv sync --extra hf --extra dev
uv run acceptometer simulate --check     # fake-data recovery: trust gate for the pipeline
uv run acceptometer elicit --grid default --items data/items.jsonl
uv run acceptometer fit runs/<run>/
uv run acceptometer validate runs/<run>/ # LOCO-CV, multiverse spread, invariance
uv run acceptometer warrant runs/<run>/  # emit the certificate
uv run acceptometer plot runs/<run>/
```

## Status

v0.1: model + simulation validation ladder + local instruments (HF logprobs,
Ollama prompted). API-model adapters are stubs, off by default. No fabricated
data anywhere: the real-data stage runs only against sources recorded in
`data/MANIFEST.yaml`.
