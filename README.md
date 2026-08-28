# Acceptometer

[![CI](https://github.com/BrettRey/acceptometer/actions/workflows/ci.yml/badge.svg)](https://github.com/BrettRey/acceptometer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A warranted Bayesian measurement instrument for LLM acceptability judgments.

The premise is that an LLM is not a judge but a **biased measurement
instrument** for human acceptability judgments. Acceptometer elicits judgments
across an explicit multiverse of methods and fits a joint Bayesian
measurement-error model in Stan. Each fit can produce two distinct artifacts:

1. **A posterior**: the predicted human response distribution for each item,
   with the LLM contributing only as much as its modeled reliability earns.
2. **A warrant** (`warrant.yaml`): a validity certificate that records the
   domain actually tested, the generalization evidence, and the claim tiers
   licensed or refused.

The posterior answers “what does this instrument say?”; the warrant answers
“where are you entitled to believe it?” Neither substitutes for the other.
See [DESIGN.md](DESIGN.md) for the statistical model and rationale.

## v0.1 status

The first real-data pilot and its eleven-judge descriptive extension are
complete. The comparison includes local models, Claude Opus 4.6 and 5, and
GPT-5.6 Sol, Terra, and Luna. The certificate remains `descriptive_only`:
participant response-style posterior-predictive checks failed and the source
items are contamination-suspect. These results license no deployment tier.

- [Pilot report](PILOT-REPORT.md)
- [Redistributable v0.1 aggregate evidence](results/v0.1/README.md)
- [Current project status](STATUS.md)

## Install and check

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --frozen
uv run pytest -m "not slow"
uv run python scripts/judge_table.py --check
uv build
```

The complete local test suite additionally requires CmdStan 2.36:

```bash
uv run pytest
uv run acceptometer simulate --check
```

## Run the pipeline on your data

Supply an items JSONL, participant-level human ratings, and an estimand YAML.
The schemas and warrant rules are documented in [DESIGN.md](DESIGN.md).

```bash
uv sync --extra hf --extra dev

uv run acceptometer elicit \
  --items data/items.jsonl \
  --out runs/example \
  --hf EleutherAI/pythia-160m \
  --ollama qwen3:8b

uv run acceptometer fit runs/example \
  --items data/items.jsonl \
  --human data/human.csv \
  --measurements runs/example/measurements.jsonl

uv run acceptometer validate runs/example \
  --items data/items.jsonl \
  --human data/human.csv \
  --measurements runs/example/measurements.jsonl

uv run acceptometer warrant runs/example \
  --estimand data/estimand.yaml

uv run acceptometer plot runs/example
```

Missing or failed evidence refuses warrant tiers; it never grants them by
default.

The frontier-judge runner is optional, resumable, quota-bearing, and requires
authenticated Claude Code or Codex CLIs:

```bash
uv run python scripts/frontier_judge.py --judge opus-5
uv run python scripts/frontier_judge.py \
  --judge gpt-5.6-sol \
  --judge gpt-5.6-terra \
  --judge gpt-5.6-luna
```

## Published aggregate evidence

The tracked JSON snapshot is sufficient to reproduce the eleven-judge table
without the restricted pilot inputs:

```bash
uv run python scripts/judge_table.py
uv run python scripts/judge_table.py --check
```

Maintainers can update the generated block in `PILOT-REPORT.md` with
`--write`. The evidence records source hashes and model-plus-harness identities
but contains no sentences, participant identifiers, raw model responses, or
posterior draws.

## Data and privacy boundary

The public repository contains code, model and threshold specifications,
tests, provenance manifests, documentation, and safe aggregate results. It
does **not** contain:

- participant-level or derived human-rating files;
- sentence/item text from the restricted pilot sources;
- raw model responses, CLI transcripts, or run directories;
- fitted posteriors or posterior draws; or
- the AI-generated survey used during development.

`data/MANIFEST.yaml` records the upstream sources and their stated terms.
`data/convert_human.py` works only after those files have been obtained
legitimately and placed under the ignored `data/raw/` directory. The Sprouse,
Schütze, and Almeida release asks researchers to contact Jon Sprouse before
novel research use. This repository does not relicense third-party data.

A separate private GitHub data repository is intentionally not part of v0.1:
private hosting would not resolve redistribution or research-use restrictions.

## License

Acceptometer’s original code and documentation are released under the
[MIT License](LICENSE). Third-party data and source materials retain their own
terms.
