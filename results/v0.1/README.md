# Acceptometer v0.1 aggregate evidence

This directory contains the redistributable aggregate evidence for the v0.1
pilot report. It deliberately excludes sentence text, participant identifiers,
participant-level ratings, raw model responses, transcripts, posterior draws,
and downloaded source data.

`judge-table.json` records the eleven descriptive judge correlations, the
contested-band definition, response category 4 usage, the human split-half
estimate, model-plus-harness identities, and SHA-256 hashes of the local inputs
from which the aggregate was generated. The hashes support provenance checks;
the restricted inputs themselves are not part of this release.

Regenerate or check the table in `PILOT-REPORT.md` with:

```bash
uv run python scripts/judge_table.py --write
uv run python scripts/judge_table.py --check
```

These aggregate results do not upgrade the pilot warrant. Its licence status
remains `descriptive_only` because the participant response-style PPC failed
and the source-item contamination risk remains unresolved.
