from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.judge_table import (
    END_MARKER,
    START_MARKER,
    format_correlation,
    generated_block,
    load_evidence,
    render_table,
    replace_generated_block,
)

EVIDENCE = Path(__file__).parent.parent / "results/v0.1/judge-table.json"


def test_published_table_has_eleven_ordered_judges_and_ceiling() -> None:
    table = render_table(load_evidence(EVIDENCE))
    assert table.count("\n|") == 13  # divider + eleven judges + ceiling
    assert table.index("Claude Opus 4.6") < table.index("Claude Opus 5")
    assert table.index("Claude Opus 5") < table.index("qwen3.8:27b")
    assert "| Pythia-160m SLOR | .35 | -.18 | .51 |" in table
    assert "| *human split-half ceiling* | *.857* | | |" in table


def test_published_source_hashes_are_sha256() -> None:
    evidence = load_evidence(EVIDENCE)
    assert evidence["source_sha256"]
    assert all(len(digest) == 64 for digest in evidence["source_sha256"].values())


def test_malformed_source_hash_is_refused(tmp_path: Path) -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["source_sha256"]["runs/pilot/measurements.jsonl"] = "too-short"
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid SHA-256"):
        load_evidence(path)


def test_rounded_ties_are_highlighted() -> None:
    table = render_table(load_evidence(EVIDENCE))
    assert "Claude Opus 4.6 (agy, batched) | **.83**" in table
    assert "Claude Opus 5 (Claude Code, batched) | **.83**" in table
    assert "qwen3.8:27b (local; 3 obs/item) | **.81** | **.42**" in table
    assert "GPT-5.6 Luna (Codex, batched) | .78 | **.42**" in table


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.834, ".83"), (-0.179, "-.18"), (1.0, "1.00")],
)
def test_correlation_format(value: float, expected: str) -> None:
    assert format_correlation(value) == expected


def test_generated_block_replacement_is_idempotent() -> None:
    evidence = load_evidence(EVIDENCE)
    block = generated_block(evidence)
    report = f"before\n{START_MARKER}\nstale\n{END_MARKER}\nafter\n"
    updated = replace_generated_block(report, block)
    assert replace_generated_block(updated, block) == updated


def test_generated_block_requires_one_marker_pair() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        replace_generated_block("no markers", "replacement")
