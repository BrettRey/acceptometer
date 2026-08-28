from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.frontier_judge import (
    JUDGES,
    JudgeCallError,
    claude_result,
    parse_ratings,
)


def test_requested_frontier_judges_are_exact() -> None:
    assert {key: spec.model for key, spec in JUDGES.items()} == {
        "opus-5": "claude-opus-5",
        "gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-terra": "gpt-5.6-terra",
        "gpt-5.6-luna": "gpt-5.6-luna",
    }
    assert len({spec.cell_id for spec in JUDGES.values()}) == 4


@pytest.mark.parametrize(
    ("text", "n", "expected"),
    [
        ('{"1": 5, "2": 2}', 2, {1: 5.0, 2: 2.0}),
        ('preamble {"1": 4} then {"1": 6}', 1, {1: 6.0}),
        ('{"1": 5}', 2, None),
        ('{"1": 5, "2": 8}', 2, None),
        ('{"1": 5, "2": 2.5}', 2, None),
        ('{"1": true, "2": 2}', 2, None),
        ("not JSON", 1, None),
    ],
)
def test_parse_ratings_requires_complete_integer_scale(
    text: str, n: int, expected: dict[int, float] | None
) -> None:
    assert parse_ratings(text, n) == expected


def test_claude_result_verifies_exact_model() -> None:
    envelope = json.dumps(
        {
            "is_error": False,
            "result": '{"1": 5}',
            "modelUsage": {"claude-opus-5": {"inputTokens": 1}},
        }
    )
    assert claude_result(envelope, "claude-opus-5") == '{"1": 5}'


def test_claude_result_refuses_unreported_model() -> None:
    envelope = json.dumps(
        {
            "is_error": False,
            "result": '{"1": 5}',
            "modelUsage": {"claude-opus-4-6": {"inputTokens": 1}},
        }
    )
    with pytest.raises(JudgeCallError, match="requested model"):
        claude_result(envelope, "claude-opus-5")
