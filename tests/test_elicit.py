from __future__ import annotations

from dataclasses import dataclass

import pytest

from acceptometer.elicit.base import (
    BINARY,
    CONTINUOUS,
    CellSpec,
    Measurement,
    load_measurements,
)
from acceptometer.elicit.grid import derive_mp_delta, run_grid
from acceptometer.elicit.ollama_chat import parse_binary, parse_scalar
from acceptometer.items import Item


@dataclass
class FakeInstrument:
    calls: int = 0

    def cells(self) -> list[CellSpec]:
        return [
            CellSpec("fake/logprob_sum", "fake:model", "logprob_sum", CONTINUOUS),
            CellSpec(
                "fake/prompt_binary/p1",
                "fake:model",
                "prompt_binary",
                BINARY,
                prompt_variant="p1",
                params={"prompt_sha256": "abc123"},
            ),
        ]

    def score(
        self, items: list[Item], cell: CellSpec, repeats: int = 1
    ) -> list[Measurement]:
        self.calls += 1
        return [
            Measurement(
                item_id=item.item_id,
                cell_id=cell.cell_id,
                kind=cell.kind,
                value=float(index + repeat),
                repeat=repeat,
            )
            for index, item in enumerate(items)
            for repeat in range(repeats)
        ]


def test_run_grid_caches_complete_cells(tmp_path) -> None:
    items = [
        Item("i1", "Birds fly.", "simple", "test"),
        Item("i2", "Fish swim.", "simple", "test"),
    ]
    instrument = FakeInstrument()
    run_grid([instrument], items, tmp_path, repeats_prompted=3)
    first = load_measurements(tmp_path / "measurements.jsonl")
    first_calls = instrument.calls

    run_grid([instrument], items, tmp_path, repeats_prompted=3)
    second = load_measurements(tmp_path / "measurements.jsonl")

    assert len(first) == 8
    assert second == first
    assert instrument.calls == first_calls
    assert (tmp_path / "grid_manifest.yaml").exists()


def test_derive_mp_delta_for_two_pairs() -> None:
    items = [
        Item("g1", "Good one.", "c", "test", "pair1", "good"),
        Item("b1", "Bad one.", "c", "test", "pair1", "bad"),
        Item("b2", "Bad two.", "c", "test", "pair2", "bad"),
        Item("g2", "Good two.", "c", "test", "pair2", "good"),
    ]
    measurements = [
        Measurement("g1", "model/logprob_sum", CONTINUOUS, -2.0),
        Measurement("b1", "model/logprob_sum", CONTINUOUS, -5.5),
        Measurement("g2", "model/logprob_sum", CONTINUOUS, -4.0),
        Measurement("b2", "model/logprob_sum", CONTINUOUS, -4.75),
    ]

    derived = derive_mp_delta(measurements, items)

    assert [(m.item_id, m.cell_id, m.value) for m in derived] == [
        ("g1", "model/mp_delta", 3.5),
        ("g2", "model/mp_delta", 0.75),
    ]
    assert [m.meta["pair_id"] for m in derived] == ["pair1", "pair2"]
    assert all(m.meta["stan_likelihood"] is False for m in derived)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes.", 1),
        ("no way", 0),
        ("  !!! YES", 1),
        ('"No, that is not acceptable."', 0),
        ("<think>blah</think>yes", 1),
        ("", None),
        ("maybe", None),
        ("yesterday", None),
        ("not yes", None),
        ("yes or no", None),
        ("No. Actually, yes.", None),
    ],
)
def test_parse_binary(text: str, expected: int | None) -> None:
    assert parse_binary(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5", 5.0),
        ("I think 5", 5.0),
        ("Rating: 7.", 7.0),
        ("<think>4 might work</think>6", 6.0),
        ("8", None),
        ("0", None),
        ("", None),
        ("five", None),
        ("5.5", None),
        ("score6", None),
        ("8, but perhaps 3", 3.0),
    ],
)
def test_parse_scalar(text: str, expected: float | None) -> None:
    assert parse_scalar(text) == expected
