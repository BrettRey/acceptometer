"""Instrument protocol: every elicitation cell implements this.

A *cell* is one (model, method, prompt-variant) combination in the multiverse
grid. Cells return Measurements; they never interpret them. Repeated draws are
repeated measurements of one instrument, never synthetic participants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Protocol, runtime_checkable

from ..items import Item

# measurement kinds the Stan model knows how to absorb
CONTINUOUS = "continuous"   # e.g. logprob_sum, slor, mp_delta, scalar rating
BINARY = "binary"           # e.g. prompted yes/no


@dataclass(frozen=True)
class Measurement:
    item_id: str
    cell_id: str              # e.g. "pythia-160m/logprob_sum" or "qwen3:8b/prompt_binary/p1"
    kind: str                 # CONTINUOUS | BINARY
    value: float              # raw score, or 0/1 for binary
    repeat: int = 0           # index of repeated draw (same instrument)
    meta: dict = field(default_factory=dict)  # temperature, prompt hash, model version, date


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    model: str                # provider-qualified id, e.g. "hf:EleutherAI/pythia-160m"
    method: str               # logprob_sum | logprob_mean | slor | mp_delta | prompt_binary | prompt_scalar | answer_token_prob
    kind: str
    prompt_variant: Optional[str] = None   # registered paraphrase id, prompted cells only
    params: dict = field(default_factory=dict)


@runtime_checkable
class Instrument(Protocol):
    """One backend (HF model, Ollama model, API model) able to run >= 1 cells."""

    def cells(self) -> list[CellSpec]:
        """The cells this instrument can produce."""
        ...

    def score(self, items: list[Item], cell: CellSpec, repeats: int = 1) -> list[Measurement]:
        """Score all items for one cell. Must be deterministic for logprob
        cells; prompted cells honor `repeats` and record temperature in meta."""
        ...


def save_measurements(ms: Iterable[Measurement], path: str | Path) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for m in ms:
            fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")


def load_measurements(path: str | Path) -> list[Measurement]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Measurement(**json.loads(line)))
    return out
