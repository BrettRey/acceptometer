"""Item schema: the unit everything else measures.

JSONL on disk, one item per line. Minimal pairs are two items sharing a
``pair_id`` with roles 'good'/'bad'. Construction family is the grouping level
for partial pooling and for leave-one-construction-out generalization tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Item:
    item_id: str
    text: str
    construction: str                 # family label, e.g. "island:whether"
    source: str                       # provenance, e.g. "blimp:npi_present_1" or "simulated"
    pair_id: Optional[str] = None     # shared by the two members of a minimal pair
    pair_role: Optional[str] = None   # 'good' | 'bad' | None
    language: str = "en"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_items(path: str | Path) -> list[Item]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(Item(**d))
    ids = [it.item_id for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate item_id in {path}")
    return items


def save_items(items: Iterable[Item], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")


def nuisance_covariates(items: list[Item]) -> "list[list[float]]":
    """Per-item nuisance covariates for the measurement model: log word count
    and mean unigram log frequency (Zipf scale, wordfreq). Centered and scaled
    by the caller (fit.py) so the Stan priors stay interpretable."""
    from wordfreq import zipf_frequency

    rows = []
    for it in items:
        words = it.text.split()
        log_len = math.log(max(len(words), 1))
        zipfs = [zipf_frequency(w.strip(".,!?;:\"'()").lower(), it.language) for w in words]
        zipfs = [z for z in zipfs if z > 0] or [0.0]
        rows.append([log_len, sum(zipfs) / len(zipfs)])
    return rows
