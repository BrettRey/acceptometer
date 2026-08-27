"""Sentinel set management and epoch drift checks.

A model version is not a fixed instrument: the same cells re-scored months
apart can shift. A fixed sentinel item set is scored per epoch, and
`drift_check` compares epochs cell by cell. A shift above half an epoch-a
standard deviation flags the instrument for recalibration; its earlier
warrant does not carry forward.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from .elicit.base import Measurement
from .items import Item


def select_sentinels(items: list[Item], k: int = 30, seed: int = 1) -> list[Item]:
    """Deterministic stratified sentinel selection: items are grouped by
    construction family, each group shuffled with `seed`, then drawn
    round-robin across families (sorted by name) until `k` items are
    chosen. Same items, k, and seed always yield the same set."""
    by_constr: dict[str, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda it: it.item_id):
        by_constr[it.construction].append(it)
    rng = random.Random(seed)
    queues = []
    for c in sorted(by_constr):
        grp = by_constr[c]
        rng.shuffle(grp)
        queues.append(grp)

    out: list[Item] = []
    while queues and len(out) < k:
        remaining = []
        for grp in queues:
            if len(out) >= k:
                break
            out.append(grp.pop(0))
            if grp:
                remaining.append(grp)
        queues = remaining
    return out


def _cell_item_means(ms: list[Measurement]) -> dict[str, dict[str, float]]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for m in ms:
        acc[(m.cell_id, m.item_id)].append(m.value)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (cell, item), vals in acc.items():
        out[cell][item] = float(np.mean(vals))
    return out


def drift_check(measurements_a: list[Measurement],
                measurements_b: list[Measurement]) -> dict:
    """Compare two scoring epochs of the sentinel set. For each cell_id
    present in both epochs, computes the mean shift between epochs in
    units of the epoch-a standard deviation (over per-item means of the
    items shared by both epochs). Returns per-cell shifts, the max
    absolute shift, and `flagged: True` if any absolute shift exceeds
    0.5. Repeats are averaged per (cell, item) first; a zero epoch-a sd
    falls back to raw units (divisor 1.0)."""
    a = _cell_item_means(measurements_a)
    b = _cell_item_means(measurements_b)
    shared = sorted(set(a) & set(b))

    cells: dict[str, dict] = {}
    max_abs = 0.0
    for cell in shared:
        items = sorted(set(a[cell]) & set(b[cell]))
        if not items:
            continue
        va = np.array([a[cell][i] for i in items])
        vb = np.array([b[cell][i] for i in items])
        sd_a = float(va.std(ddof=0))
        shift = float((vb.mean() - va.mean()) / (sd_a if sd_a > 0 else 1.0))
        cells[cell] = {
            "n_items": len(items),
            "mean_a": round(float(va.mean()), 4),
            "mean_b": round(float(vb.mean()), 4),
            "sd_a": round(sd_a, 4),
            "shift": round(shift, 4),
        }
        max_abs = max(max_abs, abs(shift))

    return {
        "shared_cells": list(cells),
        "cells": cells,
        "max_abs_shift": round(max_abs, 4),
        "flagged": bool(max_abs > 0.5),
    }
