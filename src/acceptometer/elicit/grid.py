"""Sequential elicitation-grid runner with JSONL-level caching."""

from __future__ import annotations

import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import yaml

from ..items import Item
from .base import (
    CONTINUOUS,
    Instrument,
    Measurement,
    load_measurements,
    save_measurements,
)


def derive_mp_delta(
    measurements: list[Measurement], items: list[Item]
) -> list[Measurement]:
    """Derive descriptive good-minus-bad scores for complete minimal pairs."""
    pairs: dict[str, dict[str, Item]] = {}
    for item in items:
        if item.pair_id is not None and item.pair_role in {"good", "bad"}:
            pairs.setdefault(item.pair_id, {})[item.pair_role] = item

    sums = {
        (measurement.cell_id, measurement.item_id): measurement.value
        for measurement in measurements
        if measurement.cell_id.endswith("/logprob_sum")
        and measurement.repeat == 0
    }
    cells = sorted({cell_id for cell_id, _ in sums})
    out = []
    for pair_id, roles in pairs.items():
        if set(roles) != {"good", "bad"}:
            continue
        good = roles["good"]
        bad = roles["bad"]
        for cell_id in cells:
            good_key = (cell_id, good.item_id)
            bad_key = (cell_id, bad.item_id)
            if good_key not in sums or bad_key not in sums:
                continue
            model = cell_id.removesuffix("/logprob_sum")
            out.append(
                Measurement(
                    item_id=good.item_id,
                    cell_id=f"{model}/mp_delta",
                    kind=CONTINUOUS,
                    value=sums[good_key] - sums[bad_key],
                    meta={
                        "pair_id": pair_id,
                        "pair_level": True,
                        "stan_likelihood": False,
                        "note": (
                            "v1: descriptive only, not yet in the Stan likelihood"
                        ),
                    },
                )
            )
    return out


def run_grid(
    instruments: Iterable[Instrument],
    items: list[Item],
    out_dir: str | Path,
    repeats_prompted: int = 5,
) -> None:
    """Run every cell sequentially and append only uncached measurements."""
    started = datetime.now(timezone.utc).isoformat()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    measurements_path = output / "measurements.jsonl"
    instrument_list = list(instruments)
    instrument_cells = [(instrument, instrument.cells()) for instrument in instrument_list]

    for instrument, cells in instrument_cells:
        for cell in cells:
            existing = _load_if_present(measurements_path)
            cached = {
                (measurement.item_id, measurement.cell_id, measurement.repeat)
                for measurement in existing
            }
            repeats = (
                repeats_prompted
                if cell.method in {"prompt_binary", "prompt_scalar"}
                else 1
            )
            pending_items = [
                item
                for item in items
                if any(
                    (item.item_id, cell.cell_id, repeat) not in cached
                    for repeat in range(repeats)
                )
            ]
            if not pending_items:
                continue
            fresh = instrument.score(pending_items, cell, repeats=repeats)
            fresh = [
                measurement
                for measurement in fresh
                if (
                    measurement.item_id,
                    measurement.cell_id,
                    measurement.repeat,
                )
                not in cached
            ]
            if fresh:
                save_measurements(fresh, measurements_path)

    existing = _load_if_present(measurements_path)
    cached = {
        (measurement.item_id, measurement.cell_id, measurement.repeat)
        for measurement in existing
    }
    derived = [
        measurement
        for measurement in derive_mp_delta(existing, items)
        if (measurement.item_id, measurement.cell_id, measurement.repeat) not in cached
    ]
    if derived:
        save_measurements(derived, measurements_path)

    all_cells = [cell for _, cells in instrument_cells for cell in cells]
    manifest = {
        "instruments": [
            {
                "class": type(instrument).__name__,
                "models": sorted({cell.model for cell in cells}),
            }
            for instrument, cells in instrument_cells
        ],
        "cells": [asdict(cell) for cell in all_cells],
        "item_count": len(items),
        "dates": {
            "started": started,
            "completed": datetime.now(timezone.utc).isoformat(),
        },
        "prompt_hashes": {
            cell.cell_id: cell.params["prompt_sha256"]
            for cell in all_cells
            if "prompt_sha256" in cell.params
        },
        "package_versions": _package_versions(),
    }
    with (output / "grid_manifest.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)


def _load_if_present(path: Path) -> list[Measurement]:
    return load_measurements(path) if path.exists() else []


def _package_versions() -> dict[str, str | None]:
    packages = ("acceptometer", "httpx", "PyYAML", "torch", "transformers", "wordfreq")
    found: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found
