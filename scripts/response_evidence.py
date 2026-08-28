"""Response-style and marginal-stratum evidence (adversarial-pragmatics
lessons applied to acceptability).

Two failure modes that aggregate correlations conceal:

- MARGINAL-STRATUM performance: the contested middle band of items (human
  mean in [3, 5] on a 7-point scale) is where theoretical decisions live and
  where restriction-of-range inflation does not reach. Per-cell correlations
  are reported inside and outside the band.
- CATEGORY OMISSION: an instrument can reach an aggregate advantage partly by
  never emitting certain response categories (the adversarial-pragmatics
  judge study's label-omission finding). Human vs instrument category usage
  is compared directly for prompted scalar cells.

Writes runs/pilot/response_style.json; the warrant surfaces it under
descriptive_findings.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.elicit.base import load_measurements, CONTINUOUS

RUN = Path("runs/pilot")


def main() -> int:
    human = pd.read_csv("data/pilot_human.csv")
    hm = human.groupby("item_id").rating.mean()
    ms = load_measurements(RUN / "measurements.jsonl")
    multi = Path("runs/multi/measurements.jsonl")
    if multi.exists():
        ms += load_measurements(multi)
    df = pd.DataFrame([{"item_id": m.item_id, "cell_id": m.cell_id,
                        "kind": m.kind, "value": m.value} for m in ms])

    out: dict = {"marginal_band": [3.0, 5.0], "cells": {}}
    cont = df[df.kind == CONTINUOUS].copy()
    cont["cell_id"] = cont.cell_id.str.replace(r"/p\d$", "", regex=True)
    for cell, grp in cont.groupby("cell_id"):
        v = grp.groupby("item_id").value.mean()
        common = v.index.intersection(hm.index)
        h = hm[common]
        v = v[common]
        marg = (h >= 3.0) & (h <= 5.0)
        entry = {"n_items": int(len(common)),
                 "r_all": round(float(np.corrcoef(v, h)[0, 1]), 3)}
        for label, mask in (("r_marginal_band", marg), ("r_outside_band", ~marg)):
            if mask.sum() >= 5 and h[mask].std() > 0 and v[mask].std() > 0:
                entry[label] = round(float(np.corrcoef(v[mask], h[mask])[0, 1]), 3)
                entry[label + "_n"] = int(mask.sum())
            else:
                entry[label] = "not computable (too few items or no variance)"
        out["cells"][cell] = entry

    # category usage: human vs prompted scalar (raw 1-7 draws, all repeats)
    hu_usage = (human.rating.value_counts(normalize=True)
                .reindex(range(1, 8), fill_value=0.0))
    out["category_usage"] = {
        "human": {str(k): round(float(p), 3) for k, p in hu_usage.items()}}
    scal = df[df.cell_id.str.contains("prompt_scalar")]  # includes batch cells
    for cell, grp in scal.groupby(scal.cell_id.str.replace(r"/p\d$", "", regex=True)):
        usage = (grp.value.astype(int).value_counts(normalize=True)
                 .reindex(range(1, 8), fill_value=0.0))
        omitted = [k for k, p in usage.items() if p == 0.0]
        out["category_usage"][cell] = {
            "usage": {str(k): round(float(p), 3) for k, p in usage.items()},
            "categories_never_emitted": omitted or "none",
        }

    (RUN / "response_style.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
