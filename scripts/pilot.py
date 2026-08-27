"""Run the Sprouse-LI pilot end to end against the current model.

Usage: uv run python scripts/pilot.py [--skip-loco]

Reads data/pilot_items.jsonl, data/pilot_human.csv, and the cached
measurements in runs/pilot/measurements.jsonl; refits; runs recovery, PPC,
and LOCO; leaves every warrant prerequisite in runs/pilot/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.items import load_items, nuisance_covariates
from acceptometer.elicit.base import load_measurements, CONTINUOUS
from acceptometer.model.fit import (build_stan_data, fit_model, diagnostics_gate,
                                    save_fit, sha256_file)
from acceptometer.model.simulate import simulate, recovery_check, write_report
from acceptometer.model.ppc import ppc_human
from acceptometer.model.loco import loco

RUN = Path("runs/pilot")


def fit_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One instrument per model+method: Pythia via SLOR, qwen via pooled
    scalar; binary cells stay descriptive (see DECISIONS 2026-08-27)."""
    cont = df[df.kind == CONTINUOUS].copy()
    cont = cont[(cont.cell_id == "pythia-160m/slor")
                | cont.cell_id.str.startswith("qwen3:8b/prompt_scalar")]
    cont["cell_id"] = cont.cell_id.str.replace(r"/p\d$", "", regex=True)
    return cont[["item_id", "cell_id", "value"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loco", action="store_true")
    args = ap.parse_args()

    items = load_items("data/pilot_items.jsonl")
    human = pd.read_csv("data/pilot_human.csv")
    ms = load_measurements(RUN / "measurements.jsonl")
    df = pd.DataFrame([{"item_id": m.item_id, "cell_id": m.cell_id,
                        "kind": m.kind, "value": m.value} for m in ms])
    cont = fit_cells(df)
    X = np.array(nuisance_covariates(items))
    input_hashes = {
        "items": sha256_file("data/pilot_items.jsonl"),
        "human": sha256_file("data/pilot_human.csv"),
        "measurements": sha256_file(RUN / "measurements.jsonl"),
    }

    # ladder step 1: fake-data recovery against the CURRENT model
    sdata, truth = simulate()
    sfit, sidata = fit_model(sdata, seed=7)
    srec = recovery_check(sidata, truth)
    srec["diagnostics"] = diagnostics_gate(sfit, sidata)
    write_report(srec, RUN / "recovery.json")
    print("recovery passed:", srec["passed"])

    # real fit
    data, maps = build_stan_data(items, X, human, cont, None, K=7)
    fit, idata = fit_model(data, seed=42, iter_warmup=1500, iter_sampling=1500)
    diag = diagnostics_gate(fit, idata)
    save_fit(idata, maps, diag, RUN, data=data, input_hashes=input_hashes)
    cont.to_csv(RUN / "cont.csv", index=False)
    print("fit diag:", json.dumps(diag))

    run = json.loads((RUN / "run.json").read_text())
    rep = ppc_human(idata, maps, human, items, out_path=RUN / "ppc.json",
                    cont=cont, posterior_sha256=run["posterior_sha256"])
    print("PPC passed:", rep["passed"],
          "| marginal:", rep["marginal"]["passed"],
          "| conditional:", rep["conditional"]["passed"])

    if not args.skip_loco:
        # held-out-family geometry mixes slowly; 2000/2000 keeps fold fits
        # inside the production diagnostics gate
        lrep = loco(items, X, human, cont, None, K=7,
                    iter_warmup=2000, iter_sampling=2000,
                    out_path=RUN / "loco.json", input_hashes=input_hashes)
        print("LOCO:", json.dumps({k: v for k, v in lrep.items()
                                   if k not in ("per_family",)}))
    return 0 if diag["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
