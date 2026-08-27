"""Run the Sprouse-LI pilot end to end against the current model.

Usage: uv run python scripts/pilot.py [--skip-loco]

Every warrant prerequisite is produced (or verified) here, and the script
STOPS at the first failed gate rather than proceeding to later stages: a
partially-run directory must not look like a validated one. --skip-loco
deletes any stale loco.json for the same reason.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from acceptometer.items import load_items, nuisance_covariates
from acceptometer.elicit.base import load_measurements, CONTINUOUS
from acceptometer.model.fit import (build_stan_data, fit_model, diagnostics_gate,
                                    save_fit, sha256_file, STAN_FILE)
from acceptometer.model.simulate import (simulate, recovery_check, write_report,
                                         newfam_check)
from acceptometer.model.sbc import sbc_run
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


def write_provenance(ms, out: Path) -> None:
    """Collapse measurement metadata into per-fitted-cell provenance so the
    certificate can name the model installation it certifies."""
    prov: dict = {}
    for m in ms:
        cell = m.cell_id
        cell = cell.rsplit("/p", 1)[0] if "/prompt_" in cell else cell
        entry = prov.setdefault(cell, {"revisions": set(), "digests": set(),
                                       "dates": set()})
        if m.meta.get("revision"):
            entry["revisions"].add(m.meta["revision"])
        if m.meta.get("model_digest"):
            entry["digests"].add(str(m.meta["model_digest"])[:200])
        if m.meta.get("date"):
            entry["dates"].add(m.meta["date"][:10])
    out.write_text(json.dumps(
        {c: {k: sorted(v) for k, v in e.items()} for c, e in prov.items()},
        indent=2))


def fail(stage: str) -> int:
    print(f"STOP: {stage} failed; later stages not run")
    return 1


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
    write_provenance(ms, RUN / "instruments.json")

    # ladder step 1: fake-data recovery against the CURRENT model
    sdata, truth = simulate()
    sfit, sidata = fit_model(sdata, seed=7)
    srec = recovery_check(sidata, truth)
    srec["diagnostics"] = diagnostics_gate(sfit, sidata)
    srec["stan_sha256"] = sha256_file(STAN_FILE)
    write_report(srec, RUN / "recovery.json")
    print("recovery passed:", srec["passed"])
    if not srec["passed"]:
        return fail("recovery")

    # ladder step 2: SBC (reuse a current run only if it matches this model)
    sbc_path = RUN / "sbc.json"
    sbc = json.loads(sbc_path.read_text()) if sbc_path.exists() else None
    if not (sbc and sbc.get("stan_sha256") == srec["stan_sha256"]):
        print("running SBC (no current-model report found)...")
        sbc = sbc_run(R=100, seed=17, out_path=sbc_path)
    print("SBC passed:", sbc["passed"], "| failed fits:",
          sbc.get("n_failed_fits"), "+", sbc.get("n_diag_failed"), "diag")
    if not sbc["passed"]:
        return fail("SBC")

    # ladder step 2b: simulated new-family recovery (the branch the warrant uses)
    nf = newfam_check(iter_warmup=2000, iter_sampling=2000)
    (RUN / "newfam.json").write_text(json.dumps(nf, indent=2))
    print("new-family recovery passed:", nf["passed"])
    if not nf["passed"]:
        return fail("new-family recovery")

    # real fit
    data, maps = build_stan_data(items, X, human, cont, None, K=7)
    fit, idata = fit_model(data, seed=42, iter_warmup=1500, iter_sampling=1500)
    diag = diagnostics_gate(fit, idata)
    save_fit(idata, maps, diag, RUN, data=data, input_hashes=input_hashes)
    cont.to_csv(RUN / "cont.csv", index=False)
    print("fit diag:", json.dumps(diag))
    if not diag["passed"]:
        return fail("fit diagnostics")

    # auditability: auxiliary posterior summaries the report quotes
    post = idata.posterior
    hm = human.groupby("item_id").rating.mean()
    q = cont[cont.cell_id == "qwen3:8b/prompt_scalar"].groupby("item_id").value.mean()
    common = q.index.intersection(hm.index)
    summary = {
        "tau_item_median": round(float(post.tau_item.median()), 3),
        "omega_median_by_cell": {
            c: round(float(post.omega.median(dim=("chain", "draw")).values[j]), 3)
            for j, c in enumerate(maps["cont_cells"])},
        "qwen_pooled_scalar_item_mean_r_with_human": round(
            float(np.corrcoef(q[common], hm[common])[0, 1]), 3),
        "posterior_sha256": json.loads((RUN / "run.json").read_text())["posterior_sha256"],
    }
    (RUN / "posterior_summary.json").write_text(json.dumps(summary, indent=2))

    run = json.loads((RUN / "run.json").read_text())
    rep = ppc_human(idata, maps, human, items, out_path=RUN / "ppc.json",
                    cont=cont, posterior_sha256=run["posterior_sha256"])
    print("PPC passed:", rep["passed"],
          "| marginal:", rep["marginal"]["passed"],
          "| conditional:", rep["conditional"]["passed"],
          "| instrument:", rep.get("instrument_ppc", {}).get("passed"))
    if not rep["passed"]:
        return fail("PPC")

    if args.skip_loco:
        stale = RUN / "loco.json"
        if stale.exists():
            stale.unlink()
            print("removed stale loco.json (--skip-loco)")
        return 0

    # held-out-family geometry mixes slowly; 2000/2000 keeps fold fits
    # inside the production diagnostics gate
    lrep = loco(items, X, human, cont, None, K=7,
                iter_warmup=2000, iter_sampling=2000,
                out_path=RUN / "loco.json", input_hashes=input_hashes)
    print("LOCO:", json.dumps({k: v for k, v in lrep.items()
                               if k not in ("per_family", "input_hashes")}))
    return 0 if lrep["all_diagnostics_passed"] else fail("LOCO fold diagnostics")


if __name__ == "__main__":
    raise SystemExit(main())
