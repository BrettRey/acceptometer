"""Validity-certificate builder (warrant.yaml).

The certificate answers the projectibility question the posterior cannot:
which claim tiers does the evidence on disk license? Every number in the
certificate comes from a file actually read out of the run directory;
missing evidence refuses the dependent tiers, it never grants them.

Files read from run_dir:
  diagnostics.json     required as evidence (missing refuses every tier)
  index_maps.json      required (names for cells, items, families)
  posterior.nc         required (reliability and bias summaries)
  recovery.json        optional (fake-data recovery report)
  loco.json            optional (leave-one-construction-out transfer stats)
  grid_manifest.yaml   optional (contamination record from the grid runner)
  cont.csv             optional (raw continuous scores written by the fit
                       and simulate CLI commands; used for the multiverse
                       spread statistic, which no other artifact carries)
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TIERS = (
    "screening",
    "ranking",
    "aggregate_estimation",
    "effect_reproduction",
    "distributional_claims",
    "individual_simulation",
    "mechanism_claims",
)

_RANK_TOP_KEYS = ("mean_spearman", "mean_heldout_rank_correlation",
                  "mean_rank_correlation", "mean_rank_corr")
_RANK_FOLD_KEYS = ("spearman", "rank_correlation", "rank_corr")
_COV_TOP_KEYS = ("mean_coverage90", "coverage_90", "interval_coverage_90",
                 "coverage")
_COV_FOLD_KEYS = ("coverage90", "coverage_90", "coverage")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _loco_stat(loco: dict, top_keys: tuple, fold_keys: tuple) -> float | None:
    """Pull a scalar from loco.json: a top-level key under any of the known
    names, else the mean of per-fold values. Returns None when the file
    carries no recognizable value (which refuses the dependent tier)."""
    for k in top_keys:
        v = loco.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    folds = loco.get("per_family") or loco.get("families") or loco.get("folds")
    if isinstance(folds, dict):
        folds = list(folds.values())
    vals: list[float] = []
    if isinstance(folds, list):
        for f in folds:
            if not isinstance(f, dict):
                continue
            for k in fold_keys:
                v = f.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
                    break
    return float(np.mean(vals)) if vals else None


def _multiverse_spread(run_dir: Path):
    """Mean over items of the per-item sd of standardized scores across
    continuous cells, from cont.csv. Scores are z-scored per cell here so
    the statistic is scale-free whatever scale the file stores."""
    path = run_dir / "cont.csv"
    if not path.exists():
        return "not assessed (no cont.csv in run dir)"
    df = pd.read_csv(path)
    per = df.groupby(["item_id", "cell_id"], as_index=False)["value"].mean()
    if per["cell_id"].nunique() < 2:
        return "not assessed (fewer than 2 continuous cells)"

    def _z(v: pd.Series) -> pd.Series:
        sd = v.std(ddof=0)
        return (v - v.mean()) / (sd if sd > 0 else 1.0)

    per["z"] = per.groupby("cell_id")["value"].transform(_z)
    pivot = per.pivot(index="item_id", columns="cell_id", values="z")
    sds = pivot.std(axis=1, ddof=1).dropna()
    return {
        "mean_per_item_sd_across_cells": round(float(sds.mean()), 3),
        "n_items": int(len(sds)),
        "n_cells": int(pivot.shape[1]),
    }


def _prompt_invariance(cont_cells: list[str], beta_med: np.ndarray):
    """sd of posterior-median beta across paraphrase cells of the same
    model and method, identified from cell names of the registered form
    model/method/paraphrase. Returns a per-group dict, or a string when no
    such group is identifiable."""
    groups: dict[str, list[int]] = defaultdict(list)
    for j, name in enumerate(cont_cells):
        parts = name.split("/")
        if len(parts) >= 3:
            groups["/".join(parts[:2])].append(j)
    out = {}
    for g, idxs in sorted(groups.items()):
        if len(idxs) >= 2:
            meds = [float(beta_med[j]) for j in idxs]
            out[g] = {
                "sd_of_beta_medians": round(float(np.std(meds, ddof=1)), 3),
                "cells": [cont_cells[j] for j in idxs],
            }
    return out if out else "not identifiable from cell names"


def _plain(x):
    """Recursively convert numpy scalars and paths so yaml.safe_dump accepts
    the certificate."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def build_warrant(run_dir: str | Path, estimand: dict,
                  out_path: str | Path | None = None) -> dict:
    """Build the validity certificate for one fit and write it as YAML to
    run_dir/warrant.yaml (or out_path). Returns the certificate dict.

    Granting is conservative and hierarchical: screening needs the
    diagnostics gate plus at least one cell with reliability median > 0.5;
    ranking additionally needs LOCO mean held-out rank correlation > 0.6;
    aggregate estimation additionally needs LOCO 90% interval coverage in
    [0.75, 0.98]. Effect reproduction and distributional claims are refused
    in v1; individual simulation and mechanism claims are refused
    permanently. A missing evidence file refuses the dependent tiers with
    reason "evidence not produced", never grants them."""
    import arviz as az

    run_dir = Path(run_dir)
    maps_path = run_dir / "index_maps.json"
    post_path = run_dir / "posterior.nc"
    if not maps_path.exists():
        raise FileNotFoundError(f"{maps_path} not found; cannot build a warrant")
    if not post_path.exists():
        raise FileNotFoundError(f"{post_path} not found; cannot build a warrant")
    maps = json.loads(maps_path.read_text())
    idata = az.from_netcdf(str(post_path))

    diagnostics = _read_json(run_dir / "diagnostics.json")
    recovery = _read_json(run_dir / "recovery.json")
    loco = _read_json(run_dir / "loco.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None

    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    rel_meds: list[float] = []
    if cont_cells and "reliability" in idata.posterior:
        rel = np.asarray(idata.posterior["reliability"].values)
        rel = rel.reshape(-1, rel.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        beta = np.asarray(idata.posterior["beta"].values)
        beta = beta.reshape(-1, beta.shape[-1])
        beta_med = np.median(beta, axis=0)
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel[:, j]))
            lo, hi = np.percentile(rel[:, j], [5, 95])
            rel_meds.append(med)
            instruments.append({
                "cell": name,
                "reliability_median": round(med, 3),
                "reliability_90": [round(float(lo), 3), round(float(hi), 3)],
                "bias_alpha_median": round(float(np.median(alpha[:, j])), 3),
            })
        invariance = _prompt_invariance(cont_cells, beta_med)
    else:
        invariance = "not identifiable from cell names"

    estimand = dict(estimand)
    split_half = estimand.pop("human_split_half", "not provided")
    domain = {
        "construction_families": list(maps.get("constructions", [])),
        "n_items": len(maps.get("item_ids", [])),
    }
    user_domain = estimand.pop("domain", None)
    if isinstance(user_domain, dict):
        domain.update(user_domain)

    if isinstance(manifest, dict) and "contamination" in manifest:
        contamination = manifest["contamination"]
    else:
        contamination = "not assessed"

    rank_corr = _loco_stat(loco, _RANK_TOP_KEYS, _RANK_FOLD_KEYS) if loco else None
    coverage = _loco_stat(loco, _COV_TOP_KEYS, _COV_FOLD_KEYS) if loco else None

    evidence = {
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "multiverse_spread": _multiverse_spread(run_dir),
        "prompt_invariance": invariance,
        "human_split_half": split_half,
        "contamination": contamination,
    }

    licensed: dict[str, str] = {}
    refused: dict[str, str] = {}
    diag_ok = bool(diagnostics and diagnostics.get("passed"))
    max_rel = max(rel_meds) if rel_meds else None

    if diagnostics is None:
        refused["screening"] = "evidence not produced: diagnostics.json missing"
    elif not diag_ok:
        refused["screening"] = (
            f"diagnostics gate failed (rhat_max={diagnostics.get('rhat_max')}, "
            f"divergence_rate={diagnostics.get('divergence_rate')}, "
            f"ess_bulk_min={diagnostics.get('ess_bulk_min')})")
    elif max_rel is None:
        refused["screening"] = ("evidence not produced: no continuous cell "
                                "carries a posterior reliability")
    elif max_rel > 0.5:
        licensed["screening"] = (f"diagnostics passed and max cell reliability "
                                 f"median {max_rel:.2f} > 0.5")
    else:
        refused["screening"] = (f"max cell reliability median {max_rel:.2f} "
                                f"does not exceed 0.5")

    if "screening" not in licensed:
        refused["ranking"] = "refused because screening is not granted"
    elif loco is None:
        refused["ranking"] = "evidence not produced: loco.json missing"
    elif rank_corr is None:
        refused["ranking"] = ("evidence not produced: loco.json carries no "
                              "held-out rank correlation")
    elif rank_corr > 0.6:
        licensed["ranking"] = (f"screening granted and LOCO mean held-out rank "
                               f"correlation {rank_corr:.2f} > 0.6")
    else:
        refused["ranking"] = (f"LOCO mean held-out rank correlation "
                              f"{rank_corr:.2f} does not exceed 0.6")

    if "ranking" not in licensed:
        refused["aggregate_estimation"] = "refused because ranking is not granted"
    elif coverage is None:
        refused["aggregate_estimation"] = ("evidence not produced: loco.json "
                                           "carries no 90% interval coverage")
    elif 0.75 <= coverage <= 0.98:
        licensed["aggregate_estimation"] = (f"LOCO 90% interval coverage "
                                            f"{coverage:.2f} within [0.75, 0.98]")
    else:
        refused["aggregate_estimation"] = (f"LOCO 90% interval coverage "
                                           f"{coverage:.2f} outside [0.75, 0.98]")

    refused["effect_reproduction"] = ("not yet tested: requires matched "
                                      "experimental contrasts")
    refused["distributional_claims"] = ("no participant-level validation of "
                                        "variance structure")
    refused["individual_simulation"] = ("refused permanently: an item-level "
                                        "instrument licenses no individual-level "
                                        "human simulation")
    refused["mechanism_claims"] = ("refused permanently: the model estimates a "
                                   "linking function, not a mechanism")

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "estimand": estimand,
        "domain": domain,
        "instruments": instruments,
        "evidence": evidence,
        "licensed_claims": licensed,
        "refused_claims": refused,
    })

    out_path = Path(out_path) if out_path else run_dir / "warrant.yaml"
    out_path.write_text(yaml.safe_dump(cert, sort_keys=False, allow_unicode=True))
    return cert
