"""Validity-certificate builder (warrant.yaml).

The certificate answers the projectibility question the posterior cannot:
which claim tiers does the evidence on disk license? Every number in the
certificate comes from a file actually read out of the run directory; missing
or failed evidence refuses the dependent tiers, it never grants them; and
evidence is refused unless it is BOUND to the posterior it claims to certify
(via run.json hashes), so a stale or copied report cannot license a different
fit.

The validation ladder is enforced literally:
  screening            needs diagnostics + fake-data recovery + SBC, all
                       passed, plus an unflagged cell whose NEW-FAMILY
                       predictive reliability clears the gate (the global
                       slope ratio is not a warrant quantity: a family slope
                       deviation changes how much signal that family's scores
                       carry).
  ranking              additionally needs LOCO: present, bound, every family
                       tested, all fold diagnostics passed, pooled tie-aware
                       Spearman > 0.6 with family-cluster bootstrap lower-90
                       > 0.5, and contamination assessed clean (LOCO rewards
                       contamination, so a suspect item source caps ranking
                       too).
  aggregate_estimation additionally needs the PPC (present, bound, both
                       participant modes passed) and LOCO coverage in band.
  everything above     refused in v1 with reasons; individual simulation and
                       mechanism claims permanently.

Thresholds are pre-registered decision defaults, not estimand-specific loss
analyses; the certificate says so.
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
    "population_transfer",
    "individual_simulation",
    "mechanism_claims",
)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _multiverse_spread(run_dir: Path):
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


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _bound(child: dict | None, run: dict | None, how: str) -> tuple[bool, str]:
    """Check an evidence file's binding to the run manifest."""
    if child is None:
        return False, "evidence not produced"
    if run is None:
        return False, "run.json missing; evidence cannot be bound to a posterior"
    if how == "posterior":
        want = run.get("posterior_sha256")
        got = child.get("posterior_sha256")
        if got is None:
            return False, "evidence carries no posterior_sha256 stamp"
        if got != want:
            return False, "evidence is stamped for a different posterior"
    elif how == "inputs":
        want = run.get("input_hashes") or {}
        got = child.get("input_hashes") or {}
        if not got:
            return False, "evidence carries no input_hashes stamp"
        shared = set(want) & set(got)
        if not shared:
            return False, "evidence and run share no input hashes"
        for k in shared:
            if want[k] != got[k]:
                return False, f"input hash mismatch on {k}"
    return True, "bound"


def build_warrant(run_dir: str | Path, estimand: dict,
                  out_path: str | Path | None = None) -> dict:
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

    run = _read_json(run_dir / "run.json")
    diagnostics = _read_json(run_dir / "diagnostics.json")
    recovery = _read_json(run_dir / "recovery.json")
    sbc = _read_json(run_dir / "sbc.json")
    loco = _read_json(run_dir / "loco.json")
    ppc = _read_json(run_dir / "ppc.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None

    # ---- instruments: new-family predictive reliability is the gate quantity
    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    rel_new_stats: dict[str, dict] = {}
    if cont_cells and "reliability_new" in idata.posterior:
        rel_new = np.asarray(idata.posterior["reliability_new"].values)
        rel_new = rel_new.reshape(-1, rel_new.shape[-1])
        rel_glob = np.asarray(idata.posterior["reliability"].values)
        rel_glob = rel_glob.reshape(-1, rel_glob.shape[-1])
        rel_fam = np.asarray(idata.posterior["reliability_family"].values)
        rel_fam = rel_fam.reshape(-1, rel_fam.shape[-2], rel_fam.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel_new[:, j]))
            q10, q90 = np.percentile(rel_new[:, j], [10, 90])
            fam_meds = np.median(rel_fam[:, j, :], axis=0)
            rel_new_stats[name] = {"median": med, "q10": float(q10)}
            entry = {
                "cell": name,
                "reliability_new_family_median": round(med, 3),
                "reliability_new_family_80": [round(float(q10), 3), round(float(q90), 3)],
                "reliability_by_family": {
                    f: round(float(v), 3)
                    for f, v in zip(maps.get("constructions", []), fam_meds)
                },
                "global_slope_signal_ratio_median": round(
                    float(np.median(rel_glob[:, j])), 3),
                "alpha_standardized_median": round(float(np.median(alpha[:, j])), 3),
                "standardization": maps.get("cell_standardization", {}).get(name,
                                                                            "none recorded"),
            }
            instruments.append(entry)

    # prompt invariance from paraphrase-labeled cell names, if any
    groups: dict[str, list[str]] = defaultdict(list)
    for name in cont_cells:
        parts = name.split("/")
        if len(parts) >= 3:
            groups["/".join(parts[:2])].append(name)
    invariance = ("assessed descriptively at the grid level; paraphrases enter "
                  "the fit as one instrument, so no model-based invariance "
                  "statistic exists in this fit" if not groups else groups)

    estimand = dict(estimand)
    split_half = estimand.pop("human_split_half", "not provided")
    domain = {
        "construction_families": list(maps.get("constructions", [])),
        "n_items_total": (run or {}).get("n_items_total",
                                         len(maps.get("item_ids", []))),
        "n_items_with_human_criterion": (run or {}).get("n_items_criterion",
                                                        "not recorded"),
    }
    user_domain = estimand.pop("domain", None)
    if isinstance(user_domain, dict):
        domain.update(user_domain)
    if "population" not in estimand:
        estimand["population"] = ("the criterion sample's population "
                                  "(unspecified); population transfer untested")

    if isinstance(manifest, dict) and "contamination" in manifest:
        contamination = manifest["contamination"]
    else:
        contamination = "not assessed"
    contamination_clean = (
        contamination == "clean"
        or (isinstance(contamination, dict)
            and contamination.get("status") == "clean"))

    nonresponse = (manifest or {}).get("nonresponse") if isinstance(manifest, dict) else None
    flagged_cells = sorted(
        c for c, r in (nonresponse or {}).items()
        if isinstance(r, (int, float)) and r > 0.10)

    loco_ok, loco_bind_reason = _bound(loco, run, "inputs")
    ppc_ok_bind, ppc_bind_reason = _bound(ppc, run, "posterior")

    evidence = {
        "run": run if run is not None else "not produced",
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "sbc": sbc if sbc is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "loco_binding": loco_bind_reason,
        "ppc": ppc if ppc is not None else "not produced",
        "ppc_binding": ppc_bind_reason,
        "multiverse_spread": _multiverse_spread(run_dir),
        "prompt_invariance": invariance,
        "human_split_half": split_half,
        "contamination": contamination,
        "instrument_nonresponse": (
            nonresponse if nonresponse is not None else "not recorded"),
        "flagged_cells": flagged_cells or "none",
        "generalization_axes": {
            "tested": (["construction_family (LOCO-CV, purposive family "
                        "sample; descriptive over the families tested)"]
                       if loco is not None else []),
            "untested": ["population", "register", "language", "item_source",
                         "time", "model_version (beyond drift sentinels)"]
                        + ([] if loco is not None else ["construction_family"]),
        },
        "threshold_status": ("pre-registered decision defaults; not yet "
                             "estimand-specific loss analyses"),
    }

    licensed: dict[str, str] = {}
    refused: dict[str, str] = {}

    # ---- screening: full ladder prerequisite + unflagged predictive reliability
    diag_ok = bool(diagnostics and diagnostics.get("passed") is True)
    rec_ok = bool(recovery and recovery.get("passed") is True)
    sbc_ok = bool(sbc and sbc.get("passed") is True)
    candidates = {c: v for c, v in rel_new_stats.items() if c not in flagged_cells}
    best = max(candidates.items(), key=lambda kv: kv[1]["median"]) if candidates else None

    if diagnostics is None:
        refused["screening"] = "evidence not produced: diagnostics.json missing"
    elif not diag_ok:
        refused["screening"] = (
            f"diagnostics gate failed (rhat_max={diagnostics.get('rhat_max')}, "
            f"divergence_rate={diagnostics.get('divergence_rate')}, "
            f"ess_bulk_min={diagnostics.get('ess_bulk_min')})")
    elif recovery is None:
        refused["screening"] = "evidence not produced: recovery.json missing (ladder step 1)"
    elif not rec_ok:
        refused["screening"] = "fake-data recovery failed (ladder step 1)"
    elif sbc is None:
        refused["screening"] = "evidence not produced: sbc.json missing (ladder step 2)"
    elif not sbc_ok:
        refused["screening"] = "SBC failed (ladder step 2)"
    elif best is None:
        refused["screening"] = ("no unflagged continuous cell carries a "
                                "new-family predictive reliability")
    elif best[1]["median"] > 0.5 and best[1]["q10"] > 0.35:
        licensed["screening"] = (
            f"full ladder passed and cell {best[0]} new-family predictive "
            f"reliability median {best[1]['median']:.2f} > 0.5 with "
            f"q10 {best[1]['q10']:.2f} > 0.35")
    else:
        refused["screening"] = (
            f"best unflagged cell {best[0]}: new-family predictive reliability "
            f"median {best[1]['median']:.2f}, q10 {best[1]['q10']:.2f} "
            f"(need median > 0.5 and q10 > 0.35)")

    # ---- ranking
    all_fams = set(maps.get("constructions", []))
    if "screening" not in licensed:
        refused["ranking"] = "refused because screening is not granted"
    elif loco is None:
        refused["ranking"] = "evidence not produced: loco.json missing"
    elif not loco_ok:
        refused["ranking"] = f"LOCO evidence not bound to this run: {loco_bind_reason}"
    elif not loco.get("all_diagnostics_passed"):
        refused["ranking"] = "one or more LOCO fold fits failed diagnostics"
    elif set(loco.get("families_tested", [])) != all_fams:
        missing = sorted(all_fams - set(loco.get("families_tested", [])))
        refused["ranking"] = f"LOCO did not cover every family (missing: {missing})"
    elif not contamination_clean:
        refused["ranking"] = (
            "contamination cap: item source not assessed clean; contamination "
            "inflates the held-out rank statistic itself (LOCO rewards it)")
    elif (loco.get("pooled_spearman") or -1) > 0.6 and \
         (loco.get("pooled_spearman_cluster_boot_lower90") or -1) > 0.5:
        licensed["ranking"] = (
            f"pooled tie-aware held-out Spearman "
            f"{loco['pooled_spearman']:.2f} > 0.6 with family-cluster "
            f"bootstrap lower-90 {loco['pooled_spearman_cluster_boot_lower90']:.2f} > 0.5")
    else:
        refused["ranking"] = (
            f"pooled held-out Spearman {loco.get('pooled_spearman')} "
            f"(lower-90 {loco.get('pooled_spearman_cluster_boot_lower90')}) "
            "does not clear (0.6, 0.5)")

    # ---- aggregate estimation
    coverage = loco.get("mean_coverage90") if loco else None
    if "ranking" not in licensed:
        refused["aggregate_estimation"] = "refused because ranking is not granted"
    elif ppc is None:
        refused["aggregate_estimation"] = "evidence not produced: ppc.json missing (ladder step 4)"
    elif not ppc_ok_bind:
        refused["aggregate_estimation"] = f"PPC evidence not bound to this run: {ppc_bind_reason}"
    elif not ppc.get("passed"):
        refused["aggregate_estimation"] = (
            "posterior predictive check failed; the human arm misfits the "
            "criterion data, so aggregate predictions inherit unquantified bias")
    elif coverage is None:
        refused["aggregate_estimation"] = "loco.json carries no 90% interval coverage"
    elif 0.75 <= coverage <= 0.98:
        licensed["aggregate_estimation"] = (
            f"LOCO 90% interval coverage {coverage:.2f} within [0.75, 0.98], "
            "PPC passed in both participant modes, contamination assessed clean")
    else:
        refused["aggregate_estimation"] = (
            f"LOCO 90% interval coverage {coverage:.2f} outside [0.75, 0.98]")

    refused["effect_reproduction"] = ("not yet tested: requires matched "
                                      "experimental contrasts")
    refused["distributional_claims"] = (
        "no participant-level validation of variance structure"
        + ("" if ppc is None or ppc.get("passed")
           else "; posterior predictive check failed"))
    refused["population_transfer"] = (
        "refused: the v1 ladder contains no population-transfer test; the "
        "estimand population defaults to the criterion sample's own")
    refused["individual_simulation"] = ("refused permanently: an item-level "
                                        "instrument licenses no individual-level "
                                        "human simulation")
    refused["mechanism_claims"] = ("refused permanently: the model estimates a "
                                   "linking function, not a mechanism")

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_id": (run or {}).get("run_id", "not recorded"),
        "posterior_sha256": (run or {}).get("posterior_sha256", "not recorded"),
        "estimand": estimand,
        "domain": domain,
        "instruments": instruments,
        "evidence": evidence,
        "licensed_claims": licensed,
        "refused_claims": refused,
        "residual_risks": [
            "shared pretraining bias: local instruments share web-scale "
            "training data and can share construction-specific error; a tight "
            "multiverse fan does not rule this out",
            "reliability quantities are fit- and item-set-specific; they go "
            "stale when the item pool changes",
            "thresholds are pre-registered defaults, not estimand-specific "
            "loss analyses",
        ],
    })

    out_path = Path(out_path) if out_path else run_dir / "warrant.yaml"
    out_path.write_text(yaml.safe_dump(cert, sort_keys=False, allow_unicode=True))
    return cert
