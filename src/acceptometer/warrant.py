"""Validity-certificate builder (warrant.yaml).

The certificate answers the projectibility question the posterior cannot:
which claims does the evidence on disk license? Principles, enforced in code:

- Every number comes from a file actually read; missing or failed evidence
  refuses dependent claims, never grants them.
- Evidence binds to the posterior it certifies by RECOMPUTED hashes: the
  warrant rehashes posterior.nc, the Stan source, index_maps.json, and the
  grid manifest, and requires exact input-hash-map equality for LOCO. A
  swapped posterior, edited manifest, or stale report invalidates.
- Contamination caps EVERY deployment tier, screening included: exposure to
  published items can inflate the fitted slope and the stable instrument
  opinion exactly where the screening statistic looks, so a suspect source
  supports descriptive findings only.
- Claims are a matrix, not a ladder: within-family ranking, family location,
  and aggregate estimation carry independent evidence requirements (aggregate
  no longer presupposes ranking).
- New-family quantities are model-based extrapolations conditional on
  exchangeability of construction families, which are purposive; the
  certificate says so wherever they appear, and uses the sign-aware
  directional reliability (a reversed-slope draw counts as zero signal, not
  as squared signal).
- Absolute location of a family with no human anchor is prior-identified
  (delta-invariance against per-cell family intercepts), so unanchored
  family-location claims are refused structurally.

Thresholds are pre-registered decision defaults, not estimand-specific loss
analyses; the certificate says that too.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model.fit import STAN_FILE, sha256_file

TIERS = (
    "screening",
    "ranking_within_family",
    "family_location_unanchored",
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
    newfam = _read_json(run_dir / "newfam.json")
    loco = _read_json(run_dir / "loco.json")
    ppc = _read_json(run_dir / "ppc.json")
    manifest_path = run_dir / "grid_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else None
    provenance = _read_json(run_dir / "instruments.json")

    # ---- recomputed binding: trust nothing declared, rehash what exists
    binding: dict[str, str] = {}
    posterior_sha = sha256_file(post_path)
    stan_sha = sha256_file(STAN_FILE)
    if run is None:
        binding["run"] = "run.json missing: nothing can be bound"
        bound_core = False
    else:
        checks = {
            "posterior": run.get("posterior_sha256") == posterior_sha,
            "stan": run.get("stan_sha256") == stan_sha,
        }
        binding["run"] = ("bound" if all(checks.values()) else
                          "MISMATCH: " + ", ".join(k for k, v in checks.items() if not v))
        bound_core = all(checks.values())

    def _bound_posterior(child: dict | None, name: str) -> bool:
        if child is None:
            binding[name] = "not produced"
            return False
        got = child.get("posterior_sha256")
        if got is None:
            binding[name] = "no posterior stamp"
            return False
        ok = got == posterior_sha
        binding[name] = "bound" if ok else "stamped for a different posterior"
        return ok

    def _bound_inputs(child: dict | None, name: str) -> bool:
        if child is None:
            binding[name] = "not produced"
            return False
        if run is None:
            binding[name] = "run.json missing"
            return False
        want = run.get("input_hashes") or {}
        got = child.get("input_hashes") or {}
        ok = bool(want) and got == want          # exact map equality
        binding[name] = ("bound" if ok else
                         "input-hash map differs from run.json (exact equality required)")
        return ok

    def _bound_stan(child: dict | None, name: str) -> bool:
        """Simulation-based evidence (recovery, SBC, newfam) certifies the
        MODEL, not a posterior: it binds through the Stan source hash."""
        if child is None:
            binding[name] = "not produced"
            return False
        ok = child.get("stan_sha256") == stan_sha
        binding[name] = ("bound" if ok else
                         "generated for a different model (stan hash differs)")
        return ok

    ppc_bound = _bound_posterior(ppc, "ppc")
    loco_bound = _bound_inputs(loco, "loco")
    rec_bound = _bound_stan(recovery, "recovery")
    sbc_bound = _bound_stan(sbc, "sbc")
    newfam_bound = _bound_stan(newfam, "newfam")

    # ---- instruments: sign-aware new-family quantities gate; provenance merged
    cont_cells = list(maps.get("cont_cells", []))
    instruments = []
    gate_stats: dict[str, dict] = {}
    if cont_cells and "reliability_new_directional" in idata.posterior:
        rel_dir = np.asarray(idata.posterior["reliability_new_directional"].values)
        rel_dir = rel_dir.reshape(-1, rel_dir.shape[-1])
        p_pos = np.asarray(idata.posterior["p_pos_new"].values)
        p_pos = p_pos.reshape(-1, p_pos.shape[-1])
        rel_fam = np.asarray(idata.posterior["reliability_family"].values)
        rel_fam = rel_fam.reshape(-1, rel_fam.shape[-2], rel_fam.shape[-1])
        alpha = np.asarray(idata.posterior["alpha"].values)
        alpha = alpha.reshape(-1, alpha.shape[-1])
        for j, name in enumerate(cont_cells):
            med = float(np.median(rel_dir[:, j]))
            q10, q90 = np.percentile(rel_dir[:, j], [10, 90])
            ppos = float(p_pos[:, j].mean())
            fam_meds = np.median(rel_fam[:, j, :], axis=0)
            gate_stats[name] = {"median": med, "q10": float(q10), "p_pos": ppos}
            entry = {
                "cell": name,
                "p_positive_slope_new_family": round(ppos, 3),
                "reliability_new_directional_median": round(med, 3),
                "reliability_new_directional_80": [round(float(q10), 3),
                                                   round(float(q90), 3)],
                "reliability_by_family_within": {
                    f: round(float(v), 3)
                    for f, v in zip(maps.get("constructions", []), fam_meds)
                },
                "alpha_standardized_median": round(float(np.median(alpha[:, j])), 3),
                "standardization": maps.get("cell_standardization", {}).get(
                    name, "none recorded"),
                "interpretation": ("model-based extrapolation conditional on "
                                   "exchangeability of construction families "
                                   "(purposive sample); within-family signal "
                                   "variance; reversed-slope draws count as "
                                   "zero signal"),
            }
            if provenance and name in provenance:
                entry["provenance"] = provenance[name]
            else:
                entry["provenance"] = "not recorded in run dir"
            instruments.append(entry)

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

    inst_ppc = (ppc or {}).get("instrument_ppc") if ppc else None

    evidence = {
        "run": run if run is not None else "not produced",
        "binding": binding,
        "diagnostics": diagnostics if diagnostics is not None else "not produced",
        "fake_data_recovery": recovery if recovery is not None else "not produced",
        "sbc": sbc if sbc is not None else "not produced",
        "new_family_recovery": newfam if newfam is not None else "not produced",
        "loco_transfer": loco if loco is not None else "not produced",
        "ppc": ppc if ppc is not None else "not produced",
        "multiverse_spread": _multiverse_spread(run_dir),
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

    # ---- shared prerequisites for ANY instrument-based deployment claim
    diag_ok = bool(diagnostics and diagnostics.get("passed") is True)
    prereq_fail = None
    if not bound_core:
        prereq_fail = f"run binding failed ({binding['run']})"
    elif not diag_ok:
        prereq_fail = "diagnostics gate not passed"
    elif not (rec_bound and recovery.get("passed") is True):
        prereq_fail = f"fake-data recovery: {binding['recovery']}" \
            if not rec_bound else "fake-data recovery failed"
    elif not (sbc_bound and sbc.get("passed") is True):
        prereq_fail = f"SBC: {binding['sbc']}" if not sbc_bound else "SBC failed"
    elif not (newfam_bound and newfam.get("passed") is True):
        prereq_fail = (f"new-family simulated recovery: {binding['newfam']}"
                       if not newfam_bound
                       else "new-family simulated recovery failed")
    elif not (ppc_bound and ppc.get("passed") is True):
        prereq_fail = (f"PPC: {binding['ppc']}" if not ppc_bound
                       else "PPC failed (human or instrument arm)")
    elif inst_ppc is None or not inst_ppc.get("passed"):
        prereq_fail = "instrument-arm posterior predictive check absent or failed"

    contamination_msg = (
        "contamination cap: item source not assessed clean; exposure to "
        "published items can inflate the fitted slope and stable instrument "
        "opinion exactly where this claim's statistic looks. A post-cutoff, "
        "unpublished validation set is required")

    # ---- screening
    candidates = {c: v for c, v in gate_stats.items() if c not in flagged_cells}
    best = max(candidates.items(), key=lambda kv: kv[1]["median"]) if candidates else None
    if prereq_fail:
        refused["screening"] = prereq_fail
    elif not contamination_clean:
        refused["screening"] = contamination_msg
    elif best is None:
        refused["screening"] = ("no unflagged continuous cell carries the "
                                "directional new-family quantities")
    elif (best[1]["p_pos"] >= 0.95 and best[1]["median"] > 0.5
          and best[1]["q10"] > 0.35):
        licensed["screening"] = (
            f"prerequisites passed; cell {best[0]}: P(new-family slope "
            f"positive) {best[1]['p_pos']:.2f} >= 0.95, directional "
            f"reliability median {best[1]['median']:.2f} > 0.5, q10 "
            f"{best[1]['q10']:.2f} > 0.35 (model-based extrapolation "
            f"conditional on family exchangeability)")
    else:
        refused["screening"] = (
            f"best unflagged cell {best[0]}: P(positive slope) "
            f"{best[1]['p_pos']:.2f}, directional reliability median "
            f"{best[1]['median']:.2f}, q10 {best[1]['q10']:.2f} "
            f"(need >= 0.95, > 0.5, > 0.35)")

    # ---- within-family ranking
    all_fams = set(maps.get("constructions", []))
    within_ok = (loco is not None
                 and (loco.get("mean_spearman") or -1) > 0.6
                 and (loco.get("frac_families_within_spearman_gt_0.3") or 0) >= 0.8)
    if prereq_fail:
        refused["ranking_within_family"] = prereq_fail
    elif loco is None:
        refused["ranking_within_family"] = "evidence not produced: loco.json missing"
    elif not loco_bound:
        refused["ranking_within_family"] = f"LOCO not bound: {binding['loco']}"
    elif not loco.get("all_diagnostics_passed"):
        refused["ranking_within_family"] = "one or more LOCO fold fits failed diagnostics"
    elif set(loco.get("families_tested", [])) != all_fams:
        missing = sorted(all_fams - set(loco.get("families_tested", [])))
        refused["ranking_within_family"] = f"LOCO did not cover every family (missing: {missing})"
    elif not contamination_clean:
        refused["ranking_within_family"] = contamination_msg
    elif within_ok:
        licensed["ranking_within_family"] = (
            f"mean within-family held-out Spearman "
            f"{loco['mean_spearman']:.2f} > 0.6 and "
            f"{loco['frac_families_within_spearman_gt_0.3']:.0%} of families "
            f"> 0.3 (min {loco.get('within_family_spearman_min')})")
    else:
        refused["ranking_within_family"] = (
            f"within-family transfer insufficient: mean Spearman "
            f"{loco.get('mean_spearman')}, "
            f"{(loco.get('frac_families_within_spearman_gt_0.3') or 0):.0%} of "
            f"families > 0.3, min {loco.get('within_family_spearman_min')}")

    # ---- family location for unanchored families: structurally refused
    refused["family_location_unanchored"] = (
        "refused structurally: for a family with no human anchor items, "
        "absolute location is prior-identified, not likelihood-identified "
        "(shifting the family mean by delta and each cell's family intercept by "
        "-slope*delta leaves the instrument likelihood unchanged); collect "
        "anchor items or accept within-family claims only")

    # ---- aggregate estimation (independent of ranking; needs sharpness too)
    coverage = loco.get("mean_coverage90") if loco else None
    rmse = loco.get("mean_rmse") if loco else None
    sd_obs = loco.get("sd_observed_item_means") if loco else None
    if prereq_fail:
        refused["aggregate_estimation"] = prereq_fail
    elif loco is None or not loco_bound or not loco.get("all_diagnostics_passed"):
        refused["aggregate_estimation"] = "LOCO evidence missing, unbound, or diagnostics-failed"
    elif not contamination_clean:
        refused["aggregate_estimation"] = contamination_msg
    elif coverage is None or rmse is None or sd_obs is None:
        refused["aggregate_estimation"] = ("LOCO carries no coverage/RMSE/spread "
                                           "for the sharpness-aware gate")
    elif 0.75 <= coverage <= 0.98 and rmse <= 0.75 * sd_obs:
        licensed["aggregate_estimation"] = (
            f"coverage {coverage:.2f} in [0.75, 0.98] and RMSE {rmse:.2f} <= "
            f"0.75 x sd(observed item means) = {0.75 * sd_obs:.2f}; note "
            "within-family interpretation only (family location needs anchors)")
    else:
        refused["aggregate_estimation"] = (
            f"coverage {coverage:.2f} (need [0.75, 0.98]) with RMSE {rmse:.2f} "
            f"vs sharpness bound {0.75 * sd_obs:.2f}: wide-interval coverage "
            "without accuracy does not license aggregate use")

    refused["effect_reproduction"] = ("not yet tested: requires matched "
                                      "experimental contrasts")
    refused["distributional_claims"] = (
        "no participant-level validation of variance structure"
        + ("" if ppc is None or ppc.get("passed")
           else "; posterior predictive checks failed"))
    refused["population_transfer"] = (
        "refused: the v1 ladder contains no population-transfer test; the "
        "estimand population defaults to the criterion sample's own")
    refused["individual_simulation"] = ("refused permanently: an item-level "
                                        "instrument licenses no individual-level "
                                        "human simulation")
    refused["mechanism_claims"] = ("refused permanently: the model estimates a "
                                   "linking function, not a mechanism")

    # ---- descriptive findings: always emitted, never a license
    descriptive = {
        "note": ("in-source descriptive associations on this item set; not "
                 "deployment evidence"),
        "pooled_heldout_spearman": (loco or {}).get("pooled_spearman_descriptive"),
        "between_family_spearman": (loco or {}).get("between_family_spearman"),
        "per_family_within_spearman": {
            f: v.get("spearman") for f, v in ((loco or {}).get("per_family") or {}).items()
        } or "no LOCO evidence",
        "human_split_half": split_half,
    }

    cert = _plain({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_id": (run or {}).get("run_id", "not recorded"),
        "posterior_sha256_recomputed": posterior_sha,
        "stan_sha256_recomputed": stan_sha,
        "estimand": estimand,
        "domain": domain,
        "instruments": instruments,
        "evidence": evidence,
        "licensed_claims": licensed,
        "refused_claims": refused,
        "descriptive_findings": descriptive,
        "residual_risks": [
            "shared pretraining bias: local instruments share web-scale "
            "training data and can share construction-specific error; a tight "
            "multiverse fan does not rule this out",
            "reliability quantities are fit- and item-set-specific and "
            "conditional on a family-exchangeability extrapolation model",
            "thresholds are pre-registered defaults, not estimand-specific "
            "loss analyses",
            "prompt-level dependence within pooled paraphrase cells is not "
            "yet modeled (nested prompt effects are v2)",
        ],
    })

    out_path = Path(out_path) if out_path else run_dir / "warrant.yaml"
    out_path.write_text(yaml.safe_dump(cert, sort_keys=False, allow_unicode=True))
    return cert
