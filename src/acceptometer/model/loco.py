"""Leave-one-construction-family-out: the family-axis projectibility test.

For each construction family c, refit the model with (a) family c's HUMAN data
removed and (b) family c marked NEW, so its family effect and linking
deviations come from their predictive distributions rather than the training
sum-to-zero vector (a held-out family inside that vector inherits information
through the finite-set centering). Instrument scores for family c's items are
retained: they are the transfer mechanism under test.

Evaluation target (stated, not implied): SAME participants, NEW items. The
observed criterion is the mean rating given by the actual raters of each
held-out item, so the predictive uses those raters' posterior effects where
they are known from training items, and fresh draws otherwise.

Standardization is training-only: per-cell constants are computed from
training-family measurements and applied unchanged to held-out scores, so a
held-family location or scale shift is confronted, not absorbed.

Rank transfer is reported tie-aware, pooled across all held-out items, with a
family-cluster bootstrap lower bound; per-family values are descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import build_stan_data, fit_model, diagnostics_gate


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman: average ranks, then Pearson."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _predict_observed_means_same_raters(idata, maps: dict, held: pd.DataFrame,
                                        K: int, seed: int = 0
                                        ) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive of each held-out item's observed mean rating for
    its ACTUAL raters: raters known from training items use their posterior
    u draws; raters absent from training draw fresh u from sigma_u. Includes
    ordinal sampling noise, so coverage against the observed sample mean is
    calibrated for the same-participants-new-items target."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values
    u_post = post["u"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}

    items = held.groupby("item_id")
    ids, means, los, his = [], [], [], []
    for iid, grp in items:
        th = theta[item_pos[iid]]                        # (L,)
        raters = list(grp["participant_id"])
        u_rows = []
        for r in raters:
            if r in part_pos:
                u_rows.append(u_post[part_pos[r]])
            else:
                u_rows.append(rng.normal(0.0, 1.0, L) * sigma_u)
        u = np.stack(u_rows)                             # (n_j, L)
        eta = th[None, :] + u
        cum = 1.0 / (1.0 + np.exp(-(kappa.T[None, :, :] - eta[..., None])))
        draw = rng.uniform(size=(len(raters), L))
        y = 1 + (draw[..., None] > cum).sum(axis=-1)     # (n_j, L)
        sm = y.mean(axis=0)
        ids.append(iid)
        means.append(sm.mean())
        lo, hi = np.percentile(sm, [5, 95])
        los.append(lo)
        his.append(hi)
    return pd.Index(ids), np.array(means), np.array(los), np.array(his)


def _standardize_training_only(cont: pd.DataFrame, train_items: set) -> tuple[pd.DataFrame, dict]:
    out = cont.copy()
    constants = {}
    for cell, grp in cont.groupby("cell_id"):
        tr = grp[grp["item_id"].isin(train_items)]["value"]
        mu, sd = float(tr.mean()), float(tr.std() or 1.0)
        constants[cell] = {"mean": mu, "sd": sd}
        m = out["cell_id"] == cell
        out.loc[m, "value"] = (out.loc[m, "value"] - mu) / sd
    return out, constants


def loco(items: list, X: np.ndarray, human: pd.DataFrame,
         cont: pd.DataFrame | None, binary: pd.DataFrame | None,
         K: int = 7, families: list[str] | None = None,
         out_path: str | Path | None = None,
         iter_warmup: int = 1000, iter_sampling: int = 1000, seed: int = 11,
         input_hashes: dict | None = None) -> dict:
    """Run the LOCO loop. Returns (and optionally writes) a report dict."""
    all_fams = sorted({it.construction for it in items})
    families = families or all_fams
    fam_of = {it.item_id: it.construction for it in items}

    per_family = {}
    pooled_pred, pooled_obs, pooled_fam = [], [], []
    for fam in families:
        held_items = {it.item_id for it in items if it.construction == fam}
        train_items = {it.item_id for it in items} - held_items
        train_human = human[~human["item_id"].isin(held_items)]
        held = human[human["item_id"].isin(held_items)]
        obs = held.groupby("item_id")["rating"].mean()
        if obs.empty:
            continue

        cont_std, _ = (_standardize_training_only(cont, train_items)
                       if cont is not None and len(cont) else (None, {}))
        data, maps = build_stan_data(items, X, train_human, cont_std, binary,
                                     K=K, standardize_scores=False,
                                     new_families={fam})
        fit, idata = fit_model(data, seed=seed,
                               iter_warmup=iter_warmup, iter_sampling=iter_sampling)
        diag = diagnostics_gate(fit, idata)

        ids, pred, lo, hi = _predict_observed_means_same_raters(idata, maps, held, K)
        o = obs.reindex(ids).to_numpy()
        resid = pred - o
        per_family[fam] = {
            "n_items": len(o),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "mean_signed_error": round(float(np.mean(resid)), 3),
            "spearman": (round(_spearman(pred, o), 3)
                         if _spearman(pred, o) == _spearman(pred, o) else None),
            "coverage90": round(float(np.mean((o >= lo) & (o <= hi))), 3),
            "diagnostics_passed": diag["passed"],
            "diagnostics": diag,
        }
        pooled_pred.extend(pred.tolist())
        pooled_obs.extend(o.tolist())
        pooled_fam.extend([fam] * len(o))

    fams_ok = [v for v in per_family.values() if v["diagnostics_passed"]]
    pooled_pred = np.array(pooled_pred)
    pooled_obs = np.array(pooled_obs)
    pooled_fam = np.array(pooled_fam)

    pooled_rho = _spearman(pooled_pred, pooled_obs)
    # family-cluster bootstrap: rank transfer uncertainty at the level the
    # sampling actually happened (families are the purposive units)
    rng = np.random.default_rng(seed)
    ufams = np.unique(pooled_fam)
    boots = []
    for _ in range(2000):
        pick = rng.choice(ufams, size=len(ufams), replace=True)
        idx = np.concatenate([np.flatnonzero(pooled_fam == f) for f in pick])
        r = _spearman(pooled_pred[idx], pooled_obs[idx])
        if r == r:
            boots.append(r)
    lower90 = float(np.percentile(boots, 10)) if boots else None

    report = {
        "target": "same participants, new items",
        "per_family": per_family,
        "n_families": len(per_family),
        "families_tested": sorted(per_family.keys()),
        "pooled_spearman": round(pooled_rho, 3) if pooled_rho == pooled_rho else None,
        "pooled_spearman_cluster_boot_lower90": (round(lower90, 3)
                                                 if lower90 is not None else None),
        "mean_spearman": round(float(np.mean(
            [v["spearman"] for v in fams_ok if v["spearman"] is not None])), 3) if fams_ok else None,
        "mean_rmse": round(float(np.mean([v["rmse"] for v in fams_ok])), 3) if fams_ok else None,
        "mean_coverage90": round(float(np.mean([v["coverage90"] for v in fams_ok])), 3) if fams_ok else None,
        "all_diagnostics_passed": bool(fams_ok) and all(
            v["diagnostics_passed"] for v in per_family.values()),
        "input_hashes": input_hashes or {},
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
