"""Leave-one-construction-family-out: the projectibility test.

For each construction family c, refit the model with family c's HUMAN data
removed (instrument scores retained for every item), then predict the held-out
family's human item means from the instrument arm alone, pushed through the
human response model learned on the other families. If the instrument's
validity projects across families, predictions cover the observed means; if it
was memorizing item structure, they won't. Feeds the warrant certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import build_stan_data, fit_model, diagnostics_gate


def _predict_observed_means(idata, item_indices_1based: list[int],
                            n_ratings: list[int], K: int,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive distribution of the OBSERVED mean rating per item,
    for a fresh sample of n_j new participants. Includes participant-effect
    variation and ordinal sampling noise, so interval coverage against the
    observed sample mean is calibrated by construction (an expected-rating
    interval would omit the criterion's own sampling noise and under-cover).
    Returns (mean, lo90, hi90) arrays over items."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values  # (N_item, L)
    kappa = post["kappa"].stack(d=("chain", "draw")).values  # (K-1, L)
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values  # (L,)
    L = theta.shape[1]

    means, los, his = [], [], []
    for pos, i1 in enumerate(item_indices_1based):
        n_j = max(int(n_ratings[pos]), 1)
        th = theta[i1 - 1]                                   # (L,)
        u = rng.normal(0.0, 1.0, (n_j, L)) * sigma_u[None, :]
        eta = th[None, :] + u                                # (n_j, L)
        cum = 1.0 / (1.0 + np.exp(-(kappa.T[None, :, :] - eta[..., None])))  # (n_j, L, K-1)
        r = rng.uniform(size=(n_j, L))
        y = 1 + (r[..., None] > cum).sum(axis=-1)            # (n_j, L) in 1..K
        sample_mean = y.mean(axis=0)                         # (L,)
        means.append(sample_mean.mean())
        lo, hi = np.percentile(sample_mean, [5, 95])
        los.append(lo)
        his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def loco(items: list, X: np.ndarray, human: pd.DataFrame,
         cont: pd.DataFrame | None, binary: pd.DataFrame | None,
         K: int = 7, families: list[str] | None = None,
         out_path: str | Path | None = None,
         iter_warmup: int = 1000, iter_sampling: int = 1000, seed: int = 11) -> dict:
    """Run the LOCO loop. Returns (and optionally writes) a report dict."""
    all_fams = sorted({it.construction for it in items})
    families = families or all_fams
    item_ids = [it.item_id for it in items]
    ix = {iid: i + 1 for i, iid in enumerate(item_ids)}

    per_family = {}
    for fam in families:
        held_items = [it.item_id for it in items if it.construction == fam]
        if not held_items:
            continue
        train_human = human[~human["item_id"].isin(held_items)]
        held = human[human["item_id"].isin(held_items)]
        obs = held.groupby("item_id")["rating"].mean()
        n_ratings = held.groupby("item_id")["rating"].count()
        if obs.empty:
            continue

        data, maps = build_stan_data(items, X, train_human, cont, binary, K=K)
        fit, idata = fit_model(data, seed=seed,
                               iter_warmup=iter_warmup, iter_sampling=iter_sampling)
        diag = diagnostics_gate(fit, idata)

        held_ix = [ix[i] for i in obs.index]
        pred, lo, hi = _predict_observed_means(
            idata, held_ix, [int(n_ratings[i]) for i in obs.index], K)
        o = obs.to_numpy()
        resid = pred - o
        # Spearman without scipy: rank-transform then Pearson
        def _rank(a):
            r = np.empty_like(a)
            r[np.argsort(a)] = np.arange(len(a), dtype=float)
            return r
        if len(o) > 2 and np.std(o) > 0 and np.std(pred) > 0:
            spearman = float(np.corrcoef(_rank(pred), _rank(o))[0, 1])
        else:
            spearman = float("nan")
        per_family[fam] = {
            "n_items": len(o),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "mean_signed_error": round(float(np.mean(resid)), 3),
            "spearman": round(spearman, 3) if spearman == spearman else None,
            "coverage90": round(float(np.mean((o >= lo) & (o <= hi))), 3),
            "diagnostics_passed": diag["passed"],
            "diagnostics": diag,
        }

    fams_ok = [v for v in per_family.values() if v["diagnostics_passed"]]
    report = {
        "per_family": per_family,
        "n_families": len(per_family),
        "mean_rmse": round(float(np.mean([v["rmse"] for v in fams_ok])), 3) if fams_ok else None,
        "mean_spearman": round(float(np.mean(
            [v["spearman"] for v in fams_ok if v["spearman"] is not None])), 3) if fams_ok else None,
        "mean_coverage90": round(float(np.mean([v["coverage90"] for v in fams_ok])), 3) if fams_ok else None,
        "all_diagnostics_passed": bool(fams_ok) and all(v["diagnostics_passed"] for v in per_family.values()),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
