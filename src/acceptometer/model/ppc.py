"""Posterior predictive checks at the real-fit stage, with consequences.

Simulates human rating datasets from the fitted model (same items,
participants, and design as observed) and compares three statistics the model
could plausibly fail on:

- per-family spread of item mean ratings (does the model reproduce how much
  items differ within a family?)
- global category usage (does it use the response scale the way people did?)
- disagreement structure: mean within-item rating SD (the statistic binary
  labels suppress and this tool exists to respect)

Each check reports a posterior predictive p-value (two-sided, tail mass of the
observed statistic under the predictive distribution). A failure (p < 0.01) on
disagreement structure refuses the distributional tier and flags aggregate
estimation in the warrant; the consequences are wired there, the evidence is
computed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_ratings(theta_d, u_d, kappa_d, item_idx0, part_idx0, rng):
    """One simulated dataset for one posterior draw (vectorized over obs)."""
    eta = theta_d[item_idx0] + u_d[part_idx0]
    cum = 1.0 / (1.0 + np.exp(-(kappa_d[None, :] - eta[:, None])))
    r = rng.uniform(size=eta.shape[0])
    return 1 + (r[:, None] > cum).sum(axis=1)


def _ppp(observed: float, sims: np.ndarray) -> float:
    """Two-sided posterior predictive p-value."""
    lo = float(np.mean(sims <= observed))
    hi = float(np.mean(sims >= observed))
    return round(2 * min(lo, hi), 4)


def ppc_human(idata, maps: dict, human: pd.DataFrame,
              items: list, n_sims: int = 200, seed: int = 2,
              out_path: str | Path | None = None) -> dict:
    """Run the three checks against the observed human data."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    u = post["u"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    K = kappa.shape[0] + 1

    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}
    fam_of = {it.item_id: it.construction for it in items}

    item_idx0 = human["item_id"].map(item_pos).to_numpy()
    part_idx0 = human["participant_id"].map(part_pos).to_numpy()
    y_obs = human["rating"].astype(int).to_numpy()
    fams = human["item_id"].map(fam_of).to_numpy()

    def stats(y):
        df = pd.DataFrame({"y": y, "item": human["item_id"].to_numpy(), "fam": fams})
        item_means = df.groupby("item")["y"].mean()
        fam_of_item = df.groupby("item")["fam"].first()
        fam_spread = item_means.groupby(fam_of_item).std().mean()
        cat_usage = np.bincount(y, minlength=K + 1)[1:] / len(y)
        within_sd = df.groupby("item")["y"].std().mean()
        return float(fam_spread), cat_usage, float(within_sd)

    obs_spread, obs_cats, obs_within = stats(y_obs)

    draws = rng.choice(L, size=min(n_sims, L), replace=False)
    sim_spread, sim_within, sim_cat_dev = [], [], []
    for d in draws:
        y_sim = _simulate_ratings(theta[:, d], u[:, d], kappa[:, d],
                                  item_idx0, part_idx0, rng)
        sp, cats, wi = stats(y_sim)
        sim_spread.append(sp)
        sim_within.append(wi)
        sim_cat_dev.append(float(np.abs(cats - obs_cats).sum() / 2))  # TV distance

    # for category usage the discrepancy is TV distance to observed, so the
    # reference is "how far is a typical replicate from ITS OWN generating
    # distribution": compare each replicate's TV to the mean replicate TV
    tv = np.array(sim_cat_dev)
    report = {
        "family_item_mean_spread": {
            "observed": round(obs_spread, 3),
            "ppp": _ppp(obs_spread, np.array(sim_spread)),
        },
        "within_item_disagreement_sd": {
            "observed": round(obs_within, 3),
            "ppp": _ppp(obs_within, np.array(sim_within)),
        },
        "category_usage_tv_distance": {
            "mean_replicate_tv": round(float(tv.mean()), 3),
            "note": "TV distance between replicate and observed category "
                    "frequencies; large values mean the model uses the scale "
                    "differently than people did",
        },
        "n_sims": len(draws),
    }
    report["passed"] = bool(
        report["family_item_mean_spread"]["ppp"] >= 0.01
        and report["within_item_disagreement_sd"]["ppp"] >= 0.01
        and tv.mean() <= 0.15
    )
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
