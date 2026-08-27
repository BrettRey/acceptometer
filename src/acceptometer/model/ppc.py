"""Posterior predictive checks at the real-fit stage, with consequences.

Two participant modes, both reported:

- conditional: replicates reuse the fitted participants' posterior u (checks
  the model against exactly the people observed);
- marginal: replicates draw fresh u from sigma_u (checks the participant-effect
  distribution itself, the one LOCO's fresh-rater predictions rely on).

Discrepancies, per mode:

- per-family spread of item mean ratings (vector of family ppps; the min is
  gated, so opposite family-specific failures cannot cancel into a passing
  average);
- within-item disagreement SD (global and per-family min);
- category usage, as a proper ppp: T(y) = total-variation distance between
  y's category frequencies and the mean replicate frequencies, compared for
  the observed data against the replicate distribution of the same statistic.

Plus an instrument-arm residual check: standardized mean residual per
(cell, family) from the posterior-mean linking; |z| > 3 is flagged (warn-only:
family deviations should absorb real structure, so a flag here means the
fitted deviations did not).

Gates (pre-committed): global ppps >= 0.01, min per-family ppp >= 0.005
(Bonferroni-flavored), category ppp >= 0.01, in BOTH participant modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_ratings(theta_d, u_obs, kappa_d, item_idx0, part_idx0, rng):
    eta = theta_d[item_idx0] + u_obs[part_idx0]
    cum = 1.0 / (1.0 + np.exp(-(kappa_d[None, :] - eta[:, None])))
    r = rng.uniform(size=eta.shape[0])
    return 1 + (r[:, None] > cum).sum(axis=1)


def _ppp(observed: float, sims: np.ndarray) -> float:
    lo = float(np.mean(sims <= observed))
    hi = float(np.mean(sims >= observed))
    return round(2 * min(lo, hi), 4)


def ppc_human(idata, maps: dict, human: pd.DataFrame,
              items: list, n_sims: int = 1000, seed: int = 2,
              out_path: str | Path | None = None,
              cont: pd.DataFrame | None = None,
              posterior_sha256: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    post = idata.posterior
    theta = post["theta"].stack(d=("chain", "draw")).values
    u = post["u"].stack(d=("chain", "draw")).values
    kappa = post["kappa"].stack(d=("chain", "draw")).values
    sigma_u = post["sigma_u"].stack(d=("chain", "draw")).values
    L = theta.shape[1]
    K = kappa.shape[0] + 1

    item_pos = {iid: i for i, iid in enumerate(maps["item_ids"])}
    part_pos = {p: i for i, p in enumerate(maps["participants"])}
    fam_of = {it.item_id: it.construction for it in items}

    item_idx0 = human["item_id"].map(item_pos).to_numpy()
    part_idx0 = human["participant_id"].map(part_pos).to_numpy()
    y_obs = human["rating"].astype(int).to_numpy()
    fam_col = human["item_id"].map(fam_of).to_numpy()
    fams = sorted(set(fam_col))

    def stats(y):
        df = pd.DataFrame({"y": y, "item": human["item_id"].to_numpy(),
                           "fam": fam_col})
        item_means = df.groupby("item")["y"].mean()
        fam_of_item = df.groupby("item")["fam"].first()
        spread_by_fam = item_means.groupby(fam_of_item).std()
        within = df.groupby("item")["y"].std()
        within_by_fam = within.groupby(fam_of_item).mean()
        cats = np.bincount(y, minlength=K + 1)[1:] / len(y)
        return (spread_by_fam.reindex(fams).to_numpy(),
                float(within.mean()),
                within_by_fam.reindex(fams).to_numpy(),
                cats)

    obs_spread, obs_within, obs_within_fam, obs_cats = stats(y_obs)

    draws = rng.choice(L, size=min(n_sims, L), replace=True)
    report: dict = {"n_sims": len(draws), "families": fams}
    if posterior_sha256:
        report["posterior_sha256"] = posterior_sha256
    all_pass = True

    for mode in ("conditional", "marginal"):
        sp, wi, wif, cats_list = [], [], [], []
        for d in draws:
            if mode == "conditional":
                u_d = u[:, d]
            else:
                u_d = rng.normal(0.0, sigma_u[d], u.shape[0])
            y_sim = _simulate_ratings(theta[:, d], u_d, kappa[:, d],
                                      item_idx0, part_idx0, rng)
            a, b, c, k = stats(y_sim)
            sp.append(a); wi.append(b); wif.append(c); cats_list.append(k)
        sp = np.array(sp); wi = np.array(wi); wif = np.array(wif)
        cats_arr = np.array(cats_list)

        fam_ppps = {f: _ppp(obs_spread[j], sp[:, j]) for j, f in enumerate(fams)
                    if obs_spread[j] == obs_spread[j]}
        wif_ppps = {f: _ppp(obs_within_fam[j], wif[:, j]) for j, f in enumerate(fams)
                    if obs_within_fam[j] == obs_within_fam[j]}
        # proper predictive reference for category usage: compare T(obs) with
        # the replicate distribution of the SAME statistic
        ref = cats_arr.mean(axis=0)
        t_obs = float(np.abs(obs_cats - ref).sum() / 2)
        t_rep = np.abs(cats_arr - ref[None, :]).sum(axis=1) / 2
        cat_ppp = float(np.mean(t_rep >= t_obs))

        mode_report = {
            "family_spread_ppp": fam_ppps,
            "family_spread_ppp_min": min(fam_ppps.values()),
            "within_item_sd_global_ppp": _ppp(obs_within, wi),
            "within_item_sd_family_ppp_min": min(wif_ppps.values()),
            "category_usage_ppp": round(cat_ppp, 4),
        }
        mode_pass = (mode_report["within_item_sd_global_ppp"] >= 0.01
                     and mode_report["family_spread_ppp_min"] >= 0.005
                     and mode_report["within_item_sd_family_ppp_min"] >= 0.005
                     and mode_report["category_usage_ppp"] >= 0.01)
        mode_report["passed"] = bool(mode_pass)
        report[mode] = mode_report
        all_pass = all_pass and mode_pass

    # instrument-arm residual check (warn-only)
    if cont is not None and len(cont) and "beta" in post:
        flags = []
        beta = post["beta"].mean(dim=("chain", "draw")).values
        alpha = post["alpha"].mean(dim=("chain", "draw")).values
        a_dev = post["a_dev"].mean(dim=("chain", "draw")).values
        b_dev = post["b_dev"].mean(dim=("chain", "draw")).values
        gamma = post["gamma"].mean(dim=("chain", "draw")).values
        sigma_s = post["sigma_s"].mean(dim=("chain", "draw")).values
        th = theta.mean(axis=1)
        Xs = np.asarray(maps.get("X_standardized")) if maps.get("X_standardized") else None
        cix = {c: j for j, c in enumerate(maps["cont_cells"])}
        constr_of = {iid: fam_of[iid] for iid in maps["item_ids"]}
        fam_ix = {f: j for j, f in enumerate(maps["constructions"])}
        for (cell, fam), grp in cont.assign(
                fam=cont["item_id"].map(constr_of)).groupby(["cell_id", "fam"]):
            m = cix.get(cell)
            if m is None:
                continue
            c = fam_ix[fam]
            ii = grp["item_id"].map(item_pos).to_numpy()
            mu = (alpha[m] + a_dev[m, c] + (beta[m] + b_dev[m, c]) * th[ii])
            if Xs is not None:
                mu = mu + Xs[ii] @ gamma[m]
            resid = grp["value"].to_numpy() - mu
            z = resid.mean() / (sigma_s[m] / np.sqrt(len(resid)))
            if abs(z) > 3:
                flags.append({"cell": cell, "family": fam, "z": round(float(z), 2)})
        report["instrument_residual_flags"] = flags
    report["passed"] = bool(all_pass)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
