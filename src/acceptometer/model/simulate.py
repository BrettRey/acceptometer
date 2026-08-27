"""Fake-data generator mirroring acceptometer.stan exactly.

Gelman's rule: the pipeline is untrusted until it recovers known truth.
This module simulates the full generative process (latent thetas, human
ordinal ratings with participant effects, continuous and binary instrument
cells with known bias/slope/noise), then `recovery_check` fits the Stan model
to the simulation and gates on parameter recovery and interval coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SimTruth:
    theta: np.ndarray
    mu_c: np.ndarray
    tau_constr: float
    tau_item: float
    tau_a: float
    tau_b: float
    kappa: np.ndarray
    sigma_u: float
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    sigma_s: np.ndarray
    omega: np.ndarray
    a_b: np.ndarray
    b_b: np.ndarray
    g_b: np.ndarray
    to_dict: dict = field(default_factory=dict)


def _ordered_logistic_rng(eta: np.ndarray, kappa: np.ndarray, rng) -> np.ndarray:
    """Draw ordinal outcomes 1..K given linear predictor eta and cutpoints."""
    # P(y <= k) = logistic(kappa_k - eta)
    cum = 1.0 / (1.0 + np.exp(-(kappa[None, :] - eta[:, None])))  # (N, K-1)
    u = rng.uniform(size=eta.shape[0])
    return 1 + (u[:, None] > cum).sum(axis=1)  # in 1..K


def simulate(
    n_constr: int = 8,
    items_per_constr: int = 12,
    n_part: int = 40,
    ratings_per_item: int = 12,
    K: int = 7,
    M_c: int = 3,
    M_b: int = 2,
    P: int = 2,
    tau_constr: float = 0.8,
    tau_item: float = 1.2,
    sigma_u: float = 0.5,
    seed: int = 20260827,
) -> tuple[dict, SimTruth]:
    """Returns (stan_data, truth). Scores are generated on the standardized
    scale (alpha near 0), matching what fit.build_stan_data produces for real
    data, so recovery is checked against the parameters as given."""
    rng = np.random.default_rng(seed)
    N = n_constr * items_per_constr
    constr = np.repeat(np.arange(1, n_constr + 1), items_per_constr)

    mu_c = rng.normal(0, tau_constr, n_constr)
    mu_c -= mu_c.mean()  # sum-to-zero, as in the Stan parameterization
    theta = mu_c[constr - 1] + rng.normal(0, tau_item, N)

    X = rng.normal(0, 1, (N, P))  # covariates arrive centered/scaled

    # human arm: each item rated by `ratings_per_item` random participants
    kappa = np.sort(rng.normal(0, 1.5, K - 1))
    u = rng.normal(0, sigma_u, n_part)
    item_h, part_h = [], []
    for i in range(1, N + 1):
        for p in rng.choice(n_part, size=ratings_per_item, replace=False):
            item_h.append(i)
            part_h.append(p + 1)
    item_h = np.array(item_h)
    part_h = np.array(part_h)
    y = _ordered_logistic_rng(theta[item_h - 1] + u[part_h - 1], kappa, rng)

    # continuous cells: every cell scores every item once; the linking function
    # carries family-level intercept/slope deviations (LLM error clusters by
    # phenomenon), mirroring a_dev/b_dev in the Stan model
    alpha = rng.normal(0, 0.2, M_c)
    beta = rng.uniform(0.3, 0.9, M_c)          # informative but imperfect instruments
    tau_a_true, tau_b_true = 0.15, 0.10
    a_dev = rng.normal(0, tau_a_true, (M_c, n_constr))
    b_dev = rng.normal(0, tau_b_true, (M_c, n_constr))
    gamma = rng.normal(0, 0.3, (M_c, P))
    omega_true = rng.uniform(0.2, 0.4, M_c)
    e_item = rng.normal(0, 1, (M_c, N)) * omega_true[:, None]
    sigma_s = rng.uniform(0.4, 0.9, M_c)
    reps_c = 3  # replicated cells, so the instrument-by-item error is identified
    item_c = np.tile(np.arange(1, N + 1), M_c * reps_c)
    cell_c = np.repeat(np.arange(1, M_c + 1), N * reps_c)
    fam_c = constr[item_c - 1]
    nu = alpha[cell_c - 1] + a_dev[cell_c - 1, fam_c - 1] + \
        (beta[cell_c - 1] + b_dev[cell_c - 1, fam_c - 1]) * theta[item_c - 1] + \
        np.einsum("np,np->n", gamma[cell_c - 1], X[item_c - 1]) + \
        e_item[cell_c - 1, item_c - 1]
    s = rng.normal(nu, sigma_s[cell_c - 1])

    # binary cells: every cell judges every item 5 times (repeat draws)
    a_b = rng.normal(0, 0.8, M_b)
    b_b = rng.uniform(0.5, 1.5, M_b)
    g_b = rng.normal(0, 0.3, (M_b, P))
    reps = 5
    item_b = np.tile(np.arange(1, N + 1), M_b * reps)
    cell_b = np.repeat(np.arange(1, M_b + 1), N * reps)
    eta = a_b[cell_b - 1] + b_b[cell_b - 1] * theta[item_b - 1] + \
        np.einsum("np,np->n", g_b[cell_b - 1], X[item_b - 1])
    z = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    data = dict(
        prior_only=0,
        N_item=N, N_constr=n_constr, constr=constr.tolist(),
        P=P, X=X.tolist(),
        N_h=len(y), K=K, item_h=item_h.tolist(), N_part=n_part,
        part_h=part_h.tolist(), y=y.tolist(),
        is_new=[0] * n_constr,
        N_c=len(s), M_c=M_c, item_c=item_c.tolist(), cell_c=cell_c.tolist(),
        s=s.tolist(), has_reps=[1] * M_c,
        N_b=len(z), M_b=M_b, item_b=item_b.tolist(), cell_b=cell_b.tolist(),
        z=z.tolist(),
    )
    truth = SimTruth(theta, mu_c, tau_constr, tau_item, tau_a_true, tau_b_true,
                     kappa, sigma_u, alpha, beta, gamma, sigma_s, omega_true,
                     a_b, b_b, g_b)
    return data, truth


def recovery_check(idata, truth: SimTruth) -> dict:
    """Compare posterior to simulation truth. Returns a report dict with a
    boolean `passed`. Gates: theta 90% CI coverage in [0.80, 0.98]; every
    beta_m inside its 95% CI; reliability estimates within 0.15 of truth."""
    post = idata.posterior
    report = {}

    th = post["theta"].stack(d=("chain", "draw")).values  # (N, draws)
    lo, hi = np.percentile(th, [5, 95], axis=1)
    cover = float(np.mean((truth.theta >= lo) & (truth.theta <= hi)))
    report["theta_90ci_coverage"] = round(cover, 3)
    report["theta_post_mean_corr_truth"] = round(
        float(np.corrcoef(th.mean(axis=1), truth.theta)[0, 1]), 3)

    be = post["beta"].stack(d=("chain", "draw")).values
    blo, bhi = np.percentile(be, [2.5, 97.5], axis=1)
    beta_ok = bool(np.all((truth.beta >= blo) & (truth.beta <= bhi)))
    report["beta_within_95ci"] = beta_ok
    report["beta_post_mean"] = [round(v, 3) for v in be.mean(axis=1)]
    report["beta_truth"] = [round(v, 3) for v in truth.beta]

    for name in ("tau_item", "tau_a", "tau_b"):
        draws = post[name].stack(d=("chain", "draw")).values
        truths = np.atleast_1d(getattr(truth, name))
        draws2 = draws if draws.ndim == 2 else draws[None, :]
        ok = True
        for j, tv in enumerate(np.broadcast_to(truths, (draws2.shape[0],))):
            lo, hi = np.percentile(draws2[j], [2.5, 97.5])
            ok = ok and (lo <= tv <= hi)
        report[f"{name}_within_95ci"] = bool(ok)
        report[f"{name}_post_median"] = [round(float(np.median(d)), 3) for d in draws2]

    rel = post["reliability"].stack(d=("chain", "draw")).values.mean(axis=1)
    v_theta = truth.theta.var()
    rel_truth = truth.beta**2 * v_theta / (
        truth.beta**2 * v_theta + truth.omega**2 + truth.sigma_s**2)
    rel_err = float(np.max(np.abs(rel - rel_truth)))
    report["reliability_max_abs_err"] = round(rel_err, 3)
    report["reliability_truth"] = [round(v, 3) for v in rel_truth]

    report["passed"] = bool(
        0.80 <= cover <= 0.98 and beta_ok and rel_err <= 0.15
        and report["tau_item_within_95ci"] and report["tau_a_within_95ci"]
        and report["tau_b_within_95ci"])
    return report


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def newfam_check(n_new: int = 2, seed: int = 99, **fit_kw) -> dict:
    """Simulated-LOCO validation of the new-family branch under known truth:
    simulate, mark n_new families NEW (excluded from the sum-to-zero vector),
    drop their human observations, fit, and check the quantities the warrant
    would actually use for a new family. Gates: held-item theta 90% coverage
    in [0.75, 0.98] AFTER family-mean centering (absolute family location is
    prior-identified without an anchor -- the delta-invariance with a_dev --
    so the calibrated claim is the within-family one); within-family rank
    recovery positive in every held family; and the family-location bias is
    REPORTED, not gated, as the empirical face of the prior-identification
    limit."""
    from .fit import fit_model, diagnostics_gate

    data, truth = simulate(seed=seed)
    n_constr = data["N_constr"]
    new_fams = list(range(n_constr - n_new + 1, n_constr + 1))
    constr = np.array(data["constr"])
    held_items = {i + 1 for i in range(data["N_item"]) if constr[i] in new_fams}

    keep = [j for j, it in enumerate(data["item_h"]) if it not in held_items]
    data = dict(data)
    data["item_h"] = [data["item_h"][j] for j in keep]
    data["part_h"] = [data["part_h"][j] for j in keep]
    data["y"] = [data["y"][j] for j in keep]
    data["N_h"] = len(keep)
    data["is_new"] = [1 if c in new_fams else 0 for c in range(1, n_constr + 1)]

    fit, idata = fit_model(data, seed=seed, **fit_kw)
    diag = diagnostics_gate(fit, idata)
    post = idata.posterior
    th = post["theta"].stack(d=("chain", "draw")).values

    report = {"n_new_families": n_new, "diagnostics": diag, "per_family": {}}
    cover_all, rank_ok = [], True
    for c in new_fams:
        idx = np.flatnonzero(constr == c)
        t_true = truth.theta[idx]
        t_draws = th[idx]                                   # (n_items, L)
        # center within family, per draw, to test the identified claim
        t_draws_c = t_draws - t_draws.mean(axis=0, keepdims=True)
        t_true_c = t_true - t_true.mean()
        lo, hi = np.percentile(t_draws_c, [5, 95], axis=1)
        cov = float(np.mean((t_true_c >= lo) & (t_true_c <= hi)))
        cover_all.append(cov)
        pm = t_draws.mean(axis=1)
        rho = float(np.corrcoef(
            np.argsort(np.argsort(pm)), np.argsort(np.argsort(t_true)))[0, 1])
        rank_ok = rank_ok and rho > 0
        loc_bias = float(t_draws.mean() - t_true.mean())
        report["per_family"][f"fam{c}"] = {
            "theta_within_coverage90": round(cov, 3),
            "within_rank_corr_truth": round(rho, 3),
            "family_location_bias": round(loc_bias, 3),
        }
    mean_cov = float(np.mean(cover_all))
    report["mean_within_coverage90"] = round(mean_cov, 3)
    report["passed"] = bool(diag["passed"] and 0.75 <= mean_cov <= 0.98 and rank_ok)
    from .fit import STAN_FILE, sha256_file
    report["stan_sha256"] = sha256_file(STAN_FILE)
    report["note"] = ("family_location_bias is reported unGated: absolute "
                      "location of an unanchored family is prior-identified "
                      "(delta-invariance with per-cell family intercepts)")
    return report
