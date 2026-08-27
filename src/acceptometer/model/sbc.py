"""SBC-lite: simulation-based calibration rank checks.

Unlike simulate.py (which uses convenient truth distributions for a single
recovery check), SBC draws every truth from the model's OWN priors, fits, and
records the rank of the truth within thinned posterior draws. If the pipeline
is self-consistent, ranks are uniform. We track the global scales and the
instrument slopes/noises; ~R=40 small replications is enough to catch gross
miscalibration (full SBC with many more replications is documented future work).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fit import fit_model
from .simulate import _ordered_logistic_rng


def _draw_prior(rng, n_constr, K, M_c, M_b, P):
    """Draw hyperparameters from exactly the priors in acceptometer.stan."""
    x = rng.normal(0, 1, n_constr)
    mu_c_raw = x - x.mean()                       # projected iid normal = sum_to_zero prior
    tau_constr = abs(rng.normal(0, 1))
    tau_item = rng.gamma(2.0, 1.0)
    kappa = np.sort(rng.normal(0, 2, K - 1))      # order statistics of iid normals
    sigma_u = abs(rng.normal(0, 1))
    # sigma_s has a 0.05 floor in the model: rejection-sample the truncation
    sigma_s = np.empty(M_c)
    for m in range(M_c):
        v = abs(rng.normal(0, 1))
        while v < 0.05:
            v = abs(rng.normal(0, 1))
        sigma_s[m] = v
    tau_a = abs(rng.normal(0, 0.5, M_c))
    tau_b = abs(rng.normal(0, 0.5, M_c))
    omega = abs(rng.normal(0, 0.5, M_c))
    return dict(
        mu_c=tau_constr * mu_c_raw,
        tau_constr=tau_constr,
        tau_item=tau_item,
        kappa=kappa,
        sigma_u=sigma_u,
        alpha=rng.normal(0, 1, M_c),
        beta=rng.normal(0, 1, M_c),
        tau_a=tau_a,
        tau_b=tau_b,
        a_dev=rng.normal(0, 1, (M_c, n_constr)) * tau_a[:, None],
        b_dev=rng.normal(0, 1, (M_c, n_constr)) * tau_b[:, None],
        gamma=rng.normal(0, 0.5, (M_c, P)),
        omega=omega,
        sigma_s=sigma_s,
        a_b=rng.normal(0, 1.5, M_b),
        b_b=rng.normal(0, 1, M_b),
        g_b=rng.normal(0, 0.5, (M_b, P)),
    )


def _simulate_given(pr, rng, n_constr, items_per_constr, n_part, ratings_per_item,
                    K, M_c, M_b, P):
    N = n_constr * items_per_constr
    e_item = rng.normal(0, 1, (M_c, N)) * pr["omega"][:, None]
    constr = np.repeat(np.arange(1, n_constr + 1), items_per_constr)
    theta = pr["mu_c"][constr - 1] + rng.normal(0, pr["tau_item"], N)
    X = rng.normal(0, 1, (N, P))

    u = rng.normal(0, pr["sigma_u"], n_part)
    item_h, part_h = [], []
    for i in range(1, N + 1):
        for p in rng.choice(n_part, size=min(ratings_per_item, n_part), replace=False):
            item_h.append(i); part_h.append(p + 1)
    item_h = np.array(item_h); part_h = np.array(part_h)
    y = _ordered_logistic_rng(theta[item_h - 1] + u[part_h - 1], pr["kappa"], rng)

    reps_c = 3
    item_c = np.tile(np.arange(1, N + 1), M_c * reps_c)
    cell_c = np.repeat(np.arange(1, M_c + 1), N * reps_c)
    fam_c = constr[item_c - 1]
    nu = pr["alpha"][cell_c - 1] + pr["a_dev"][cell_c - 1, fam_c - 1] + \
        (pr["beta"][cell_c - 1] + pr["b_dev"][cell_c - 1, fam_c - 1]) * theta[item_c - 1] + \
        np.einsum("np,np->n", pr["gamma"][cell_c - 1], X[item_c - 1]) + \
        e_item[cell_c - 1, item_c - 1]
    s = rng.normal(nu, pr["sigma_s"][cell_c - 1])

    reps = 3
    item_b = np.tile(np.arange(1, N + 1), M_b * reps)
    cell_b = np.repeat(np.arange(1, M_b + 1), N * reps)
    eta = pr["a_b"][cell_b - 1] + pr["b_b"][cell_b - 1] * theta[item_b - 1] + \
        np.einsum("np,np->n", pr["g_b"][cell_b - 1], X[item_b - 1])
    z = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    return dict(
        prior_only=0, N_item=N, N_constr=n_constr, constr=constr.tolist(),
        is_new=[0] * n_constr,
        P=P, X=X.tolist(),
        N_h=len(y), K=K, item_h=item_h.tolist(), N_part=n_part,
        part_h=part_h.tolist(), y=y.tolist(),
        N_c=len(s), M_c=M_c, item_c=item_c.tolist(), cell_c=cell_c.tolist(),
        s=s.tolist(), has_reps=[1] * M_c,
        N_b=len(z), M_b=M_b, item_b=item_b.tolist(), cell_b=cell_b.tolist(),
        z=z.tolist(),
    )


def sbc_run(R: int = 40, n_thin: int = 63, seed: int = 5,
            n_constr: int = 4, items_per_constr: int = 6, n_part: int = 12,
            ratings_per_item: int = 6, K: int = 5, M_c: int = 2, M_b: int = 1,
            P: int = 2, out_path: str | Path | None = None) -> dict:
    """Run R replications; return rank-uniformity report.

    Tracked: tau_constr, sigma_u, beta[m], sigma_s[m]. Gate: chi-square
    uniformity p > 0.005 for every tracked parameter (loose Bonferroni; this is
    a smoke alarm, not a certificate)."""
    rng = np.random.default_rng(seed)
    tracked = ["tau_constr", "tau_item", "sigma_u"] + \
        [f"beta[{m}]" for m in range(1, M_c + 1)] + \
        [f"sigma_s[{m}]" for m in range(1, M_c + 1)] + \
        [f"tau_b[{m}]" for m in range(1, M_c + 1)] + \
        [f"omega[{m}]" for m in range(1, M_c + 1)]
    ranks: dict[str, list[int]] = {t: [] for t in tracked}
    n_failed = 0
    n_diag_failed = 0
    failed_truths: list[dict] = []

    for r in range(R):
        pr = _draw_prior(rng, n_constr, K, M_c, M_b, P)
        data = _simulate_given(pr, rng, n_constr, items_per_constr, n_part,
                               ratings_per_item, K, M_c, M_b, P)
        try:
            fit, idata = fit_model(data, seed=seed + r, iter_warmup=500,
                                   iter_sampling=500, chains=4)
        except Exception:
            n_failed += 1
            continue
        # per-fit sanity check scaled to SBC's small fits: the production
        # gate's ESS>400 is unreachable with 2x300 draws by construction.
        # Ranks need R-hat convergence, few divergences, and enough effective
        # draws to support n_thin thinned ranks.
        import numpy as _np
        div = int(_np.sum(fit.method_variables()["divergent__"]))
        ndr = int(_np.prod(fit.method_variables()["divergent__"].shape))
        import arviz as _az
        summ = _az.summary(idata, var_names=[v for v in ("beta", "sigma_s",
                           "tau_constr", "tau_item", "sigma_u")
                           if v in idata.posterior])
        if not (div / ndr < 0.01 and float(summ["r_hat"].max()) < 1.02
                and float(summ["ess_bulk"].min()) > 100):
            # exclude this replication's ranks entirely: draws the check judged
            # unreliable must not enter the uniformity evidence; record where
            # in prior space the failures concentrate instead
            n_diag_failed += 1
            failed_truths.append({"tau_constr": round(float(pr["tau_constr"]), 2),
                                  "tau_item": round(float(pr["tau_item"]), 2),
                                  "sigma_u": round(float(pr["sigma_u"]), 2)})
            continue
        post = idata.posterior
        for t in tracked:
            if "[" in t:
                base, i = t.split("["); i = int(i.rstrip("]")) - 1
                draws = post[base].stack(d=("chain", "draw")).values[i]
                truth = pr[base][i]
            else:
                draws = post[t].stack(d=("chain", "draw")).values
                truth = pr[t]
            step = max(len(draws) // n_thin, 1)
            thinned = draws[::step][:n_thin]
            ranks[t].append(int(np.sum(thinned < truth)))

    report = {"R": R, "n_failed_fits": n_failed,
              "n_diag_failed": n_diag_failed, "n_thin": n_thin, "params": {}}
    B = 7  # rank bins
    passed = True
    for t, rk in ranks.items():
        if len(rk) < 10:
            report["params"][t] = {"error": "too few completed replications"}
            passed = False
            continue
        hist, _ = np.histogram(rk, bins=B, range=(0, n_thin + 1))
        expected = len(rk) / B
        chi2 = float(np.sum((hist - expected) ** 2 / expected))
        # chi-square tail via Wilson-Hilferty normal approximation (adequate
        # for a smoke alarm; avoids a scipy dependency)
        from math import erfc, sqrt
        df = B - 1
        zwh = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
        p = 0.5 * erfc(zwh / sqrt(2))
        report["params"][t] = {"chi2": round(chi2, 2), "p_uniform_approx": round(p, 4),
                               "rank_hist": hist.tolist()}
        if p < 0.005:
            passed = False
    # a pipeline whose small-data fits routinely fail or misbehave is not
    # validated by the survivors' rank uniformity
    report["failed_fit_truths"] = failed_truths
    from ..spec import load_spec
    cap = load_spec()["sbc"]["failure_frac_max"]
    if (n_failed + n_diag_failed) > cap * R:
        passed = False
        report["failure_note"] = (f"more than {cap:.0%} of replications failed "
                                  "or failed diagnostics")
    report["passed"] = passed
    from .fit import STAN_FILE, sha256_file
    report["stan_sha256"] = sha256_file(STAN_FILE)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
