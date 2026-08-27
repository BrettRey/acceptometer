"""CmdStanPy wrapper: data building, sampling, and the diagnostics gate.

Nothing downstream (warrant, plots, design) accepts a fit that fails the gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

STAN_FILE = Path(__file__).parent / "acceptometer.stan"


def build_stan_data(
    items: list,                      # list[acceptometer.items.Item]
    X: np.ndarray,                    # (N, P) nuisance covariates, raw scale
    human: pd.DataFrame | None,       # columns: item_id, participant_id, rating (1..K)
    cont: pd.DataFrame | None,        # columns: item_id, cell_id, value
    binary: pd.DataFrame | None,      # columns: item_id, cell_id, value (0/1)
    K: int = 7,
    standardize_scores: bool = True,
    new_families: set | None = None,
    X_stats: tuple | None = None,
) -> tuple[dict, dict]:
    """Returns (stan_data, index_maps). index_maps records the id->index
    mappings and per-cell standardization constants so posteriors can be
    mapped back to names and raw scales."""
    item_ids = [it.item_id for it in items]
    item_ix = {iid: i + 1 for i, iid in enumerate(item_ids)}
    constrs = sorted({it.construction for it in items})
    constr_ix = {c: i + 1 for i, c in enumerate(constrs)}

    X = np.asarray(X, dtype=float)
    if X_stats is not None:
        X_mean, X_sd = (np.asarray(X_stats[0], dtype=float),
                        np.asarray(X_stats[1], dtype=float))
    else:
        X_mean, X_sd = X.mean(axis=0), X.std(axis=0)
    X_sd = np.where(X_sd == 0, 1.0, X_sd)
    Xs = (X - X_mean) / X_sd

    new_families = new_families or set()
    data: dict = dict(
        prior_only=0,
        N_item=len(items),
        N_constr=len(constrs),
        constr=[constr_ix[it.construction] for it in items],
        is_new=[int(c in new_families) for c in constrs],
        P=Xs.shape[1],
        X=Xs.tolist(),
    )
    maps: dict = dict(
        item_ids=item_ids, constructions=constrs,
        X_mean=X_mean.tolist(), X_sd=X_sd.tolist(), X_standardized=Xs.tolist(),
        cont_cells=[], bin_cells=[], participants=[], cell_standardization={},
    )

    if human is not None and len(human):
        parts = sorted(human["participant_id"].unique())
        part_ix = {p: i + 1 for i, p in enumerate(parts)}
        maps["participants"] = list(parts)
        data.update(
            N_h=len(human), K=K,
            item_h=[item_ix[i] for i in human["item_id"]],
            N_part=len(parts),
            part_h=[part_ix[p] for p in human["participant_id"]],
            y=human["rating"].astype(int).tolist(),
        )
    else:
        data.update(N_h=0, K=K, item_h=[], N_part=1, part_h=[], y=[])

    if cont is not None and len(cont):
        cells = sorted(cont["cell_id"].unique())
        cix = {c: i + 1 for i, c in enumerate(cells)}
        maps["cont_cells"] = cells
        vals = cont["value"].astype(float).to_numpy().copy()
        if standardize_scores:
            for c in cells:
                m = (cont["cell_id"] == c).to_numpy()
                mu, sd = vals[m].mean(), vals[m].std() or 1.0
                vals[m] = (vals[m] - mu) / sd
                maps["cell_standardization"][c] = {"mean": float(mu), "sd": float(sd)}
        reps = cont.groupby(["cell_id", "item_id"]).size()
        has_reps = [int(reps.xs(c, level="cell_id").max() > 1) for c in cells]
        data.update(
            N_c=len(cont), M_c=len(cells),
            item_c=[item_ix[i] for i in cont["item_id"]],
            cell_c=[cix[c] for c in cont["cell_id"]],
            s=vals.tolist(),
            has_reps=has_reps,
        )
    else:
        data.update(N_c=0, M_c=0, item_c=[], cell_c=[], s=[], has_reps=[0])

    if binary is not None and len(binary):
        cells = sorted(binary["cell_id"].unique())
        cix = {c: i + 1 for i, c in enumerate(cells)}
        maps["bin_cells"] = cells
        data.update(
            N_b=len(binary), M_b=len(cells),
            item_b=[item_ix[i] for i in binary["item_id"]],
            cell_b=[cix[c] for c in binary["cell_id"]],
            z=binary["value"].astype(int).tolist(),
        )
    else:
        data.update(N_b=0, M_b=0, item_b=[], cell_b=[], z=[])

    return data, maps


def _default_inits(data: dict) -> dict:
    """Start chains in the identified basin. The joint posterior has a
    spurious reflection mode (instrument slopes and latent orientation jointly
    flipped) that is locally stable: escaping it requires every theta to cross
    the likelihood valley at once, which HMC will not do. Random inits land
    ~1-in-4 chains there. So initialize the latent items at the standardized
    observed human item means (data-informed inits select a basin; they do not
    bias the posterior), with modest positive slopes. A genuinely
    anti-correlated instrument can still walk to negative beta."""
    inits = {
        "tau_constr": 0.5, "tau_item": 1.0, "sigma_u": 0.5,
        "kappa": [float(k) for k in np.linspace(-2, 2, data["K"] - 1)],
    }
    if data.get("N_h", 0) > 0:
        item_h = np.asarray(data["item_h"])
        y = np.asarray(data["y"], dtype=float)
        means = np.full(data["N_item"], y.mean())
        for i in range(1, data["N_item"] + 1):
            m = item_h == i
            if m.any():
                means[i - 1] = y[m].mean()
        sd = means.std() or 1.0
        inits["z_item"] = ((means - means.mean()) / sd).tolist()
    if data.get("M_c", 0) > 0:
        inits["beta"] = [0.3] * data["M_c"]
        inits["sigma_s"] = [0.8] * data["M_c"]
        inits["omega"] = [0.3] * data["M_c"]
    if data.get("M_b", 0) > 0:
        inits["b_b"] = [0.5] * data["M_b"]
    return inits


def fit_model(data: dict, out_dir: str | Path | None = None, seed: int = 1,
              iter_warmup: int = 1000, iter_sampling: int = 1000,
              adapt_delta: float = 0.95, chains: int = 4, inits: dict | None = None):
    """Compile (cached), sample, and return (CmdStanMCMC, arviz.InferenceData)."""
    import arviz as az
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(
        data=data, chains=chains, parallel_chains=min(chains, 4),
        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        adapt_delta=adapt_delta, seed=seed, show_progress=False,
        inits=inits if inits is not None else _default_inits(data),
        output_dir=str(out_dir) if out_dir else None,
    )
    idata = az.from_cmdstanpy(fit)
    return fit, idata


def diagnostics_gate(fit, idata) -> dict:
    """Hard gate: divergences < 0.5%, max R-hat < 1.01, min bulk ESS > 400
    on core parameters. Returns report dict with `passed`."""
    import arviz as az

    div = int(np.sum(fit.method_variables()["divergent__"]))
    n_draws = int(np.prod(fit.method_variables()["divergent__"].shape))
    core = [v for v in ["beta", "sigma_s", "tau_constr", "tau_item", "tau_a",
                        "tau_b", "omega", "kappa", "sigma_u", "b_b",
                        "a_dev", "b_dev", "theta"]
            if v in idata.posterior]
    if "mu_new_raw" in idata.posterior and idata.posterior["mu_new_raw"].shape[-1] > 0:
        core.append("mu_new_raw")
    summ = az.summary(idata, var_names=core)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    report = {
        "divergences": div,
        "divergence_rate": round(div / n_draws, 4),
        "rhat_max": round(rhat_max, 4),
        "ess_bulk_min": int(ess_min),
        "passed": bool(div / n_draws < 0.005 and rhat_max < 1.01 and ess_min > 400),
    }
    return report


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_fit(idata, maps: dict, report: dict, out_dir: str | Path,
             data: dict | None = None, input_hashes: dict | None = None) -> None:
    """Persist the fit plus a run manifest (run.json) binding the artifacts:
    downstream evidence writers stamp posterior_sha256 and the warrant refuses
    evidence whose stamp does not match the posterior it certifies."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(out / "posterior.nc"))
    (out / "index_maps.json").write_text(json.dumps(maps, indent=2))
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        commit = "unknown"
    n_crit = 0
    if data is not None and data.get("N_h", 0) > 0:
        n_crit = len(set(data["item_h"]))
    run = {
        "run_id": str(uuid.uuid4()),
        "posterior_sha256": sha256_file(out / "posterior.nc"),
        "stan_sha256": sha256_file(STAN_FILE),
        "code_commit": commit,
        "n_items_criterion": n_crit,
        "n_items_total": len(maps.get("item_ids", [])),
        "input_hashes": input_hashes or {},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))


def mode_audit(data: dict, seed: int = 3, iter_warmup: int = 750,
               iter_sampling: int = 750) -> dict:
    """Deliberately overdispersed-start check for the known reflection basin.

    Data-informed inits place all production chains in one basin; agreement
    among them cannot show the other basin has negligible mass. This audit
    runs extra chains from random inits and compares the best log posterior
    density found in each orientation (sign of the first instrument slope).
    A reflected basin within ~10 lp of the main one is a hard warning: the
    posterior is genuinely bimodal and single-basin results are truncated."""
    import arviz as az
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(data=data, chains=4, parallel_chains=4,
                       iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                       adapt_delta=0.95, seed=seed, show_progress=False)
    idata = az.from_cmdstanpy(fit)
    lp = fit.method_variables()["lp__"]                    # (draws, chains)
    beta0 = idata.posterior["beta"].values[..., 0]          # (chains, draws)
    lp_by_chain = lp.T                                      # (chains, draws)
    pos_mask = beta0.mean(axis=1) > 0
    out = {
        "chains_positive_orientation": int(pos_mask.sum()),
        "chains_negative_orientation": int((~pos_mask).sum()),
        "max_lp_positive": (round(float(lp_by_chain[pos_mask].max()), 1)
                            if pos_mask.any() else None),
        "max_lp_negative": (round(float(lp_by_chain[~pos_mask].max()), 1)
                            if (~pos_mask).any() else None),
    }
    if out["max_lp_positive"] is not None and out["max_lp_negative"] is not None:
        gap = out["max_lp_positive"] - out["max_lp_negative"]
        out["lp_gap_positive_minus_negative"] = round(float(gap), 1)
        out["bimodality_warning"] = bool(abs(gap) < 10)
    else:
        out["lp_gap_positive_minus_negative"] = None
        out["bimodality_warning"] = False
    return out
