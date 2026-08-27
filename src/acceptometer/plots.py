"""Diagnostic and reporting plots. Matplotlib only, no seaborn.

Every function draws one figure, saves it as a PNG at the given path
(parent directories created), and returns the path. Figures are built on
`matplotlib.figure.Figure` directly, so no global pyplot state and no
display backend is touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_BLUE = "#4477aa"
_LIGHTBLUE = "#99bbdd"
_GREY = "#999999"


def _flat(idata, var: str) -> np.ndarray:
    """Posterior draws of a vector parameter as (n_draws, dim)."""
    v = np.asarray(idata.posterior[var].values)
    return v.reshape(-1, v.shape[-1])


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", format="png")
    return path


def secret_weapon(idata, maps: dict, path: str | Path) -> Path:
    """Gelman's "secret weapon": forest plot of construction-family means
    mu_c with 50% (thick) and 90% (thin) posterior intervals, sorted by
    posterior mean. Family names come from maps["constructions"]."""
    draws = _flat(idata, "mu_c")
    names = list(maps["constructions"])
    mean = draws.mean(axis=0)
    q5, q25, q75, q95 = np.percentile(draws, [5, 25, 75, 95], axis=0)
    order = np.argsort(mean)
    y = np.arange(len(order))

    fig = Figure(figsize=(6.0, max(2.0, 0.4 * len(names) + 1.2)))
    ax = fig.subplots()
    ax.hlines(y, q5[order], q95[order], color=_BLUE, lw=1.0)
    ax.hlines(y, q25[order], q75[order], color=_BLUE, lw=3.0)
    ax.plot(mean[order], y, "o", color="#222222", ms=4)
    ax.axvline(0.0, color=_GREY, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("construction-family mean acceptability, mu_c (human-logit units)")
    return _save(fig, path)


def reliability_forest(idata, maps: dict, path: str | Path) -> Path:
    """Forest plot of posterior instrument reliability per continuous
    elicitation cell, with 50% and 90% intervals, sorted by posterior
    median. Cell names come from maps["cont_cells"]."""
    draws = _flat(idata, "reliability")
    names = list(maps["cont_cells"])
    med = np.median(draws, axis=0)
    q5, q25, q75, q95 = np.percentile(draws, [5, 25, 75, 95], axis=0)
    order = np.argsort(med)
    y = np.arange(len(order))

    fig = Figure(figsize=(6.0, max(2.0, 0.4 * len(names) + 1.2)))
    ax = fig.subplots()
    ax.hlines(y, q5[order], q95[order], color=_BLUE, lw=1.0)
    ax.hlines(y, q25[order], q75[order], color=_BLUE, lw=3.0)
    ax.plot(med[order], y, "o", color="#222222", ms=4)
    ax.axvline(0.5, color=_GREY, lw=0.8, ls=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("instrument reliability (share of score variance carried by theta)")
    return _save(fig, path)


def item_scatter(idata, maps: dict, human_item_means, path: str | Path) -> Path:
    """Posterior mean theta with 90% interval (vertical error bars) against
    the observed human item mean rating. The two axes are on different
    scales, so no identity line; the Pearson r of posterior means against
    human means is annotated in the corner.

    `human_item_means` is a mapping or pandas Series keyed by item_id, or
    an array aligned with maps["item_ids"]. Items without a human mean are
    dropped."""
    ids = list(maps["item_ids"])
    if isinstance(human_item_means, pd.Series):
        lookup = {str(k): float(v) for k, v in human_item_means.items()}
    elif isinstance(human_item_means, Mapping):
        lookup = {str(k): float(v) for k, v in human_item_means.items()}
    else:
        arr = np.asarray(human_item_means, dtype=float)
        if len(arr) != len(ids):
            raise ValueError(
                f"array of human means has length {len(arr)}, expected {len(ids)}")
        lookup = dict(zip(ids, arr))
    idx = [i for i, iid in enumerate(ids)
           if iid in lookup and np.isfinite(lookup[iid])]
    if len(idx) < 2:
        raise ValueError("item_scatter needs human means for at least 2 items")

    draws = _flat(idata, "theta")
    x = np.array([lookup[ids[i]] for i in idx])
    m = draws.mean(axis=0)[idx]
    lo = np.percentile(draws, 5, axis=0)[idx]
    hi = np.percentile(draws, 95, axis=0)[idx]
    r = float(np.corrcoef(m, x)[0, 1])

    fig = Figure(figsize=(5.0, 4.4))
    ax = fig.subplots()
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5,
                lw=0.8, color=_BLUE, ecolor=_LIGHTBLUE)
    ax.annotate(f"r = {r:.2f}", xy=(0.04, 0.96), xycoords="axes fraction",
                fontsize=9, va="top")
    ax.set_xlabel("observed human item mean rating")
    ax.set_ylabel("posterior theta, mean and 90% interval (human-logit units)")
    return _save(fig, path)


def multiverse_fan(cont_df: pd.DataFrame, path: str | Path) -> Path:
    """Per-item spread of the elicitation multiverse. `cont_df` is tidy
    (item_id, cell_id, value) with per-cell standardized scores; repeats
    are averaged per (item, cell). Items on x sorted by cross-cell mean,
    one line per cell, shaded band for the cross-cell range. A tight fan
    means the multiverse agrees; a wide fan means the elicitation is the
    result."""
    pivot = cont_df.pivot_table(index="item_id", columns="cell_id",
                                values="value", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    x = np.arange(len(pivot))

    fig = Figure(figsize=(max(5.0, 0.16 * len(pivot) + 2.0), 4.0))
    ax = fig.subplots()
    ax.fill_between(x, pivot.min(axis=1), pivot.max(axis=1),
                    color="#cccccc", alpha=0.5, lw=0, label="cross-cell range")
    for cell in pivot.columns:
        ax.plot(x, pivot[cell], marker=".", ms=3, lw=0.7, alpha=0.85,
                label=str(cell))
    ax.set_xlabel("items, sorted by cross-cell mean")
    ax.set_ylabel("standardized score per elicitation cell")
    if len(pivot.columns) <= 8:
        ax.legend(fontsize=7, frameon=False)
    if len(pivot) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=90, fontsize=6)
    return _save(fig, path)


def calibration_plot(pred_mean, pred_lo, pred_hi, observed,
                     path: str | Path) -> Path:
    """Predicted against observed with intervals and a 45-degree line.
    The empirical coverage of the supplied intervals (fraction of observed
    values inside [pred_lo, pred_hi]) is annotated in the corner. All four
    arrays must be aligned and on the same scale."""
    pm = np.asarray(pred_mean, dtype=float)
    lo = np.asarray(pred_lo, dtype=float)
    hi = np.asarray(pred_hi, dtype=float)
    obs = np.asarray(observed, dtype=float)
    cover = float(np.mean((obs >= lo) & (obs <= hi)))

    fig = Figure(figsize=(4.6, 4.4))
    ax = fig.subplots()
    ax.errorbar(obs, pm, yerr=[pm - lo, hi - pm], fmt="o", ms=3.5,
                lw=0.8, color=_BLUE, ecolor=_LIGHTBLUE)
    span = [min(obs.min(), lo.min()), max(obs.max(), hi.max())]
    ax.plot(span, span, color=_GREY, lw=0.8, ls="--")
    ax.annotate(f"interval coverage = {cover:.2f} (n = {len(obs)})",
                xy=(0.04, 0.96), xycoords="axes fraction", fontsize=9, va="top")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted, mean and interval")
    return _save(fig, path)
