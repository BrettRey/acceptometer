"""Human-budget allocator.

Human judgments are the scarce, expensive criterion; the instrument is
cheap. This module spends the human budget where the instrument is most
uncertain: items are ranked by the posterior sd of theta, optionally
weighted per construction family, and the top `budget` items are the ones
to send to participants next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def allocate(idata, maps: dict, budget: int,
             weights: dict | None = None) -> pd.DataFrame:
    """Rank items by posterior sd of theta (descending) and return the top
    `budget` rows with columns item_id, construction, post_sd, priority.

    `weights` optionally maps construction family to a multiplier;
    priority = post_sd * weight, and the ranking is by priority. Items
    whose construction is not in `weights` get weight 1.0. Constructions
    come from maps["item_constructions"] (dict item_id -> family, or list
    aligned with maps["item_ids"]); absent that, "unknown"."""
    theta = np.asarray(idata.posterior["theta"].values)
    post_sd = theta.reshape(-1, theta.shape[-1]).std(axis=0, ddof=1)
    ids = list(maps["item_ids"])
    if len(post_sd) != len(ids):
        raise ValueError(
            f"theta has {len(post_sd)} items, index_maps has {len(ids)}")

    ic = maps.get("item_constructions")
    if isinstance(ic, dict):
        constr = [ic.get(iid, "unknown") for iid in ids]
    elif isinstance(ic, list) and len(ic) == len(ids):
        constr = [str(c) for c in ic]
    else:
        constr = ["unknown"] * len(ids)

    weights = weights or {}
    priority = [sd * float(weights.get(c, 1.0))
                for sd, c in zip(post_sd, constr)]
    df = pd.DataFrame({
        "item_id": ids,
        "construction": constr,
        "post_sd": np.round(post_sd, 4),
        "priority": np.round(priority, 4),
    })
    df = df.sort_values("priority", ascending=False, kind="mergesort")
    return df.head(budget).reset_index(drop=True)
