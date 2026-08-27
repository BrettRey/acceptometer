"""Model-layer tests: data building invariants and a fast end-to-end recovery
smoke. The full ladder (simulate --check, sbc_run R=40, LOCO) is run
separately; these tests guard the plumbing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acceptometer.items import Item
from acceptometer.model.fit import build_stan_data


def _items():
    return [
        Item("a1", "the cat sleeps", "agr", "test"),
        Item("a2", "the cat sleep", "agr", "test"),
        Item("b1", "who did you see him", "island", "test"),
    ]


def test_build_stan_data_indexing():
    items = _items()
    X = np.array([[1.0, 2.0], [1.5, 2.5], [3.0, 1.0]])
    human = pd.DataFrame({
        "item_id": ["a1", "a2", "b1", "a1"],
        "participant_id": ["p2", "p1", "p1", "p1"],
        "rating": [6, 2, 3, 7],
    })
    cont = pd.DataFrame({
        "item_id": ["a1", "a2", "b1"],
        "cell_id": ["m/logprob_sum"] * 3,
        "value": [-10.0, -20.0, -30.0],
    })
    data, maps = build_stan_data(items, X, human, cont, None, K=7)

    assert data["N_item"] == 3 and data["N_constr"] == 2
    # constructions sorted: agr=1, island=2
    assert data["constr"] == [1, 1, 2]
    # participants sorted: p1=1, p2=2
    assert data["part_h"] == [2, 1, 1, 1]
    assert data["y"] == [6, 2, 3, 7]
    # covariates standardized
    Xs = np.array(data["X"])
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-12)
    # scores standardized per cell, constants recorded for back-mapping
    s = np.array(data["s"])
    assert abs(s.mean()) < 1e-12 and abs(s.std() - 1) < 1e-12
    assert "m/logprob_sum" in maps["cell_standardization"]
    # empty binary arm handled
    assert data["N_b"] == 0 and data["M_b"] == 0


def test_build_stan_data_no_human_arm():
    items = _items()
    X = np.zeros((3, 2))
    cont = pd.DataFrame({"item_id": ["a1"], "cell_id": ["c"], "value": [0.5]})
    data, _ = build_stan_data(items, X, None, cont, None)
    assert data["N_h"] == 0 and data["N_part"] == 1


def test_duplicate_item_ids_rejected(tmp_path):
    from acceptometer.items import save_items, load_items
    p = tmp_path / "items.jsonl"
    save_items([Item("x", "a", "c", "t"), Item("x", "b", "c", "t")], p)
    with pytest.raises(ValueError):
        load_items(p)


@pytest.mark.slow
def test_tiny_recovery_end_to_end():
    from acceptometer.model.simulate import simulate, recovery_check
    from acceptometer.model.fit import fit_model, diagnostics_gate

    data, truth = simulate(n_constr=3, items_per_constr=6, n_part=10,
                           ratings_per_item=5, M_c=2, M_b=1, seed=3)
    fit, idata = fit_model(data, seed=3, iter_warmup=400, iter_sampling=400,
                           chains=2, adapt_delta=0.95)
    diag = diagnostics_gate(fit, idata)
    rec = recovery_check(idata, truth)
    # tiny data: family deviations are weakly identified at this size, so this
    # is a plumbing smoke test with a loose divergence bound (1% of draws);
    # the production gates in diagnostics_gate stay strict
    assert diag["divergence_rate"] < 0.01
    assert rec["theta_post_mean_corr_truth"] > 0.7
