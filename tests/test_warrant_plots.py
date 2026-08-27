"""End-to-end tests for warrant.py and plots.py on a real (tiny) fit.

The fixture actually runs the Stan model on a minimal simulation, so these
tests exercise the same artifact surfaces (posterior.nc, index_maps.json,
diagnostics.json) the real pipeline produces. The tiny fit takes seconds;
its diagnostics gate may legitimately fail, which the warrant must handle
by refusing, not by crashing."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from acceptometer import plots
from acceptometer.cli import _sim_maps, _write_sim_inputs
from acceptometer.model import simulate as simmod
from acceptometer.model.fit import diagnostics_gate, fit_model, save_fit
from acceptometer.warrant import TIERS, build_warrant


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim-run")
    data, truth = simmod.simulate(n_constr=3, items_per_constr=4, n_part=8,
                                  ratings_per_item=4, M_c=2, M_b=1)
    stanfit, idata = fit_model(data, iter_warmup=200, iter_sampling=200,
                               chains=2)
    diag = diagnostics_gate(stanfit, idata)
    maps = _sim_maps(data)
    save_fit(idata, maps, diag, out)
    _write_sim_inputs(data, maps, out)
    return {"dir": out, "idata": idata, "maps": maps,
            "data": data, "truth": truth}


def test_warrant_certificate(run):
    cert = build_warrant(run["dir"], {"population": "simulated",
                                      "response_scale": "7-point ordinal"})
    path = run["dir"] / "warrant.yaml"
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())

    assert "mechanism_claims" in loaded["refused_claims"]
    assert "individual_simulation" in loaded["refused_claims"]

    all_tiers = {**loaded["licensed_claims"], **loaded["refused_claims"]}
    assert set(all_tiers) == set(TIERS)
    for tier, reason in all_tiers.items():
        assert isinstance(reason, str) and reason.strip(), tier

    # no loco.json in this run dir, so ranking must not have been granted
    assert "ranking" in loaded["refused_claims"]
    assert cert["licensed_claims"] == loaded["licensed_claims"]

    # every instrument row carries numbers read from the posterior
    for inst in loaded["instruments"]:
        assert 0.0 <= inst["reliability_median"] <= 1.0
        assert len(inst["reliability_90"]) == 2


def _assert_png(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_secret_weapon(run, tmp_path):
    _assert_png(plots.secret_weapon(run["idata"], run["maps"],
                                    tmp_path / "secret_weapon.png"))


def test_reliability_forest(run, tmp_path):
    _assert_png(plots.reliability_forest(run["idata"], run["maps"],
                                         tmp_path / "reliability_forest.png"))


def test_item_scatter(run, tmp_path):
    hm = pd.read_csv(run["dir"] / "human_item_means.csv")
    means = dict(zip(hm["item_id"], hm["mean_rating"]))
    _assert_png(plots.item_scatter(run["idata"], run["maps"], means,
                                   tmp_path / "item_scatter.png"))


def test_multiverse_fan(run, tmp_path):
    cont = pd.read_csv(run["dir"] / "cont.csv")
    _assert_png(plots.multiverse_fan(cont, tmp_path / "multiverse_fan.png"))


def test_calibration_plot(run, tmp_path):
    import numpy as np

    th = run["idata"].posterior["theta"].values
    th = th.reshape(-1, th.shape[-1])
    lo, hi = np.percentile(th, [5, 95], axis=0)
    _assert_png(plots.calibration_plot(th.mean(axis=0), lo, hi,
                                       run["truth"].theta,
                                       tmp_path / "calibration.png"))
