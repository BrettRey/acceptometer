"""Command-line interface: acceptometer simulate | fit | warrant | plot | elicit | validate.

Heavy imports (cmdstanpy, arviz, pandas) happen inside commands so the CLI
starts fast. `elicit` and `validate` are wired to elicit.grid and
model.loco; if those modules are absent they say "not wired in v1" and
exit 1 rather than pretending.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def main() -> None:
    """Acceptometer: a warranted Bayesian measurement instrument for LLM
    acceptability judgments. Fits emit a posterior and a validity
    certificate; claims outside the certificate are the user's own risk."""


def _sim_maps(data: dict) -> dict:
    """Index maps for a simulated dataset, mirroring what
    fit.build_stan_data records for real data, so warrant, plots, and
    design work on simulation runs."""
    item_ids = [f"sim-item-{i:03d}" for i in range(1, data["N_item"] + 1)]
    constructions = [f"sim-constr-{c:02d}" for c in range(1, data["N_constr"] + 1)]
    return {
        "item_ids": item_ids,
        "constructions": constructions,
        "item_constructions": {
            item_ids[i]: constructions[data["constr"][i] - 1]
            for i in range(data["N_item"])
        },
        "X_mean": [0.0] * data["P"],
        "X_sd": [1.0] * data["P"],
        "cont_cells": [f"sim-cont-{m:02d}" for m in range(1, data["M_c"] + 1)],
        "bin_cells": [f"sim-bin-{m:02d}" for m in range(1, data["M_b"] + 1)],
        "participants": [f"sim-part-{p:03d}" for p in range(1, data["N_part"] + 1)],
        "cell_standardization": {},
    }


def _write_sim_inputs(data: dict, maps: dict, out_dir: Path) -> None:
    """Write cont.csv and human_item_means.csv for a simulation run, the
    same input surfaces the fit command writes for real data, so the
    warrant multiverse statistic and the item scatter plot have evidence
    files to read."""
    import pandas as pd

    if data["N_c"]:
        pd.DataFrame({
            "item_id": [maps["item_ids"][i - 1] for i in data["item_c"]],
            "cell_id": [maps["cont_cells"][m - 1] for m in data["cell_c"]],
            "value": data["s"],
        }).to_csv(out_dir / "cont.csv", index=False)
    if data["N_h"]:
        hdf = pd.DataFrame({
            "item_id": [maps["item_ids"][i - 1] for i in data["item_h"]],
            "rating": data["y"],
        })
        hm = hdf.groupby("item_id")["rating"].mean().rename("mean_rating")
        hm.reset_index().to_csv(out_dir / "human_item_means.csv", index=False)


@main.command()
@click.option("--check", is_flag=True,
              help="fit the simulation and gate on diagnostics and recovery")
@click.option("--seed", type=int, default=20260827, show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path),
              default=Path("runs/sim-check"), show_default=True)
def simulate(check: bool, seed: int, out_dir: Path) -> None:
    """Simulate the full generative process; with --check, run the
    fake-data recovery gate (untrusted pipeline until it passes)."""
    from .model import simulate as simmod

    data, truth = simmod.simulate(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not check:
        (out_dir / "sim-data.json").write_text(json.dumps(data))
        click.echo(f"wrote simulated Stan data to {out_dir / 'sim-data.json'} "
                   f"(no fit; use --check to gate)")
        return

    from .model.fit import diagnostics_gate, fit_model, save_fit

    stanfit, idata = fit_model(data, seed=seed)
    diag = diagnostics_gate(stanfit, idata)
    rec = simmod.recovery_check(idata, truth)
    maps = _sim_maps(data)
    save_fit(idata, maps, diag, out_dir)
    simmod.write_report(rec, out_dir / "recovery.json")
    _write_sim_inputs(data, maps, out_dir)

    click.echo("diagnostics gate:")
    click.echo(json.dumps(diag, indent=2))
    click.echo("recovery check:")
    click.echo(json.dumps(rec, indent=2))
    if not (diag["passed"] and rec["passed"]):
        click.echo("FAILED: the pipeline did not pass the fake-data ladder; "
                   "nothing downstream may trust this fit.", err=True)
        raise SystemExit(1)
    click.echo(f"passed both gates; artifacts in {out_dir}")


def _load_fit_inputs(items_path: Path, human_path: Path | None,
                     meas_path: Path):
    """Load items, nuisance covariates, human ratings, and measurements
    split by kind: the shared input surface of fit and validate."""
    import numpy as np
    import pandas as pd

    from .elicit.base import BINARY, CONTINUOUS, load_measurements
    from .items import load_items, nuisance_covariates

    items = load_items(items_path)
    X = np.asarray(nuisance_covariates(items), dtype=float)
    human = pd.read_csv(human_path) if human_path else None

    ms = load_measurements(meas_path)
    mdf = pd.DataFrame([{"item_id": m.item_id, "cell_id": m.cell_id,
                         "kind": m.kind, "value": m.value} for m in ms])
    unknown = sorted(set(mdf["kind"]) - {CONTINUOUS, BINARY}) if len(mdf) else []
    if unknown:
        raise click.ClickException(f"unknown measurement kinds: {unknown}")

    def _kind(kind: str) -> pd.DataFrame | None:
        sub = mdf[mdf["kind"] == kind][["item_id", "cell_id", "value"]]
        return sub.reset_index(drop=True) if len(sub) else None

    return items, X, human, _kind(CONTINUOUS), _kind(BINARY)


@main.command()
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option("--items", "items_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="items JSONL (items.py schema)")
@click.option("--human", "human_path", default=None,
              type=click.Path(exists=True, path_type=Path),
              help="CSV with columns item_id,participant_id,rating")
@click.option("--measurements", "meas_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="measurements JSONL (elicit.base schema)")
@click.option("--k", type=int, default=7, show_default=True,
              help="points on the human rating scale")
@click.option("--seed", type=int, default=1, show_default=True)
def fit(run_dir: Path, items_path: Path, human_path: Path | None,
        meas_path: Path, k: int, seed: int) -> None:
    """Fit the joint measurement model to real data. Refuses (exit 1) if
    the diagnostics gate fails; artifacts are still saved for inspection,
    but a failed fit licenses nothing."""
    from .model.fit import build_stan_data, diagnostics_gate, fit_model, save_fit

    items, X, human, cont, binary = _load_fit_inputs(
        items_path, human_path, meas_path)
    data, maps = build_stan_data(items, X, human, cont, binary, K=k)
    maps["item_constructions"] = {it.item_id: it.construction for it in items}

    stanfit, idata = fit_model(data, seed=seed)
    diag = diagnostics_gate(stanfit, idata)
    save_fit(idata, maps, diag, run_dir)
    run_dir = Path(run_dir)
    if cont is not None:
        cont.to_csv(run_dir / "cont.csv", index=False)
    if binary is not None:
        binary.to_csv(run_dir / "bin.csv", index=False)
    if human is not None and len(human):
        hm = human.groupby("item_id")["rating"].mean().rename("mean_rating")
        hm.reset_index().to_csv(run_dir / "human_item_means.csv", index=False)

    click.echo(json.dumps(diag, indent=2))
    if not diag["passed"]:
        click.echo("REFUSED: diagnostics gate failed (see report above); "
                   "artifacts saved for inspection only.", err=True)
        raise SystemExit(1)
    click.echo(f"fit passed the diagnostics gate; artifacts in {run_dir}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--estimand", "estimand_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="YAML mapping stating population, response scale, context")
def warrant(run_dir: Path, estimand_path: Path) -> None:
    """Build the validity certificate (warrant.yaml) from the run's
    evidence files. Missing evidence refuses tiers, never grants them."""
    import yaml

    from .warrant import build_warrant

    est = yaml.safe_load(estimand_path.read_text())
    if not isinstance(est, dict):
        raise click.ClickException("the estimand YAML must be a mapping")
    cert = build_warrant(run_dir, est)
    click.echo(f"wrote {Path(run_dir) / 'warrant.yaml'}")
    for tier, reason in cert["licensed_claims"].items():
        click.echo(f"granted {tier}: {reason}")
    for tier, reason in cert["refused_claims"].items():
        click.echo(f"refused {tier}: {reason}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def plot(run_dir: Path) -> None:
    """Render whichever plots the run dir supports into run_dir/plots/."""
    import arviz as az
    import pandas as pd

    from . import plots

    run_dir = Path(run_dir)
    idata = az.from_netcdf(str(run_dir / "posterior.nc"))
    maps = json.loads((run_dir / "index_maps.json").read_text())
    pdir = run_dir / "plots"
    made = [plots.secret_weapon(idata, maps, pdir / "secret_weapon.png")]
    if maps.get("cont_cells") and "reliability" in idata.posterior:
        made.append(plots.reliability_forest(
            idata, maps, pdir / "reliability_forest.png"))

    hm_path = run_dir / "human_item_means.csv"
    if hm_path.exists():
        hm = pd.read_csv(hm_path)
        means = dict(zip(hm["item_id"], hm["mean_rating"]))
        made.append(plots.item_scatter(
            idata, maps, means, pdir / "item_scatter.png"))

    cont_path = run_dir / "cont.csv"
    if cont_path.exists():
        cont = pd.read_csv(cont_path)
        cont = cont.groupby(["item_id", "cell_id"], as_index=False)["value"].mean()

        def _z(v: pd.Series) -> pd.Series:
            sd = v.std(ddof=0)
            return (v - v.mean()) / (sd if sd > 0 else 1.0)

        cont["value"] = cont.groupby("cell_id")["value"].transform(_z)
        made.append(plots.multiverse_fan(cont, pdir / "multiverse_fan.png"))

    for p in made:
        click.echo(str(p))


@main.command()
@click.option("--items", "items_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="items JSONL (items.py schema)")
@click.option("--out", "out_dir", required=True,
              type=click.Path(path_type=Path),
              help="output dir for measurements.jsonl and grid_manifest.yaml")
@click.option("--hf", "hf_models", multiple=True,
              help="HF causal-LM id for logprob cells (repeatable); "
                   "needs the 'hf' optional dependencies")
@click.option("--ollama", "ollama_models", multiple=True,
              help="Ollama model for prompted cells (repeatable)")
@click.option("--prompts", "prompts_path", default="prompts/prompts.yaml",
              show_default=True, help="registered prompt paraphrases (YAML)")
@click.option("--host", default="http://localhost:11434", show_default=True,
              help="Ollama host")
@click.option("--repeats", type=int, default=5, show_default=True,
              help="repeated draws per prompted cell")
def elicit(items_path: Path, out_dir: Path, hf_models: tuple[str, ...],
           ollama_models: tuple[str, ...], prompts_path: str, host: str,
           repeats: int) -> None:
    """Run the elicitation multiverse grid over local instruments.
    Measurements are cached in OUT/measurements.jsonl; reruns score only
    what is missing."""
    try:
        from .elicit.grid import run_grid
    except ModuleNotFoundError:
        click.echo("elicit: not wired in v1 (acceptometer.elicit.grid is "
                   "not present)")
        raise SystemExit(1)
    from .items import load_items

    if not hf_models and not ollama_models:
        raise click.ClickException(
            "no instruments given: pass at least one --hf or --ollama model")
    instruments = []
    for model_id in hf_models:
        from .elicit.hf_logprob import HFLogprobScorer
        try:
            instruments.append(HFLogprobScorer(model_id=model_id))
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
    for model in ollama_models:
        from .elicit.ollama_chat import OllamaChatJudge
        instruments.append(OllamaChatJudge(model=model, host=host,
                                           prompts_path=prompts_path))

    items = load_items(items_path)
    run_grid(instruments, items, out_dir, repeats_prompted=repeats)
    click.echo(f"grid complete; measurements and manifest in {out_dir}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--items", "items_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="items JSONL (items.py schema)")
@click.option("--human", "human_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="CSV with columns item_id,participant_id,rating")
@click.option("--measurements", "meas_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="measurements JSONL (elicit.base schema)")
@click.option("--k", type=int, default=7, show_default=True,
              help="points on the human rating scale")
@click.option("--families", default=None,
              help="comma-separated subset of families to hold out "
                   "(default: all)")
@click.option("--seed", type=int, default=11, show_default=True)
def validate(run_dir: Path, items_path: Path, human_path: Path,
             meas_path: Path, k: int, families: str | None,
             seed: int) -> None:
    """Run the generalization tests: leave-one-construction-out transfer.
    Writes RUN_DIR/loco.json, the evidence the warrant's ranking and
    aggregate-estimation tiers read. One refit per family, so this is
    slow in proportion to the number of families."""
    try:
        from .model.loco import loco
    except ModuleNotFoundError:
        click.echo("validate: not wired in v1 (acceptometer.model.loco is "
                   "not present)")
        raise SystemExit(1)

    items, X, human, cont, binary = _load_fit_inputs(
        items_path, human_path, meas_path)
    fam_list = [f.strip() for f in families.split(",")] if families else None
    report = loco(items, X, human, cont, binary, K=k, families=fam_list,
                  out_path=Path(run_dir) / "loco.json", seed=seed)
    summary = {k_: report[k_] for k_ in
               ("n_families", "mean_rmse", "mean_spearman", "mean_coverage90",
                "all_diagnostics_passed")}
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"wrote {Path(run_dir) / 'loco.json'}")


if __name__ == "__main__":
    main()
