"""Small validation-split grid over the multi-task loss weights (and optionally
the recommendation loss / conflict weight) for the primary AIPA model.

The test split is never used.  Results are written to
``outputs/results/validation_sweep_<tag>.csv``; the chosen values must then be
copied by hand into ``configs/default.yaml`` (with a comment naming the metric).

Examples::

    python scripts/tune_lambdas.py --run-mode quick
    python scripts/tune_lambdas.py --run-mode quick --lambda-rel 0.25 0.5 1.0 --lambda-act 0.1 0.3 0.6
    python scripts/tune_lambdas.py --run-mode quick --rec-loss softmax sampled_softmax bpr --tag rec_loss
    python scripts/tune_lambdas.py --run-mode quick --conflict-weight 1 2 3 --tag conflict
    python scripts/tune_lambdas.py --run-mode quick --axis learned_action_weights=false,true --axis residual_gate=false,true --tag arb
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aipa.config import load_config  # noqa: E402
from aipa.experiments import prepare, validation_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-mode", default=None, choices=["quick", "full"])
    ap.add_argument("--lambda-rel", type=float, nargs="*", default=None)
    ap.add_argument("--lambda-act", type=float, nargs="*", default=None)
    ap.add_argument("--rec-loss", nargs="*", default=None, choices=["softmax", "sampled_softmax", "bpr"])
    ap.add_argument("--conflict-weight", type=float, nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="fixed config overrides applied to every grid point (YAML-typed)")
    ap.add_argument("--axis", action="append", default=[], metavar="KEY=V1,V2,...",
                    help="extra grid axis over any config key (YAML-typed values); repeatable")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: first configured seed")
    ap.add_argument("--select-by", default="valid_Hit@10", help="column used to print the best row")
    ap.add_argument("--tag", default="lambdas")
    a = ap.parse_args()

    import yaml

    cfg = load_config(a.run_mode)
    fixed = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg.values.update(fixed)
    extra = {k: [yaml.safe_load(x) for x in vs.split(",")] for k, vs in (s.split("=", 1) for s in a.axis)}
    lambdas_only = not (a.rec_loss or a.conflict_weight or extra)
    axes = {
        "lambda_rel": a.lambda_rel or ([0.25, 0.5, 1.0] if lambdas_only else [cfg.lambda_rel]),
        "lambda_act": a.lambda_act or ([0.1, 0.3, 0.6] if lambdas_only else [cfg.lambda_act]),
        **extra,
    }
    if a.rec_loss:
        axes["rec_loss"] = a.rec_loss
    if a.conflict_weight:
        axes["conflict_loss_weight"] = a.conflict_weight
    grid = [dict(zip(axes, vals)) for vals in itertools.product(*axes.values())]
    print(f"{len(grid)} grid points x {len(a.seeds or cfg.seeds[:1])} seed(s); fixed overrides: {fixed}")
    prepared = prepare(cfg)
    df = validation_sweep(cfg, grid, prepared=prepared, seeds=a.seeds)
    out = cfg.path("output_path") / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"validation_sweep_{a.tag}.csv"
    df.to_csv(path, index=False)
    mean = df[df.seed.astype(str) == "mean"] if (df.seed.astype(str) == "mean").any() else df
    best = mean.sort_values(a.select_by, ascending=False).iloc[0]
    print(mean.to_string(index=False))
    print(f"\nbest by {a.select_by}: " + ", ".join(f"{k}={best[k]}" for k in axes))
    print(f"written {path}")


if __name__ == "__main__":
    main()
