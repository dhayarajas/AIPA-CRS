"""Top-level orchestration used by the notebook and the CLI (``python -m aipa.pipeline``)."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import (
    dataset_statistics,
    dataset_status,
    download_dataset,
    genre_frame,
    load_dataset,
    per_seeker_frame,
    validate_dataset,
)
from .experiments import PRIMARY, Results, run_experiments
from .figures import make_all
from .report import build_report


def write_tables(res: Results) -> Path:
    out = res.cfg.path("output_path") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.tables.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_csv(out / f"{k}.csv", index=False)
            v.to_markdown(out / f"{k}.md", index=False)
    return out


def validate_components(res: Results, figures: dict, report_paths: tuple[Path, Path] | None) -> pd.DataFrame:
    """PASS / FAIL / NOT RUN for every component of the pipeline."""
    T = res.tables
    out = res.cfg.path("output_path")
    checks: list[tuple[str, str, str]] = []

    def add(name, ok, note=""):
        checks.append((name, "PASS" if ok else "FAIL", note))

    add("dataset acquisition & validation", bool(res.extra.get("file_hashes")), f"{len(res.extra.get('file_hashes', {}))} files hashed")
    add("instance construction (leak-free)", res.extra.get("n_train", 0) > 0 and len(res.labels) > 0, f"train={res.extra.get('n_train')}, test={len(res.labels)}")
    add("weak-rule relationship labels", (res.labels.relationship_source == "weak_rule").any(), "")
    add("controlled synthetic injection", res.labels.is_synthetic.any(), f"{int(res.labels.is_synthetic.sum())} synthetic test instances")
    checks.append(("human-verified labels", "PASS" if res.status.get("human_verified_labels") == "RUN" else "NOT RUN", res.status.get("human_verified_labels", "")))
    models = set(res.per_sample.model)
    for m in ["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion", "Sequential (GRU)", "Conversation-aware"]:
        add(f"baseline: {m}", m in models, "")
    for m in ["AIPA w/o relationship", "AIPA w/o counterfactual", "AIPA w/o clarification", "AIPA w/o persistence", "AIPA (rule policy)", PRIMARY]:
        add(f"model: {m}", m in models, "")
    add("relationship classifier metrics", len(T.get("relationship", [])) > 0, "")
    add("arbitration & clarification metrics", len(T.get("arbitration", [])) > 0, "")
    add("counterfactual driver diagnostic", len(res.counterfactual) > 0, "")
    add("temporal persistence tracker", "persistence_shifts" in T, f"{len(T.get('persistence_shifts', []))} shifts detected")
    add("ranking metrics + bootstrap CI", len(T.get("overall_natural", [])) > 0, "")
    add("paired significance tests", len(T.get("significance", [])) > 0, "")
    add("multi-seed evaluation", res.per_sample.seed.nunique() >= 2, f"{res.per_sample.seed.nunique()} seed(s)" + ("" if res.per_sample.seed.nunique() >= 2 else " - increase `seeds` for std estimates"))
    add("conflict-sensitive evaluation", len(T.get("conflict_synthetic", [])) > 0 or len(T.get("conflict_natural", [])) > 0, "")
    add("sensitivity analyses", len(T.get("sens_history", [])) > 0 and len(T.get("sens_intensity", [])) > 0, "")
    add("alpha sweep", len(T.get("alpha_sweep", [])) > 0, "")
    add("calibration analysis", len(T.get("calibration", [])) > 0, "")
    add("efficiency accounting", len(T.get("efficiency", [])) > 0, "")
    add("error analysis", len(T.get("error_analysis", [])) > 0, "")
    add("case studies (>=10)", len(T.get("case_studies", [])) >= 10, f"{len(T.get('case_studies', []))} cases")
    add("figures", len(figures) >= 10, f"{len(figures)} figures in {out / 'figures'}")
    add("tables", (out / "tables").exists() and any((out / "tables").iterdir()), "")
    add("results serialised", (out / "results" / "run_metadata.json").exists(), "")
    add("report (Markdown + HTML)", report_paths is not None and all(p.exists() for p in report_paths), "")
    return pd.DataFrame(checks, columns=["component", "status", "note"])


def run_all(run_mode: str | None = None, verbose: bool = True, clean_outputs: bool = True) -> tuple[Results, pd.DataFrame]:
    t0 = time.perf_counter()
    cfg = load_config(run_mode)
    if clean_outputs:
        shutil.rmtree(cfg.path("output_path"), ignore_errors=True)
    print(dataset_status(cfg).to_string())
    if not dataset_status(cfg).query("source == 'ReDial'").present.all():
        download_dataset(cfg)
    validate_dataset(cfg)
    ds = load_dataset(cfg)
    stats = dataset_statistics(ds)
    res = run_experiments(cfg, verbose=verbose)
    figs = make_all(res, stats, genre_frame(ds), per_seeker_frame(ds))
    write_tables(res)
    paths = build_report(res, figs, None, stats)
    val = validate_components(res, figs, paths)
    paths = build_report(res, figs, val, stats)
    val.to_csv(cfg.path("output_path") / "results" / "component_validation.csv", index=False)
    if verbose:
        print(val.to_string(index=False))
        print(f"total runtime {time.perf_counter() - t0:.1f}s; report: {paths[0]}")
    return res, val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-mode", default=None, choices=["quick", "full"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    _, val = run_all(a.run_mode, verbose=not a.quiet)
    sys.exit(0 if (val.status != "FAIL").all() else 1)
