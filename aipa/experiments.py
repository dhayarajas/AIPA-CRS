"""End-to-end experimental pipeline: data -> instances -> labels -> models ->
predictions -> metrics / statistics / diagnostics.  All artefacts are written
under ``outputs/results`` as CSV / JSON so that figures, tables and the report
are generated from files, never from values typed by hand."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import ACTIONS, REL2ID, RELATIONSHIPS
from .config import Config, environment_report
from .data import ReDial, load_dataset
from .evaluate import (
    arbitration_metrics,
    bootstrap_ci,
    calibration,
    driver_summary,
    holm_bonferroni,
    paired_test,
    per_sample_ranking,
    relationship_metrics,
)
from .labeling import inject_controlled, label_all, labels_frame, load_human_verified
from .models import AIPA, BASELINE_NAMES, PersistenceTracker, clarification_question
from .preprocess import (
    Instance,
    ItemIndex,
    TextEncoder,
    build_instances,
    build_item_index,
    instances_frame,
    load_instances,
    save_instances,
    tensorise,
)
from .train import label_tensors, predict, train_model

MODEL_ORDER = BASELINE_NAMES + [
    "AIPA w/o relationship", "AIPA w/o counterfactual", "AIPA w/o clarification", "AIPA w/o persistence",
    "AIPA (rule policy)", "AIPA (full)",
]
PRIMARY = "AIPA (full)"


@dataclass
class Results:
    cfg: Config
    per_sample: pd.DataFrame  # one row per (model, seed, sample)
    labels: pd.DataFrame  # test labels (natural + synthetic)
    instances_meta: pd.DataFrame
    efficiency: pd.DataFrame
    training_curves: pd.DataFrame
    counterfactual: pd.DataFrame
    persistence_shifts: pd.DataFrame
    alpha_sweep: pd.DataFrame
    status: dict[str, str] = field(default_factory=dict)  # component -> RUN / NOT RUN reason
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------

def prepare(cfg: Config, verbose: bool = True):
    ds = load_dataset(cfg)
    name = f"instances_{cfg.run_mode}_s{cfg.seed}_f{cfg.subset_fraction}"
    inst = load_instances(cfg, name)
    if inst is None:
        inst = build_instances(ds, cfg)
        save_instances(inst, cfg, name)
    enc = TextEncoder(cfg).fit(
        [x.seeker_recent_text for x in inst["train"]] + [" ".join(x.profile_sentences) for x in inst["train"]]
        + list(ds.movie_titles.values())
    )
    index = build_item_index(ds, enc)
    syn_train = inject_controlled(inst["train"], ds, cfg, inst["train"], cfg.seed)
    syn_test = inject_controlled(inst["test"], ds, cfg, inst["train"], cfg.seed + 1)
    human = load_human_verified(cfg)
    train_all = inst["train"] + syn_train
    test_all = inst["test"] + syn_test
    lab_train = label_all(train_all, cfg, human)
    lab_test = label_all(test_all, cfg, human)
    lab_valid = label_all(inst["valid"], cfg, human)
    if verbose:
        print(f"instances: train={len(inst['train'])} (+{len(syn_train)} synthetic) valid={len(inst['valid'])} "
              f"test={len(inst['test'])} (+{len(syn_test)} synthetic); items={index.n - 1}")
    tensors = {
        "train": tensorise(train_all, enc, index, cfg),
        "valid": tensorise(inst["valid"], enc, index, cfg),
        "test": tensorise(test_all, enc, index, cfg),
    }
    return ds, enc, index, {"train": train_all, "valid": inst["valid"], "test": test_all}, {
        "train": lab_train, "valid": lab_valid, "test": lab_test}, tensors, human


def _persistence_override(model: AIPA, test_inst: list[Instance], X: dict, pred: dict, cfg: Config) -> tuple[torch.Tensor, list[dict]]:
    tracker = PersistenceTracker(k=cfg.persistence_k, gain=cfg.persistence_gain)
    order = sorted(range(len(test_inst)), key=lambda i: (test_inst[i].seeker_id, test_inst[i].conv_id, test_inst[i].turn))
    acts = pred["act_logits"].argmax(1)
    override = X["ltp_genres"].clone()
    seen_sessions: set[tuple[str, int]] = set()
    for i in order:
        x = test_inst[i]
        if x.is_synthetic:
            continue
        override[i] = tracker.adjust(x.seeker_id, X["ltp_genres"][i])
        key = (x.seeker_id, x.conv_id)
        if key not in seen_sessions:
            tracker.observe(x.seeker_id, x.conv_id, ACTIONS[int(acts[i])], x.sti_genres)
            seen_sessions.add(key)
    return override, tracker.shifts


def run_experiments(cfg: Config, verbose: bool = True, models: list[str] | None = None, prepared: tuple | None = None) -> Results:
    t_start = time.perf_counter()
    ds, enc, index, inst, labels, X, human = prepared or prepare(cfg, verbose)
    models = models or MODEL_ORDER
    Y = {k: label_tensors(v) for k, v in labels.items()}
    test_inst = inst["test"]
    test_meta = instances_frame(test_inst)
    lab_df = labels_frame(test_inst, labels["test"])

    rows, eff_rows, curve_rows, cf_rows, shift_rows, alpha_rows = [], [], [], [], [], []
    status = {"human_verified_labels": "RUN" if human is not None else "NOT RUN (no data/annotations/human_verified.csv provided)"}
    trained_primary = {}
    for seed in cfg.seeds:
        for name in models:
            model, info = train_model(name, index.content, X["train"], Y["train"], X["valid"], cfg, seed,
                                      lambda_rel=cfg.lambda_rel, lambda_act=cfg.lambda_act, verbose=verbose)
            pred = predict(model, X["test"], cfg)
            eff = dict(info["efficiency"])
            eff.update({"inference_time_s": round(pred["inference_time_s"], 3),
                        "cpu_inference_ms_per_sample": round(pred["inference_ms_per_sample"], 4)})
            eff_rows.append(eff)
            for h in info["history"]:
                curve_rows.append({"model": name, "seed": seed, **h})
            is_aipa_full = isinstance(model, AIPA) and model.variant.fusion == "aipa"
            if is_aipa_full and model.variant.use_persist:
                override, shifts = _persistence_override(model, test_inst, X["test"], pred, cfg)
                pred = predict(model, X["test"], cfg, ltp_override=override)
                for s in shifts:
                    shift_rows.append({"model": name, "seed": seed, **s})
            df = per_sample_ranking(pred["rank"], ks=tuple(cfg.top_k))
            df.insert(0, "sample_id", [x.sample_id for x in test_inst])
            df.insert(0, "seed", seed)
            df.insert(0, "model", name)
            df["rank"] = pred["rank"]
            if "act_logits" in pred:
                df["rel_pred"] = pred["rel_logits"].argmax(1)
                probs = torch.tensor(pred["rel_logits"]).softmax(-1).numpy()
                df["rel_conf"] = probs.max(1)
                for i, r in enumerate(RELATIONSHIPS):
                    df[f"p_{r}"] = probs[:, i]
                df["act_pred"] = pred["act_logits"].argmax(1)
                df["w_ltp"] = pred["w_ltp"]
                df["w_sti"] = pred["w_sti"]
                cf = pred["cf"]
                df["cf_delta_ltp"], df["cf_delta_sti"] = cf[:, 0], cf[:, 1]
                df["cf_driver"] = model.cf.driver(torch.tensor(cf))
            elif isinstance(model, AIPA):
                df["w_ltp"] = pred["w_ltp"]
                df["w_sti"] = pred["w_sti"]
            rows.append(df)
            if name == PRIMARY:
                trained_primary[seed] = model
                # counterfactual interventions on the *fused* recommender
                p_noltp = predict(model, X["test"], cfg, ltp_scale=0.0)
                p_nosti = predict(model, X["test"], cfg, sti_scale=0.0)
                full_rank = pred["rank"]
                for i, x in enumerate(test_inst):
                    cf_rows.append({
                        "seed": seed, "sample_id": x.sample_id,
                        "rank_full": int(full_rank[i]), "rank_noLTP": int(p_noltp["rank"][i]), "rank_noSTI": int(p_nosti["rank"][i]),
                        "ndcg10_full": float(1 / np.log2(full_rank[i] + 1)) if full_rank[i] <= 10 else 0.0,
                        "ndcg10_noLTP": float(1 / np.log2(p_noltp["rank"][i] + 1)) if p_noltp["rank"][i] <= 10 else 0.0,
                        "ndcg10_noSTI": float(1 / np.log2(p_nosti["rank"][i] + 1)) if p_nosti["rank"][i] <= 10 else 0.0,
                        "overlap10_noLTP": float(len(set(pred["topk"][i][:10]) & set(p_noltp["topk"][i][:10])) / 10),
                        "overlap10_noSTI": float(len(set(pred["topk"][i][:10]) & set(p_nosti["topk"][i][:10])) / 10),
                        "driver": df["cf_driver"].iloc[i],
                    })
            if name == "Naive fusion":
                for a in cfg.alpha_grid:
                    pa = predict(model, X["test"], cfg, fixed_alpha=a)
                    r = per_sample_ranking(pa["rank"], ks=tuple(cfg.top_k))
                    nat = ~lab_df.is_synthetic.values
                    alpha_rows.append({"seed": seed, "alpha_ltp": a, **{c: r[c].values[nat].mean() for c in r.columns},
                                       "Hit@10_synthetic": r["Hit@10"].values[~nat].mean() if (~nat).any() else np.nan})
    per_sample = pd.concat(rows, ignore_index=True)
    per_sample.attrs["sample_ids"] = np.array([x.sample_id for x in test_inst])
    assert len(set(per_sample.attrs["sample_ids"])) == len(test_inst), "sample ids must be unique"
    res = Results(
        cfg=cfg, per_sample=per_sample, labels=lab_df, instances_meta=test_meta,
        efficiency=pd.DataFrame(eff_rows), training_curves=pd.DataFrame(curve_rows),
        counterfactual=pd.DataFrame(cf_rows), persistence_shifts=pd.DataFrame(shift_rows),
        alpha_sweep=pd.DataFrame(alpha_rows), status=status,
    )
    res.extra["primary_models"] = trained_primary
    res.extra["index"] = index
    res.extra["ds"] = ds
    res.extra["test_instances"] = test_inst
    res.extra["test_tensors"] = X["test"]
    res.extra["train_labels"] = labels_frame(inst["train"], labels["train"])
    res.extra["dataset_source"] = ds.source
    res.extra["file_hashes"] = ds.file_hashes
    res.extra["n_train"] = len(inst["train"])
    res.extra["n_valid"] = len(inst["valid"])
    res.extra["runtime_s"] = time.perf_counter() - t_start
    build_tables(res)
    save_results(res)
    return res


# --------------------------------------------------------------------------
# validation-split hyper-parameter sweep (never touches the test split)
# --------------------------------------------------------------------------

def validation_sweep(cfg: Config, grid: list[dict], prepared: tuple | None = None, model_name: str = PRIMARY,
                     seeds: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Train ``model_name`` once per config override in ``grid`` and score it on
    the VALIDATION split: Hit@10 / NDCG@10 on natural instances and relationship
    macro-F1 / Conflict-F1 against the weak-rule validation labels (noisy; not
    human labels).  Returns one row per (override, seed) plus the mean over seeds."""
    ds, enc, index, inst, labels, X, human = prepared or prepare(cfg, verbose)
    Y = {k: label_tensors(v) for k, v in labels.items()}
    rel_true = Y["valid"]["rel"].numpy()
    conflict = np.isin(rel_true, [REL2ID["Conflict"], REL2ID["Override"]])
    rows = []
    for over in grid:
        c = Config(values={**cfg.values, **over}, run_mode=cfg.run_mode)
        for seed in seeds or cfg.seeds[:1]:
            model, info = train_model(model_name, index.content, X["train"], Y["train"], X["valid"], c, seed,
                                      lambda_rel=c.lambda_rel, lambda_act=c.lambda_act, verbose=False)
            pred = predict(model, X["valid"], c)
            r = per_sample_ranking(pred["rank"], ks=(10,))
            row = {**over, "seed": seed, "valid_Hit@10": float(r["Hit@10"].mean()), "valid_NDCG@10": float(r["NDCG@10"].mean()),
                   "valid_conflict_Hit@10": float(r["Hit@10"].values[conflict].mean()) if conflict.any() else np.nan,
                   "epochs_run": info["efficiency"]["epochs_run"], "train_time_s": info["efficiency"]["train_time_s"]}
            if "rel_logits" in pred:
                rm = relationship_metrics(rel_true, pred["rel_logits"].argmax(1))
                row.update(valid_macro_f1=rm["macro_f1"], valid_F1_Conflict=rm["F1_Conflict"], valid_F1_Override=rm["F1_Override"])
            rows.append(row)
            if verbose:
                print("[sweep] " + ", ".join(f"{k}={v}" for k, v in row.items()))
    df = pd.DataFrame(rows)
    keys = [k for k in df.columns if any(k in o for o in grid)]
    if not keys:
        return df
    mean = df.groupby(keys, dropna=False, sort=False).mean(numeric_only=True).drop(columns=["seed"]).reset_index()
    mean["seed"] = "mean"
    return pd.concat([df, mean], ignore_index=True)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _metric_cols(cfg: Config) -> list[str]:
    return [f"{m}@{k}" for k in cfg.top_k for m in ["Recall", "Hit", "NDCG", "MRR"]]


def aggregate(per_sample: pd.DataFrame, mask: np.ndarray | pd.Series, cfg: Config, n_boot: int | None = None) -> pd.DataFrame:
    """Mean +- std over seeds and bootstrap CI (pooled over seeds) per model."""
    n_boot = n_boot or cfg.bootstrap_samples
    cols = _metric_cols(cfg)
    m = np.asarray(mask, bool)
    ids = per_sample.attrs["sample_ids"]
    sub = per_sample[per_sample.sample_id.isin(set(ids[m]))]
    rows = []
    for model, g in sub.groupby("model", sort=False):
        row = {"model": model, "n": int(g.sample_id.nunique()), "seeds": int(g.seed.nunique())}
        seed_means = g.groupby("seed")[cols].mean()
        for c in cols:
            row[f"{c}_mean"] = seed_means[c].mean()
            row[f"{c}_std"] = seed_means[c].std(ddof=0) if len(seed_means) > 1 else 0.0
            _, lo, hi = bootstrap_ci(g[c].values, n_boot, seed=0)
            row[f"{c}_ci_low"], row[f"{c}_ci_high"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return out.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def significance(per_sample: pd.DataFrame, mask, cfg: Config, metric: str = "Hit@10", treatment: str = PRIMARY) -> pd.DataFrame:
    m = np.asarray(mask, bool)
    ids = per_sample.attrs["sample_ids"]
    keep_ids = set(ids[m])
    sub = per_sample[per_sample.sample_id.isin(keep_ids)]
    piv = sub.pivot_table(index="sample_id", columns="model", values=metric, aggfunc="mean")
    if treatment not in piv:
        return pd.DataFrame()
    rows = []
    for model in piv.columns:
        if model == treatment:
            continue
        r = paired_test(piv[treatment].values, piv[model].values)
        rows.append({"treatment": treatment, "control": model, "metric": metric, **r})
    df = pd.DataFrame(rows)
    if len(df):
        df["t_p_holm"] = holm_bonferroni(df["t_p"].tolist())
        df["wilcoxon_p_holm"] = holm_bonferroni(df["wilcoxon_p"].tolist())
    return df


def build_tables(res: Results) -> None:
    cfg, ps, lab = res.cfg, res.per_sample, res.labels
    nat = ~lab.is_synthetic.values
    syn = lab.is_synthetic.values
    T = res.tables
    T["overall_natural"] = aggregate(ps, nat, cfg)
    T["overall_synthetic"] = aggregate(ps, syn, cfg) if syn.any() else pd.DataFrame()
    rel = lab.relationship_label.values
    for r in RELATIONSHIPS:
        m = nat & (rel == r)
        T[f"subset_{r}_natural"] = aggregate(ps, m, cfg) if m.sum() >= 5 else pd.DataFrame()
        m2 = syn & (rel == r)
        T[f"subset_{r}_synthetic"] = aggregate(ps, m2, cfg) if m2.sum() >= 5 else pd.DataFrame()
    conflict = np.isin(rel, ["Conflict", "Override"])
    T["conflict_natural"] = aggregate(ps, nat & conflict, cfg) if (nat & conflict).sum() >= 5 else pd.DataFrame()
    T["nonconflict_natural"] = aggregate(ps, nat & ~conflict, cfg)
    T["conflict_synthetic"] = aggregate(ps, syn & conflict, cfg) if (syn & conflict).sum() >= 5 else pd.DataFrame()
    # significance
    sig = [significance(ps, nat, cfg, "Hit@10"), significance(ps, nat, cfg, "NDCG@10")]
    if (nat & conflict).sum() >= 5:
        for d in (significance(ps, nat & conflict, cfg, "Hit@10"), significance(ps, nat & conflict, cfg, "NDCG@10")):
            d["subset"] = "conflict_natural"
            sig.append(d)
    if (syn & conflict).sum() >= 5:
        d = significance(ps, syn & conflict, cfg, "Hit@10")
        d["subset"] = "conflict_synthetic"
        sig.append(d)
    for d in sig[:2]:
        d["subset"] = "natural"
    T["significance"] = pd.concat([d for d in sig if len(d)], ignore_index=True) if any(len(d) for d in sig) else pd.DataFrame()
    # relationship / arbitration / calibration / drivers per AIPA model
    rel_true = lab.relationship_label.map({r: i for i, r in enumerate(RELATIONSHIPS)}).values
    act_true = lab.gold_action.map({a: i for i, a in enumerate(ACTIONS)}).values
    conf_true = lab.confidence.values
    rel_rows, arb_rows, cal_rows, drv_rows, conf_mats, cal_bins = [], [], [], [], {}, {}
    for (model, seed), g in ps[ps.rel_pred.notna()].groupby(["model", "seed"]) if "rel_pred" in ps else []:
        g = g.set_index("sample_id").loc[lab.sample_id]
        for subset, m in [("natural", nat), ("synthetic", syn), ("all", np.ones_like(nat))]:
            if m.sum() < 5:
                continue
            rm = relationship_metrics(rel_true[m], g.rel_pred.values[m].astype(int))
            conf_mats[(model, seed, subset)] = rm.pop("confusion")
            rel_rows.append({"model": model, "seed": seed, "subset": subset, **rm})
            arb_rows.append({"model": model, "seed": seed, "subset": subset,
                             **arbitration_metrics(act_true[m], g.act_pred.values[m].astype(int), rel_true[m], conf_true[m],
                                                   g["Hit@10"].values[m], cfg.clarification_threshold)})
            probs = g[[f"p_{r}" for r in RELATIONSHIPS]].values[m]
            ece, bins, brier = calibration(probs, rel_true[m])
            cal_rows.append({"model": model, "seed": seed, "subset": subset, "ECE": ece, "Brier": brier})
            cal_bins[(model, seed, subset)] = bins
            drv_rows.append({"model": model, "seed": seed, "subset": subset,
                             **driver_summary(g.cf_driver.values[m].tolist(), g.cf_delta_ltp.values[m], g.cf_delta_sti.values[m],
                                              1 - g.cf_delta_ltp.values[m], 1 - g.cf_delta_sti.values[m])})
    T["relationship"] = pd.DataFrame(rel_rows)
    T["arbitration"] = pd.DataFrame(arb_rows)
    T["calibration"] = pd.DataFrame(cal_rows)
    T["drivers"] = pd.DataFrame(drv_rows)
    res.extra["confusion"] = conf_mats
    res.extra["calibration_bins"] = cal_bins
    # label distribution / class imbalance
    T["label_distribution_test"] = lab.groupby(["relationship_source", "relationship_label"]).size().rename("count").reset_index()
    T["label_distribution_train"] = res.extra["train_labels"].groupby(["relationship_source", "relationship_label"]).size().rename("count").reset_index()
    # sensitivity analyses (primary model, natural)
    meta = res.instances_meta.set_index("sample_id")
    prim = ps[ps.model == PRIMARY].copy()
    prim["history_len"] = prim.sample_id.map(meta.history_len)
    prim["seeker_turns"] = prim.sample_id.map(meta.seeker_turns)
    prim["is_synthetic"] = prim.sample_id.map(meta.is_synthetic)
    prim["intensity"] = prim.sample_id.map(lab.set_index("sample_id").intensity)
    prim["relationship_label"] = prim.sample_id.map(lab.set_index("sample_id").relationship_label)
    natp = prim[~prim.is_synthetic]
    natp = natp.assign(history_bucket=pd.cut(natp.history_len, [-1, 2, 10, 25, 50, 10_000], labels=["0-2", "3-10", "11-25", "26-50", ">50"]),
                       sti_bucket=pd.cut(natp.seeker_turns, [-1, 1, 3, 6, 10_000], labels=["1", "2-3", "4-6", ">6"]))
    T["sens_history"] = natp.groupby("history_bucket", observed=True).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index()
    T["sens_sti_length"] = natp.groupby("sti_bucket", observed=True).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index()
    all_models = ps.copy()
    all_models["intensity"] = all_models.sample_id.map(lab.set_index("sample_id").intensity)
    all_models["relationship_label"] = all_models.sample_id.map(lab.set_index("sample_id").relationship_label)
    synp = all_models[(all_models.intensity > 0) & all_models.relationship_label.isin(["Conflict", "Override"])]
    T["sens_intensity"] = synp.groupby(["model", "intensity"]).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index() if len(synp) else pd.DataFrame()
    if "act_pred" in prim:
        T["action_by_relationship"] = pd.crosstab(prim.relationship_label, prim.act_pred.map(lambda i: ACTIONS[int(i)]), normalize="index").round(3).reset_index()
    T["alpha_sweep"] = res.alpha_sweep.groupby("alpha_ltp").mean(numeric_only=True).drop(columns="seed").reset_index() if len(res.alpha_sweep) else pd.DataFrame()
    T["efficiency"] = res.efficiency.groupby("model", sort=False).agg(
        n_parameters=("n_parameters", "first"), model_size_mb=("model_size_mb", "first"),
        train_time_s=("train_time_s", "mean"), epochs_run=("epochs_run", "mean"),
        inference_time_s=("inference_time_s", "mean"), cpu_inference_ms_per_sample=("cpu_inference_ms_per_sample", "mean"),
        gpu_peak_mem_mb=("gpu_peak_mem_mb", "mean")).reset_index()
    T["persistence_shifts"] = res.persistence_shifts
    if len(res.counterfactual):
        cf = res.counterfactual.copy()
        cf["is_synthetic"] = cf.sample_id.map(meta.is_synthetic)
        cf["relationship_label"] = cf.sample_id.map(lab.set_index("sample_id").relationship_label)
        cf["delta_ndcg_LTP"] = cf.ndcg10_full - cf.ndcg10_noLTP
        cf["delta_ndcg_STI"] = cf.ndcg10_full - cf.ndcg10_noSTI
        T["counterfactual_by_relationship"] = cf.groupby(["is_synthetic", "relationship_label"]).agg(
            n=("sample_id", "nunique"), mean_abs_delta_ndcg_LTP=("delta_ndcg_LTP", lambda s: s.abs().mean()),
            mean_abs_delta_ndcg_STI=("delta_ndcg_STI", lambda s: s.abs().mean()),
            mean_delta_ndcg_LTP=("delta_ndcg_LTP", "mean"), mean_delta_ndcg_STI=("delta_ndcg_STI", "mean"),
            overlap10_noLTP=("overlap10_noLTP", "mean"), overlap10_noSTI=("overlap10_noSTI", "mean"),
            STI_driven=("driver", lambda s: (s == "STI-driven").mean()), LTP_driven=("driver", lambda s: (s == "LTP-driven").mean()),
            Jointly_driven=("driver", lambda s: (s == "Jointly-driven").mean()), Neither_driven=("driver", lambda s: (s == "Neither-driven").mean()),
        ).reset_index()
        # causal-driver agreement: does the driver agree with the predicted arbitration action?
        pa = prim.set_index(["seed", "sample_id"]).act_pred
        cf["act_pred"] = [ACTIONS[int(pa.get((s, i), 0))] for s, i in zip(cf.seed, cf.sample_id)]
        agree = ((cf.act_pred == "Prioritize_STI") & (cf.driver == "STI-driven")) | ((cf.act_pred == "Prioritize_LTP") & (cf.driver == "LTP-driven")) | \
                ((cf.act_pred == "Fuse") & (cf.driver == "Jointly-driven")) | ((cf.act_pred == "Ask_Clarification") & (cf.driver == "Neither-driven"))
        cf["driver_action_agreement"] = agree
        T["driver_action_agreement"] = cf.groupby("is_synthetic").driver_action_agreement.mean().rename("agreement").reset_index()
        res.extra["counterfactual_detail"] = cf
    T["case_studies"] = case_studies(res)
    T["error_analysis"] = error_analysis(res)


# --------------------------------------------------------------------------
# qualitative + error analysis
# --------------------------------------------------------------------------

def case_studies(res: Results, n: int | None = None) -> pd.DataFrame:
    n = n or res.cfg.n_case_studies
    ps = res.per_sample
    lab = res.labels.set_index("sample_id")
    if "act_pred" not in ps or PRIMARY not in set(ps.model):
        return pd.DataFrame()
    seed = res.cfg.seeds[0]
    prim = ps[(ps.model == PRIMARY) & (ps.seed == seed)].set_index("sample_id")
    inst = {x.sample_id: x for x in res.extra["test_instances"]}
    index: ItemIndex = res.extra["index"]
    ds: ReDial = res.extra["ds"]
    model = res.extra["primary_models"][seed]
    chosen: list[str] = []
    rng = np.random.RandomState(0)
    # coverage: every relationship label (natural), every action, synthetic conflict/override, hits and misses
    for r in RELATIONSHIPS:
        ids = [s for s in prim.index if lab.relationship_label[s] == r and not lab.is_synthetic[s] and s not in chosen]
        if ids:
            chosen.append(ids[rng.randint(len(ids))])
    for a in range(len(ACTIONS)):
        ids = [s for s in prim.index if int(prim.act_pred[s]) == a and s not in chosen]
        if ids:
            chosen.append(ids[rng.randint(len(ids))])
    for r in ["Conflict", "Override"]:
        ids = [s for s in prim.index if lab.is_synthetic[s] and lab.relationship_label[s] == r and s not in chosen]
        if ids:
            chosen.append(ids[rng.randint(len(ids))])
    for hit in [1.0, 0.0]:
        ids = [s for s in prim.index if prim["Hit@10"][s] == hit and s not in chosen]
        if ids:
            chosen.append(ids[rng.randint(len(ids))])
    while len(chosen) < n:
        ids = [s for s in prim.index if s not in chosen]
        if not ids:
            break
        chosen.append(ids[rng.randint(len(ids))])
    rows = []
    with torch.no_grad():
        for cid in chosen[:max(n, len(chosen))]:
            x = inst[cid]
            row = prim.loc[cid]
            xb = {k: v[[list(inst).index(cid)]] for k, v in res.extra["test_tensors"].items()} if "test_tensors" in res.extra else None
            top = None
            if xb is not None:
                out = model({k: v.to(res.cfg.device) for k, v in xb.items()})
                top = out["scores"].topk(5, -1).indices[0].tolist()
            rel_pred = RELATIONSHIPS[int(row.rel_pred)]
            act = ACTIONS[int(row.act_pred)]
            rows.append({
                "sample_id": cid, "seeker_id": x.seeker_id, "dialogue_id": x.conv_id, "is_synthetic": x.is_synthetic,
                "dialogue_excerpt": " | ".join(f"{c['role']}: {c['text']}" for c in x.context[-4:]),
                "ltp_profile": "; ".join(f"{g} {v:.2f}" for g, v in sorted(x.ltp_genres.items(), key=lambda kv: -kv[1])[:3]) or "(none: cold seeker)",
                "ltp_history_len": len(x.history_items),
                "sti_signal": "; ".join(f"{g} {v:.2f}" for g, v in sorted(x.sti_genres.items(), key=lambda kv: -kv[1])[:3]) or "(no genre cue)",
                "reference_relationship": lab.relationship_label[cid], "reference_source": lab.relationship_source[cid],
                "predicted_relationship": rel_pred, "relationship_confidence": round(float(row.rel_conf), 3),
                "arbitration_action": act, "w_ltp": round(float(row.w_ltp), 2), "w_sti": round(float(row.w_sti), 2),
                "counterfactual_driver": row.cf_driver,
                "clarification": clarification_question(x.ltp_genres, x.sti_genres, rel_pred) if act == "Ask_Clarification" else "",
                "target": ds.movie_titles.get(x.target, str(x.target)),
                "target_rank": int(row["rank"]),
                "top5": "; ".join(ds.movie_titles.get(index.ids[t], "?") for t in top) if top else "",
                "hit@10": bool(row["Hit@10"]),
            })
    return pd.DataFrame(rows)


def error_analysis(res: Results) -> pd.DataFrame:
    ps, lab = res.per_sample, res.labels.set_index("sample_id")
    if PRIMARY not in set(ps.model):
        return pd.DataFrame()
    prim = ps[ps.model == PRIMARY].copy()
    prim["relationship_label"] = prim.sample_id.map(lab.relationship_label)
    prim["is_synthetic"] = prim.sample_id.map(lab.is_synthetic)
    prim["history_len"] = prim.sample_id.map(res.instances_meta.set_index("sample_id").history_len)
    rows = []
    for (syn, r), g in prim.groupby(["is_synthetic", "relationship_label"]):
        rows.append({
            "subset": "synthetic" if syn else "natural", "relationship_label": r, "n": int(g.sample_id.nunique()),
            "miss_rate@10": float(1 - g["Hit@10"].mean()),
            "relationship_error_rate": float((g.rel_pred.map(lambda i: RELATIONSHIPS[int(i)]) != g.relationship_label).mean()) if "rel_pred" in g else np.nan,
            "clarification_rate": float((g.act_pred == ACTIONS.index("Ask_Clarification")).mean()) if "act_pred" in g else np.nan,
            "mean_target_rank": float(g["rank"].mean()),
            "median_target_rank": float(g["rank"].median()),
            "cold_seeker_share": float((g.history_len < res.cfg.min_history_for_ltp).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_results(res: Results) -> None:
    out = res.cfg.path("output_path") / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.per_sample.to_csv(out / "per_sample_metrics.csv.gz", index=False, compression="gzip")
    res.labels.to_csv(out / "test_relationship_labels.csv", index=False)
    res.extra["train_labels"].to_csv(out / "train_relationship_labels.csv", index=False)
    res.instances_meta.to_csv(out / "test_instances_meta.csv", index=False)
    res.efficiency.to_csv(out / "efficiency_raw.csv", index=False)
    res.training_curves.to_csv(out / "training_curves.csv", index=False)
    res.counterfactual.to_csv(out / "counterfactual_raw.csv", index=False)
    res.alpha_sweep.to_csv(out / "alpha_sweep_raw.csv", index=False)
    for k, v in res.tables.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_csv(out / f"table_{k}.csv", index=False)
    meta = {
        "run_mode": res.cfg.run_mode, "config": res.cfg.to_dict(), "environment": environment_report(),
        "status": res.status, "dataset_source": res.extra.get("dataset_source"), "file_hashes": res.extra.get("file_hashes"),
        "n_train": res.extra.get("n_train"), "n_valid": res.extra.get("n_valid"), "n_test": int(len(res.labels)),
        "runtime_s": res.extra.get("runtime_s"), "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
