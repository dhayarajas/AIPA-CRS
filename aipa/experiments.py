"""End-to-end experimental pipeline: data -> instances -> labels -> models ->
predictions -> metrics / statistics / diagnostics.  All artefacts are written
under ``outputs/results`` as CSV / JSON so that figures, tables and the report
are generated from files, never from values typed by hand."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import ACTIONS, RELATIONSHIPS
from .config import Config, environment_report
from .data import ReDial, load_dataset
from .evaluate import (
    arbitration_metrics,
    bootstrap_ci,
    calibration,
    driver_summary,
    holm_bonferroni,
    per_sample_ranking,
    pooled_paired_test,
    relationship_metrics,
)
from .labeling import (
    disagreement_mask,
    inject_controlled,
    label_all,
    labels_frame,
    load_human_verified,
    strict_conflict_mask,
)
from .models import AIPA, BASELINE_NAMES, PersistenceTracker, clarification_question
from .preprocess import (
    Instance,
    ItemIndex,
    TextEncoder,
    build_instances,
    build_item_index,
    instances_frame,
    item_texts,
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
    data_tag = hashlib.sha1(json.dumps(ds.file_hashes, sort_keys=True).encode()).hexdigest()[:12]
    name = f"instances_{cfg.run_mode}_s{cfg.seed}_f{cfg.subset_fraction}_{data_tag}"
    inst = load_instances(cfg, name)
    if inst is None:
        inst = build_instances(ds, cfg)
        save_instances(inst, cfg, name)
    enc = TextEncoder(cfg).fit(
        [x.seeker_recent_text for x in inst["train"]] + [" ".join(x.profile_sentences) for x in inst["train"]]
        + item_texts(ds, cfg)
    )
    index = build_item_index(ds, enc, cfg)
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
    if verbose:
        s = enc.summary()
        print(f"text encoder: {s['name']} (dim={s['dim']}) encoded {s['n_newly_encoded']} new strings, "
              f"{s['n_cache_hits']} cache hits, {s['encode_seconds']}s")
    return ds, enc, index, {"train": train_all, "valid": inst["valid"], "test": test_all}, {
        "train": lab_train, "valid": lab_valid, "test": lab_test}, tensors, human


def _persistence_override(model: AIPA, test_inst: list[Instance], X: dict, pred: dict, cfg: Config,
                          k: int | None = None) -> tuple[torch.Tensor, list[dict], np.ndarray]:
    """Replay the persistence tracker over the natural instances of a split in
    chronological order per seeker (``conv_id`` then ``turn``).  Every
    recommendation turn is observed (a genre counts once per session), and the
    adjusted LTP prior is applied to all later turns of that seeker.  Returns
    the adjusted ``ltp_genres`` tensor, the detected shifts and a boolean mask
    of the instances whose prior was actually changed."""
    tracker = PersistenceTracker(k=k if k is not None else cfg.persistence_k, gain=cfg.persistence_gain)
    order = sorted(range(len(test_inst)), key=lambda i: (test_inst[i].seeker_id, test_inst[i].conv_id, test_inst[i].turn))
    acts = pred["act_logits"].argmax(1)
    override = X["ltp_genres"].clone()
    affected = np.zeros(len(test_inst), bool)
    for i in order:
        x = test_inst[i]
        if x.is_synthetic:
            continue
        adj = tracker.adjust(x.seeker_id, X["ltp_genres"][i])
        if not torch.allclose(adj, X["ltp_genres"][i]):
            override[i] = adj
            affected[i] = True
        tracker.observe(x.seeker_id, x.conv_id, ACTIONS[int(acts[i])], x.sti_genres)
    return override, tracker.shifts, affected


def _sessions_per_seeker(instances: list[Instance]) -> np.ndarray:
    """Number of distinct (natural) sessions of each instance's seeker within the split."""
    sess: dict[str, set[int]] = {}
    for x in instances:
        if not x.is_synthetic:
            sess.setdefault(x.seeker_id, set()).add(x.conv_id)
    return np.array([len(sess.get(x.seeker_id, ())) for x in instances])


def _persistence_sweep(model: AIPA, inst: list[Instance], X: dict, cfg: Config, seed: int, split: str) -> list[dict]:
    """Effect of the tracker for each k in ``persistence_k_grid`` on one split."""
    base = predict(model, X, cfg)
    n_sess = _sessions_per_seeker(inst)
    nat = np.array([not x.is_synthetic for x in inst])
    multi = nat & (n_sess >= cfg.persistence_min_sessions)
    rows = []
    for k in cfg.persistence_k_grid:
        override, shifts, affected = _persistence_override(model, inst, X, base, cfg, k=k)
        rank_b = base["rank"]
        rank_w = predict(model, X, cfg, ltp_override=override)["rank"] if affected.any() else rank_b
        hit_b, hit_w = (rank_b <= 10).astype(float), (rank_w <= 10).astype(float)
        rows.append({
            "split": split, "seed": seed, "k": k, "n_shifts": len(shifts), "n_seekers_shifted": len({s["seeker_id"] for s in shifts}),
            "n_multi_session": int(multi.sum()), "n_seekers_multi_session": len({inst[i].seeker_id for i in np.flatnonzero(multi)}),
            "n_affected": int(affected.sum()),
            "hit10_multi_without": float(hit_b[multi].mean()) if multi.any() else np.nan,
            "hit10_multi_with": float(hit_w[multi].mean()) if multi.any() else np.nan,
            "hit10_affected_without": float(hit_b[affected].mean()) if affected.any() else np.nan,
            "hit10_affected_with": float(hit_w[affected].mean()) if affected.any() else np.nan,
            "n_rank_changed": int((rank_b != rank_w).sum()),
        })
    return rows


def run_experiments(cfg: Config, verbose: bool = True, models: list[str] | None = None, prepared: tuple | None = None) -> Results:
    t_start = time.perf_counter()
    ds, enc, index, inst, labels, X, human = prepared or prepare(cfg, verbose)
    disabled = set(cfg.values.get("disabled_models", []))
    models = models or [m for m in MODEL_ORDER if m not in disabled]
    Y = {k: label_tensors(v) for k, v in labels.items()}
    test_inst = inst["test"]
    test_meta = instances_frame(test_inst)
    lab_df = labels_frame(test_inst, labels["test"])

    rows, eff_rows, curve_rows, cf_rows, shift_rows, alpha_rows, persist_rows = [], [], [], [], [], [], []
    test_sessions = _sessions_per_seeker(test_inst)
    affected_by_seed: dict[int, np.ndarray] = {}
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
                override, shifts, affected = _persistence_override(model, test_inst, X["test"], pred, cfg)
                if affected.any():
                    pred = predict(model, X["test"], cfg, ltp_override=override)
                for s in shifts:
                    shift_rows.append({"model": name, "seed": seed, **s})
                if name == PRIMARY:
                    affected_by_seed[seed] = affected
                    persist_rows += _persistence_sweep(model, inst["valid"], X["valid"], cfg, seed, "valid")
                    persist_rows += _persistence_sweep(model, test_inst, X["test"], cfg, seed, "test")
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
    res.extra["persistence_sweep"] = pd.DataFrame(persist_rows)
    res.extra["persistence_affected"] = affected_by_seed
    res.extra["test_sessions_per_seeker"] = test_sessions
    res.extra["index"] = index
    res.extra["ds"] = ds
    res.extra["test_instances"] = test_inst
    res.extra["test_tensors"] = X["test"]
    res.extra["train_labels"] = labels_frame(inst["train"], labels["train"])
    res.extra["text_encoder"] = enc.summary()
    res.extra["dataset_source"] = ds.source
    res.extra["file_hashes"] = ds.file_hashes
    res.extra["n_train"] = len(inst["train"])
    res.extra["n_valid"] = len(inst["valid"])
    res.extra["runtime_s"] = time.perf_counter() - t_start
    build_tables(res)
    save_results(res)
    return res


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
    """Paired tests of ``treatment`` against every other model.  Differences are
    formed per sample *within* each seed and pooled over seeds (``n`` = samples x
    seeds); p-values are Holm-corrected within the table."""
    m = np.asarray(mask, bool)
    ids = per_sample.attrs["sample_ids"]
    keep_ids = set(ids[m])
    sub = per_sample[per_sample.sample_id.isin(keep_ids)]
    piv = sub.pivot_table(index="sample_id", columns=["model", "seed"], values=metric, aggfunc="mean")
    models = piv.columns.get_level_values(0).unique()
    if treatment not in models:
        return pd.DataFrame()
    treat = {s: piv[(treatment, s)].values for s in piv[treatment].columns}
    rows = []
    for model in models:
        if model == treatment:
            continue
        ctrl = {s: piv[(model, s)].values for s in piv[model].columns}
        r = pooled_paired_test(treat, ctrl, n_perm=cfg.permutation_samples, seed=0)
        rows.append({"treatment": treatment, "control": model, "metric": metric, **r})
    df = pd.DataFrame(rows)
    if len(df):
        for c in ["t_p", "wilcoxon_p", "perm_p"]:
            df[f"{c}_holm"] = holm_bonferroni(df[c].tolist())
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
    strict = strict_conflict_mask(lab, cfg)
    broad = disagreement_mask(lab, cfg)
    subsets = {"strict": strict, "broad": broad}
    res.extra["conflict_subsets"] = subsets
    conf_tabs = []
    for name, m in subsets.items():
        if m.sum() >= 5:
            t = aggregate(ps, m, cfg)
            t.insert(0, "subset", name)
            conf_tabs.append(t)
    T["conflict_natural"] = pd.concat(conf_tabs, ignore_index=True) if conf_tabs else pd.DataFrame()
    T["nonconflict_natural"] = aggregate(ps, nat & ~broad, cfg)
    T["conflict_synthetic"] = aggregate(ps, syn & conflict, cfg) if (syn & conflict).sum() >= 5 else pd.DataFrame()
    T["conflict_subset_sizes"] = pd.DataFrame([
        {"subset": "strict", "definition": "weak-rule label in " + "/".join(cfg.conflict_strict_labels), "n": int(strict.sum())},
        {"subset": "broad", "definition": f"Conflict/Override or (confidence >= {cfg.disagreement_conf_min} and JS(ltp, sti) >= {cfg.disagreement_js_min})",
         "n": int(broad.sum())},
        {"subset": "broad_only", "definition": "broad minus strict", "n": int((broad & ~strict).sum())},
        {"subset": "synthetic_conflict", "definition": "synthetic Conflict/Override", "n": int((syn & conflict).sum())},
        {"subset": "natural", "definition": "all natural test instances", "n": int(nat.sum())},
    ])
    # significance (differences within seed, pooled over seeds)
    sig = []
    for subset, m in [("natural", nat), ("conflict_natural_strict", strict), ("conflict_natural_broad", broad),
                      ("conflict_synthetic", syn & conflict)]:
        if m.sum() < 5:
            continue
        for metric in (["Hit@10", "NDCG@10"] if subset != "conflict_synthetic" else ["Hit@10"]):
            d = significance(ps, m, cfg, metric)
            if len(d):
                d["subset"] = subset
                sig.append(d)
    T["significance"] = pd.concat(sig, ignore_index=True) if sig else pd.DataFrame()
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
    natp = natp.assign(history_bucket=history_bucket(natp.history_len, cfg),
                       sti_bucket=pd.cut(natp.seeker_turns, [-1, 1, 3, 6, 10_000], labels=["1", "2-3", "4-6", ">6"]))
    T["sens_history"] = natp.groupby("history_bucket", observed=True).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index()
    T["sens_sti_length"] = natp.groupby("sti_bucket", observed=True).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index()
    all_models = ps.copy()
    all_models["intensity"] = all_models.sample_id.map(lab.set_index("sample_id").intensity)
    all_models["relationship_label"] = all_models.sample_id.map(lab.set_index("sample_id").relationship_label)
    all_models["is_synthetic"] = all_models.sample_id.map(meta.is_synthetic)
    synp = all_models[all_models.is_synthetic & all_models.relationship_label.isin(["Conflict", "Override"])]
    T["sens_intensity"] = synp.groupby(["model", "intensity"]).agg(n=("sample_id", "nunique"), **{c: (c, "mean") for c in _metric_cols(cfg)}).reset_index() if len(synp) else pd.DataFrame()
    T["conflict_synthetic_by_intensity"] = _seed_agg(synp, ["intensity", "model"], cfg) if len(synp) else pd.DataFrame()
    # per-history-length bucket (Hit@10, every model) and per-target-genre breakdown
    natm = all_models[~all_models.is_synthetic].copy()
    natm["history_bucket"] = history_bucket(natm.sample_id.map(meta.history_len), cfg)
    T["history_buckets"] = _seed_agg(natm, ["history_bucket", "model"], cfg, cols=["Hit@10", "NDCG@10"])
    genres = res.extra["ds"].movie_genres if "ds" in res.extra else {}
    tg = natm.sample_id.map(meta.target).map(lambda t: genres.get(int(t), []))
    top_genres = pd.Series([g for gl in tg[natm.model == PRIMARY] for g in gl]).value_counts().head(cfg.genre_breakdown_top).index.tolist()
    exploded = natm.assign(target_genre=tg).explode("target_genre")
    exploded = exploded[exploded.target_genre.isin(top_genres)]
    T["genre_breakdown"] = _seed_agg(exploded, ["target_genre", "model"], cfg, cols=["Hit@10", "NDCG@10"]) if len(exploded) else pd.DataFrame()
    # persistence: effect on seekers with >= persistence_min_sessions test sessions
    T["persistence_sweep"] = _persistence_sweep_table(res)
    T["persistence_effect"] = _persistence_effect(res)
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
    T["success_criteria"] = success_criteria(res)


def history_bucket(history_len: pd.Series, cfg: Config) -> pd.Series:
    """Map history length (items) to the configured named buckets."""
    edges, labels = [-np.inf], []
    for name, _lo, hi in cfg.history_buckets:
        labels.append(str(name))
        edges.append(np.inf if hi is None else float(hi))
    return pd.cut(pd.Series(history_len).astype(float).values, edges, labels=labels, include_lowest=True)


def _seed_agg(df: pd.DataFrame, by: list[str], cfg: Config, cols: list[str] | None = None) -> pd.DataFrame:
    """Mean +- std over seeds of per-seed means, with n = distinct samples."""
    cols = cols or _metric_cols(cfg)
    if not len(df):
        return pd.DataFrame()
    seed_means = df.groupby(by + ["seed"], observed=True)[cols].mean().reset_index()
    agg = seed_means.groupby(by, observed=True)[cols].agg(["mean", lambda s: s.std(ddof=0)])
    agg.columns = [f"{c}_{'mean' if f == 'mean' else 'std'}" for c, f in agg.columns]
    n = df.groupby(by, observed=True).sample_id.nunique().rename("n")
    seeds = df.groupby(by, observed=True).seed.nunique().rename("seeds")
    out = pd.concat([n, seeds, agg], axis=1).reset_index()
    if "model" in by:
        order = {m: i for i, m in enumerate(MODEL_ORDER)}
        out["_o"] = out.model.map(order)
        out = out.sort_values([c for c in by if c != "model"] + ["_o"]).drop(columns="_o").reset_index(drop=True)
    return out


def _persistence_sweep_table(res: Results) -> pd.DataFrame:
    sw = res.extra.get("persistence_sweep")
    if sw is None or not len(sw):
        return pd.DataFrame()
    num = [c for c in sw.columns if c not in ("split", "seed", "k")]
    g = sw.groupby(["split", "k"])[num]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std")).reset_index()
    out["seeds"] = sw.groupby(["split", "k"]).seed.nunique().values
    out["hit10_multi_delta_mean"] = out.hit10_multi_with_mean - out.hit10_multi_without_mean
    return out


def _persistence_effect(res: Results) -> pd.DataFrame:
    """AIPA (full) vs AIPA w/o persistence on natural test instances of seekers
    with >= ``persistence_min_sessions`` sessions, and on the instances whose LTP
    prior the tracker actually changed.  Paired within seed, pooled over seeds."""
    cfg, ps, lab = res.cfg, res.per_sample, res.labels
    ctrl = "AIPA w/o persistence"
    if PRIMARY not in set(ps.model) or ctrl not in set(ps.model):
        return pd.DataFrame()
    n_sess = res.extra.get("test_sessions_per_seeker")
    if n_sess is None:
        return pd.DataFrame()
    nat = ~lab.is_synthetic.values
    multi = nat & (np.asarray(n_sess) >= cfg.persistence_min_sessions)
    affected_by_seed = res.extra.get("persistence_affected", {})
    affected_any = np.zeros(len(lab), bool)
    for a in affected_by_seed.values():
        affected_any |= np.asarray(a, bool)
    shifts = res.persistence_shifts
    n_shift = shifts[shifts.model == PRIMARY].groupby("seed").size() if len(shifts) else pd.Series(dtype=int)
    rows = []
    for subset, m in [("natural", nat), (f"seekers_with_ge{cfg.persistence_min_sessions}_sessions", multi), ("tracker_affected", affected_any)]:
        row = {"subset": subset, "n": int(m.sum()), "n_seekers": int(lab.seeker_id[m].nunique()) if "seeker_id" in lab else np.nan,
               "persistence_k": cfg.persistence_k, "n_shifts_mean": float(n_shift.mean()) if len(n_shift) else 0.0}
        if m.sum() == 0:
            rows.append({**row, "hit10_full": np.nan, "hit10_without": np.nan, "mean_diff": np.nan})
            continue
        agg = aggregate(ps, m, cfg).set_index("model")
        sig = significance(ps, m, cfg, "Hit@10")
        s = sig[sig.control == ctrl].iloc[0].to_dict() if len(sig) and (sig.control == ctrl).any() else {}
        rows.append({**row, "hit10_full": float(agg.loc[PRIMARY, "Hit@10_mean"]), "hit10_without": float(agg.loc[ctrl, "Hit@10_mean"]),
                     "mean_diff": s.get("mean_diff", np.nan), "t_p": s.get("t_p", np.nan), "wilcoxon_p": s.get("wilcoxon_p", np.nan),
                     "perm_p": s.get("perm_p", np.nan), "n_pairs": s.get("n", 0)})
    return pd.DataFrame(rows)


def success_criteria(res: Results) -> pd.DataFrame:
    """Automatic met / not met verdicts computed from the result tables."""
    cfg, T = res.cfg, res.tables
    alpha = cfg.criteria_alpha
    rows: list[dict] = []

    def add(cid, hypothesis, criterion, value, threshold, met, note=""):
        rows.append({"id": cid, "hypothesis": hypothesis, "criterion": criterion, "value": value, "threshold": threshold,
                     "met": "met" if met is True else ("not met" if met is False else "not evaluated"), "note": note})

    sig = T.get("significance", pd.DataFrame())
    baselines = set(BASELINE_NAMES)

    def best_baseline(table):
        if table is None or not len(table) or PRIMARY not in set(table.model):
            return None, None, None
        b = table[table.model.isin(baselines)]
        if not len(b):
            return None, None, None
        best = b.sort_values("Hit@10_mean", ascending=False).iloc[0]
        prim = table[table.model == PRIMARY].iloc[0]
        return prim, best, float(prim["Hit@10_mean"] - best["Hit@10_mean"])

    def sig_row(subset, control, metric="Hit@10"):
        if not len(sig):
            return None
        d = sig[(sig.subset == subset) & (sig.control == control) & (sig.metric == metric)]
        return d.iloc[0] if len(d) else None

    # H1 overall
    prim, best, diff = best_baseline(T.get("overall_natural"))
    if prim is not None:
        s = sig_row("natural", best.model)
        p = float(s["perm_p_holm"]) if s is not None else np.nan
        add("H1", "H1: AIPA (full) beats the best baseline on natural Hit@10", f"Hit@10 gain vs {best.model} > 0 and Holm perm p < {alpha}",
            round(diff, 4), f"> 0, p < {alpha}", bool(diff > 0 and p < alpha), f"p_holm={p:.3g}, n={int(prim['n'])}")
    else:
        add("H1", "H1: overall", "Hit@10 gain vs best baseline", np.nan, "", None, "overall table missing")
    # H2 natural conflict (strict and broad)
    conf = T.get("conflict_natural", pd.DataFrame())
    for subset in ["strict", "broad"]:
        t = conf[conf.subset == subset] if len(conf) and "subset" in conf else pd.DataFrame()
        prim, best, diff = best_baseline(t)
        if prim is None:
            add(f"H2-{subset}", f"H2: natural conflict ({subset})", "Hit@10 gain vs best baseline", np.nan, "", None, "subset too small (< 5) or missing")
            continue
        s = sig_row(f"conflict_natural_{subset}", best.model)
        p = float(s["perm_p_holm"]) if s is not None else np.nan
        add(f"H2-{subset}", f"H2: AIPA (full) beats the best baseline on natural conflict ({subset}) Hit@10",
            f"gain vs {best.model} > 0 and Holm perm p < {alpha}", round(diff, 4), f"> 0, p < {alpha}", bool(diff > 0 and p < alpha),
            f"p_holm={p:.3g}, n={int(prim['n'])}")
    # H3 synthetic conflict
    prim, best, diff = best_baseline(T.get("conflict_synthetic"))
    if prim is not None:
        s = sig_row("conflict_synthetic", best.model)
        p = float(s["perm_p_holm"]) if s is not None else np.nan
        add("H3", "H3: AIPA (full) beats the best baseline on synthetic conflict Hit@10", f"gain vs {best.model} > 0 and Holm perm p < {alpha}",
            round(diff, 4), f"> 0, p < {alpha}", bool(diff > 0 and p < alpha), f"p_holm={p:.3g}, n={int(prim['n'])} (synthetic, reported separately)")
    else:
        add("H3", "H3: synthetic conflict", "Hit@10 gain vs best baseline", np.nan, "", None, "subset missing")
    # relationship classification
    relt = T.get("relationship", pd.DataFrame())
    if len(relt) and (relt.model == PRIMARY).any():
        r = relt[(relt.model == PRIMARY) & (relt.subset == "natural")]
        mf1 = float(r.macro_f1.mean())
        cf1 = float(r.F1_Conflict.mean()) if "F1_Conflict" in r else np.nan
        add("REL-macroF1", "Relationship classifier (natural, weak-rule reference)", f"macro-F1 >= {cfg.criteria_macro_f1_min}",
            round(mf1, 4), f">= {cfg.criteria_macro_f1_min}", bool(mf1 >= cfg.criteria_macro_f1_min), "reference labels are weak-rule labels, not human labels")
        add("REL-ConflictF1", "Relationship classifier (natural, weak-rule reference)", f"Conflict-F1 >= {cfg.criteria_conflict_f1_min}",
            round(cf1, 4), f">= {cfg.criteria_conflict_f1_min}", bool(cf1 >= cfg.criteria_conflict_f1_min) if not np.isnan(cf1) else None, "")
    else:
        add("REL-macroF1", "Relationship classifier", "macro-F1", np.nan, "", None, "relationship table missing")
    # arbitration
    arb = T.get("arbitration", pd.DataFrame())
    if len(arb) and (arb.model == PRIMARY).any():
        a = arb[(arb.model == PRIMARY) & (arb.subset == "natural")]
        prec = float(a.clarification_precision.mean()) if "clarification_precision" in a else np.nan
        unn = float(a.unnecessary_clarification_rate.mean()) if "unnecessary_clarification_rate" in a else np.nan
        add("ARB-precision", "Arbitration: clarification precision (natural)", f">= {cfg.criteria_clarification_precision_min}", round(prec, 4),
            f">= {cfg.criteria_clarification_precision_min}", bool(prec >= cfg.criteria_clarification_precision_min) if not np.isnan(prec) else None, "")
        add("ARB-unnecessary", "Arbitration: unnecessary clarification rate (natural)", f"<= {cfg.criteria_unnecessary_clarification_max}", round(unn, 4),
            f"<= {cfg.criteria_unnecessary_clarification_max}", bool(unn <= cfg.criteria_unnecessary_clarification_max) if not np.isnan(unn) else None, "")
    # ablations: every ablation must be below AIPA (full) on natural Hit@10 (mean over seeds)
    ov = T.get("overall_natural", pd.DataFrame())
    if len(ov) and PRIMARY in set(ov.model):
        full = float(ov.set_index("model").loc[PRIMARY, "Hit@10_mean"])
        for abl in [m for m in MODEL_ORDER if m.startswith("AIPA w/o") and m in set(ov.model)]:
            v = float(ov.set_index("model").loc[abl, "Hit@10_mean"])
            s = sig_row("natural", abl)
            p = float(s["perm_p_holm"]) if s is not None else np.nan
            add(f"ABL-{abl.split('w/o ')[1]}", f"Ablation: removing {abl.split('w/o ')[1]} hurts natural Hit@10", f"AIPA (full) - {abl} > 0 and Holm perm p < {alpha}",
                round(full - v, 4), f"> 0, p < {alpha}", bool(full - v > 0 and p < alpha), f"p_holm={p:.3g}")
    pe = T.get("persistence_effect", pd.DataFrame())
    if len(pe):
        aff = pe[pe.subset == "tracker_affected"].iloc[0]
        add("ABL-persistence-affected", "Persistence tracker changes Hit@10 on the instances it affected", f"n > 0, gain > 0 and perm p < {alpha}",
            aff.mean_diff if not np.isnan(aff.mean_diff) else np.nan, f"n > 0, p < {alpha}",
            bool(aff.n > 0 and aff.mean_diff > 0 and aff.get("perm_p", np.nan) < alpha) if aff.n > 0 else None,
            f"n_affected={int(aff.n)}, shifts/seed={aff.n_shifts_mean:.1f}" + ("; tracker never fired" if aff.n == 0 else ""))
    return pd.DataFrame(rows)


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
    sc = res.tables.get("success_criteria")
    if isinstance(sc, pd.DataFrame) and len(sc):
        sc.to_csv(out / "success_criteria.csv", index=False)
    sw = res.extra.get("persistence_sweep")
    if isinstance(sw, pd.DataFrame) and len(sw):
        sw.to_csv(out / "persistence_sweep_raw.csv", index=False)
    meta = {
        "run_mode": res.cfg.run_mode, "config": res.cfg.to_dict(), "environment": environment_report(),
        "status": res.status, "text_encoder": res.extra.get("text_encoder"),
        "dataset_source": res.extra.get("dataset_source"), "file_hashes": res.extra.get("file_hashes"),
        "n_train": res.extra.get("n_train"), "n_valid": res.extra.get("n_valid"), "n_test": int(len(res.labels)),
        "runtime_s": res.extra.get("runtime_s"), "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
