"""Publication-quality figures (matplotlib / seaborn), generated from Results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import RELATIONSHIPS  # noqa: E402
from .experiments import MODEL_ORDER, PRIMARY, Results  # noqa: E402

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = sns.color_palette("colorblind")


def _save(fig, out: Path, name: str, formats=("png", "pdf")) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out / f"{name}.{f}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def make_all(res: Results, ds_stats: pd.DataFrame | None = None, genre_df: pd.DataFrame | None = None,
             per_seeker: pd.DataFrame | None = None) -> dict[str, Path]:
    out = res.cfg.path("output_path") / "figures"
    made: dict[str, Path] = {}

    def reg(name, paths):
        if paths:
            made[name] = paths[0]

    T = res.tables
    # 1. dataset overview
    if genre_df is not None and per_seeker is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        sns.barplot(data=genre_df.head(12), x="movies", y="genre", ax=ax[0], color=PALETTE[0])
        ax[0].set_title("Item genre coverage (MovieLens join)")
        sns.histplot(per_seeker.dialogues, bins=30, ax=ax[1], color=PALETTE[1])
        ax[1].set_yscale("log")
        ax[1].set_title("Dialogues per seeker (cross-session availability)")
        ax[1].set_xlabel("dialogues")
        sns.histplot(res.instances_meta.history_len, bins=25, ax=ax[2], color=PALETTE[2])
        ax[2].set_title("LTP history length per test instance")
        ax[2].set_xlabel("prior liked/seen movies (capped)")
        reg("fig01_dataset_overview", _save(fig, out, "fig01_dataset_overview"))
    # 2. label distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for a, (key, title) in zip(ax, [("label_distribution_train", "Train"), ("label_distribution_test", "Test")]):
        d = T[key]
        sns.barplot(data=d, x="relationship_label", y="count", hue="relationship_source", order=RELATIONSHIPS, ax=a)
        a.set_title(f"{title}: relationship labels by source")
        a.set_xlabel("")
        a.tick_params(axis="x", rotation=20)
    reg("fig02_label_distribution", _save(fig, out, "fig02_label_distribution"))
    # 3. overall performance with CIs
    perf = [("overall_natural", "Natural test instances"), ("overall_synthetic", "Controlled synthetic test instances")]
    perf_tabs = dict(T)
    cn = T.get("conflict_natural")
    if cn is not None and len(cn) and "subset" in cn:
        for s in cn.subset.unique():
            perf_tabs[f"conflict_natural_{s}"] = cn[cn.subset == s].reset_index(drop=True)
            perf.append((f"conflict_natural_{s}", f"Natural conflict subset ({s})"))
    perf.append(("conflict_synthetic", "Synthetic Conflict/Override subset"))
    for key, title in perf:
        d = perf_tabs.get(key)
        if d is None or not len(d):
            continue
        metrics = ["Hit@10", "NDCG@10", "MRR@10", "Hit@20"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5), sharey=True)
        for a, m in zip(axes, metrics):
            y = np.arange(len(d))
            err = np.vstack([d[f"{m}_mean"] - d[f"{m}_ci_low"], d[f"{m}_ci_high"] - d[f"{m}_mean"]]).clip(min=0)
            colors = [PALETTE[3] if mo == PRIMARY else (PALETTE[7] if mo.startswith("AIPA") else PALETTE[0]) for mo in d.model]
            a.barh(y, d[f"{m}_mean"], xerr=err, color=colors, capsize=3)
            a.set_yticks(y)
            a.set_yticklabels(d.model)
            a.invert_yaxis()
            a.set_title(m)
        fig.suptitle(f"{title} (n={int(d.n.iloc[0])}; bars = mean over seeds, whiskers = 95% bootstrap CI)")
        reg(f"fig03_{key}", _save(fig, out, f"fig03_{key}"))
    # 4. conflict vs non-conflict
    if len(T.get("conflict_natural", [])) and len(T.get("nonconflict_natural", [])):
        cn = T["conflict_natural"]
        d = pd.DataFrame({"Non-disagreement (natural)": T["nonconflict_natural"].set_index("model")["Hit@10_mean"]})
        for s, lbl in [("strict", "Conflict (natural, strict weak label)"), ("broad", "Disagreement (natural, broad)")]:
            if "subset" in cn and (cn.subset == s).any():
                d[lbl] = cn[cn.subset == s].set_index("model")["Hit@10_mean"]
        if len(T.get("conflict_synthetic", [])):
            d["Conflict/Override (synthetic, controlled)"] = T["conflict_synthetic"].set_index("model")["Hit@10_mean"]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        d.loc[[m for m in MODEL_ORDER if m in d.index]].plot.bar(ax=ax, width=0.8, color=PALETTE[:len(d.columns)])
        ax.set_ylabel("Hit@10")
        ax.set_title("Conflict-sensitive evaluation: Hit@10 by subset")
        ax.tick_params(axis="x", rotation=30)
        reg("fig04_conflict_vs_nonconflict", _save(fig, out, "fig04_conflict_vs_nonconflict"))
    # 5. per-relationship subset heatmap
    rows = []
    for r in RELATIONSHIPS:
        for src in ["natural", "synthetic"]:
            d = T.get(f"subset_{r}_{src}")
            if d is not None and len(d):
                for _, row in d.iterrows():
                    rows.append({"model": row.model, "subset": f"{r}\n({src}, n={int(row.n)})", "Hit@10": row["Hit@10_mean"]})
    if rows:
        piv = pd.DataFrame(rows).pivot(index="model", columns="subset", values="Hit@10").loc[[m for m in MODEL_ORDER if m in set(pd.DataFrame(rows).model)]]
        fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 6))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Hit@10 per relationship subset")
        reg("fig05_relationship_subsets", _save(fig, out, "fig05_relationship_subsets"))
    # 6. relationship confusion matrix (primary, natural + synthetic)
    conf = res.extra.get("confusion", {})
    seed0 = res.cfg.seeds[0]
    keys = [(PRIMARY, seed0, "natural"), (PRIMARY, seed0, "synthetic")]
    keys = [k for k in keys if k in conf]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
        axes = np.atleast_1d(axes)
        for a, k in zip(axes, keys):
            cm = conf[k]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=RELATIONSHIPS, yticklabels=RELATIONSHIPS, ax=a)
            a.set_xlabel("predicted")
            a.set_ylabel("reference label")
            a.set_title(f"Relationship confusion: {k[2]} (seed {k[1]})")
        reg("fig06_relationship_confusion", _save(fig, out, "fig06_relationship_confusion"))
    # 7. arbitration actions by relationship
    if len(T.get("action_by_relationship", [])):
        d = T["action_by_relationship"].set_index("relationship_label")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        d.loc[[r for r in RELATIONSHIPS if r in d.index]].plot.bar(stacked=True, ax=ax, color=PALETTE[:4], width=0.7)
        ax.set_ylabel("share of instances")
        ax.set_title(f"{PRIMARY}: arbitration action by reference relationship (test)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="action", bbox_to_anchor=(1.02, 1), loc="upper left")
        reg("fig07_actions_by_relationship", _save(fig, out, "fig07_actions_by_relationship"))
    # 8. counterfactual deltas
    cf = res.extra.get("counterfactual_detail")
    if cf is not None and len(cf):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        sns.histplot(data=cf, x="delta_ndcg_LTP", hue="is_synthetic", bins=30, ax=ax[0], element="step", stat="density", common_norm=False)
        ax[0].set_title("Δ NDCG@10 when LTP is neutralised")
        sns.histplot(data=cf, x="delta_ndcg_STI", hue="is_synthetic", bins=30, ax=ax[1], element="step", stat="density", common_norm=False)
        ax[1].set_title("Δ NDCG@10 when STI is neutralised")
        drv = cf.groupby(["relationship_label", "driver"]).size().unstack(fill_value=0)
        drv = drv.div(drv.sum(1), axis=0).loc[[r for r in RELATIONSHIPS if r in drv.index]]
        drv.plot.bar(stacked=True, ax=ax[2], color=PALETTE[:4], width=0.7)
        ax[2].set_title("Counterfactual driver label by relationship")
        ax[2].set_ylabel("share")
        ax[2].tick_params(axis="x", rotation=20)
        ax[2].legend(fontsize=8)
        fig.suptitle("Model-based counterfactual driver diagnostic (interventions on the trained model; not causal effects)")
        reg("fig08_counterfactual", _save(fig, out, "fig08_counterfactual"))
    # 9. sensitivity
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    d = T.get("sens_history")
    if d is not None and len(d):
        ax[0].plot(d.history_bucket.astype(str), d["Hit@10"], marker="o", color=PALETTE[0], label="Hit@10")
        ax[0].plot(d.history_bucket.astype(str), d["NDCG@10"], marker="s", color=PALETTE[1], label="NDCG@10")
        for x, n in zip(d.history_bucket.astype(str), d.n):
            ax[0].annotate(f"n={n}", (x, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax[0].set_title(f"{PRIMARY}: LTP history-length sensitivity")
        ax[0].set_xlabel("prior liked/seen movies")
        ax[0].legend()
    d = T.get("sens_sti_length")
    if d is not None and len(d):
        ax[1].plot(d.sti_bucket.astype(str), d["Hit@10"], marker="o", color=PALETTE[0], label="Hit@10")
        ax[1].plot(d.sti_bucket.astype(str), d["NDCG@10"], marker="s", color=PALETTE[1], label="NDCG@10")
        for x, n in zip(d.sti_bucket.astype(str), d.n):
            ax[1].annotate(f"n={n}", (x, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax[1].set_title(f"{PRIMARY}: STI context-length sensitivity")
        ax[1].set_xlabel("seeker turns before recommendation")
        ax[1].legend()
    d = T.get("sens_intensity")
    if d is not None and len(d):
        for i, m in enumerate([m for m in ["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion", PRIMARY] if m in set(d.model)]):
            dd = d[d.model == m]
            ax[2].plot(dd.intensity, dd["Hit@10"], marker="o", label=m, color=PALETTE[i])
        ax[2].set_title("Synthetic conflict intensity (Conflict/Override)")
        ax[2].set_xlabel("injection intensity")
        ax[2].set_ylabel("Hit@10 (synthetic target)")
        ax[2].set_xticks(sorted(d.intensity.unique()))
        ax[2].legend(fontsize=8)
    reg("fig09_sensitivity", _save(fig, out, "fig09_sensitivity"))
    # 9b/9c. per-history-length bucket and per-target-genre breakdowns (Hit@10, every model, mean +- std over seeds)
    for key, col, title, xlabel, fname in [
        ("history_buckets", "history_bucket", "Hit@10 by LTP history length (natural test)", "history bucket (prior liked/seen movies)", "fig09b_history_buckets"),
        ("genre_breakdown", "target_genre", "Hit@10 by target genre (natural test, top genres)", "target genre", "fig09c_genre_breakdown"),
    ]:
        d = T.get(key)
        if d is None or not len(d):
            continue
        models = [m for m in MODEL_ORDER if m in set(d.model)]
        cats = [str(c) for c in d[col].astype(str).unique()]
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(cats) + 4), 4.8))
        width = 0.8 / max(1, len(models))
        xs = np.arange(len(cats))
        for i, m in enumerate(models):
            dd = d[d.model == m].set_index(d[d.model == m][col].astype(str)).reindex(cats)
            color = PALETTE[3] if m == PRIMARY else (PALETTE[7] if m.startswith("AIPA") else PALETTE[i % 3])
            ax.bar(xs + (i - len(models) / 2 + 0.5) * width, dd["Hit@10_mean"], width, yerr=dd["Hit@10_std"], label=m, color=color, capsize=2,
                   alpha=0.5 if (m.startswith("AIPA") and m != PRIMARY) else 1.0)
        ns = d[d.model == models[0]].set_index(d[d.model == models[0]][col].astype(str)).reindex(cats).n
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c}\n(n={int(n)})" if not np.isnan(n) else c for c, n in zip(cats, ns)])
        ax.set_ylabel("Hit@10 (mean ± std over seeds)")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        reg(fname, _save(fig, out, fname))
    # 10. alpha sweep
    d = T.get("alpha_sweep")
    if d is not None and len(d):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d.alpha_ltp, d["Hit@10"], marker="o", label="Hit@10 (natural)")
        ax.plot(d.alpha_ltp, d["NDCG@10"], marker="s", label="NDCG@10 (natural)")
        if "Hit@10_synthetic" in d:
            ax.plot(d.alpha_ltp, d["Hit@10_synthetic"], marker="^", label="Hit@10 (synthetic)")
        ax.set_xlabel("fixed LTP weight α (STI weight = 1-α)")
        ax.set_title("Fixed fusion weight sweep (Naive-fusion encoders)")
        ax.legend()
        reg("fig10_alpha_sweep", _save(fig, out, "fig10_alpha_sweep"))
    # 11. calibration
    bins = res.extra.get("calibration_bins", {})
    k = (PRIMARY, seed0, "all")
    if k in bins and len(bins[k]):
        b = bins[k]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(b.confidence, b.accuracy, marker="o", color=PALETTE[0])
        for _, r in b.iterrows():
            ax.annotate(int(r["count"]), (r.confidence, r.accuracy), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ece = T["calibration"].query("model == @PRIMARY and seed == @seed0 and subset == 'all'").ECE.iloc[0]
        ax.set_title(f"Relationship classifier reliability (ECE={ece:.3f})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        reg("fig11_calibration", _save(fig, out, "fig11_calibration"))
    # 12. training curves
    tc = res.training_curves
    if len(tc):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, m in enumerate(MODEL_ORDER):
            d = tc[(tc.model == m) & (tc.seed == seed0)]
            if len(d):
                ax[0].plot(d.epoch, d.rec, label=m, color=PALETTE[i % len(PALETTE)], ls="-" if m.startswith("AIPA") else "--")
                ax[1].plot(d.epoch, d["valid_hit@10"], label=m, color=PALETTE[i % len(PALETTE)], ls="-" if m.startswith("AIPA") else "--")
        ax[0].set_title("Recommendation loss (train)")
        ax[1].set_title("Validation Hit@10")
        ax[0].set_xlabel("epoch")
        ax[1].set_xlabel("epoch")
        ax[1].legend(fontsize=7, ncol=2)
        reg("fig12_training_curves", _save(fig, out, "fig12_training_curves"))
    # 13. efficiency
    d = T.get("efficiency")
    if d is not None and len(d):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(data=d, y="model", x="train_time_s", ax=ax[0], color=PALETTE[0])
        ax[0].set_title("Training time (s, mean over seeds)")
        sns.barplot(data=d, y="model", x="cpu_inference_ms_per_sample", ax=ax[1], color=PALETTE[1])
        ax[1].set_title("Inference time per test instance (ms)")
        reg("fig13_efficiency", _save(fig, out, "fig13_efficiency"))
    # 14. architecture diagram
    reg("fig00_architecture", _save(architecture_diagram(), out, "fig00_architecture"))
    return made


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")
    boxes = {
        "hist": (0.02, 0.70, 0.18, 0.18, "Cross-session history\n(earlier ReDial sessions:\nliked/seen movies, preference statements)"),
        "ctx": (0.02, 0.12, 0.18, 0.18, "Current dialogue context\n(recent seeker turns, in-dialogue items,\ngenre cues, lexical markers)"),
        "ltp": (0.26, 0.70, 0.14, 0.18, "LTP encoder\nh_LTP"),
        "sti": (0.26, 0.12, 0.14, 0.18, "STI encoder\nh_STI"),
        "rel": (0.46, 0.55, 0.16, 0.18, "Relationship classifier\nConsistent / Complement /\nConflict / Override / Uncertain"),
        "cf": (0.46, 0.27, 0.16, 0.18, "Counterfactual driver\ndiagnostic (mask LTP / STI)\nΔ top-K, driver label"),
        "arb": (0.68, 0.41, 0.14, 0.18, "Arbitration policy\nFuse / Prioritize_LTP /\nPrioritize_STI / Ask"),
        "rec": (0.86, 0.60, 0.12, 0.16, "Fused ranking\nw_LTP·s_LTP + w_STI·s_STI"),
        "clar": (0.86, 0.22, 0.12, 0.16, "Clarification\nquestion (English)"),
        "per": (0.68, 0.08, 0.14, 0.14, "Persistence tracker\n(temporary override vs\npersistent shift)"),
    }
    for k, (x, y, w, h, t) in boxes.items():
        color = "#dbe9f6" if k in ("hist", "ctx") else ("#fde2c8" if k in ("rel", "cf", "arb") else "#e3f2e1")
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="black", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

    def arrow(a, b, dy_a=0.5, dy_b=0.5):
        xa, ya, wa, ha, _ = boxes[a]
        xb, yb, wb, hb, _ = boxes[b]
        ax.annotate("", xy=(xb, yb + hb * dy_b), xytext=(xa + wa, ya + ha * dy_a), arrowprops=dict(arrowstyle="->", lw=1.2))

    for args in [("hist", "ltp"), ("ctx", "sti"), ("ltp", "rel", 0.5, 0.8), ("sti", "rel", 0.5, 0.2),
                 ("ltp", "cf", 0.3, 0.8), ("sti", "cf", 0.7, 0.2), ("rel", "arb", 0.5, 0.8), ("cf", "arb", 0.5, 0.2),
                 ("arb", "rec", 0.7, 0.5), ("arb", "clar", 0.3, 0.5), ("per", "arb", 0.5, 0.1)]:
        arrow(*args)
    ax.set_title("AIPA-CRS architecture (this implementation)", fontsize=12)
    return fig
