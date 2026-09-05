"""Publication-quality figures (matplotlib / seaborn), generated from Results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from . import ACTIONS, RELATIONSHIPS  # noqa: E402
from .config import Config, load_config  # noqa: E402
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
    # 14. architecture diagram (two-column wide + single-column compact variant)
    reg("fig00_architecture", _save(architecture_diagram(cfg=res.cfg), out, "fig00_architecture"))
    reg("fig00_architecture_compact",
        _save(architecture_diagram(compact=True, cfg=res.cfg), out, "fig00_architecture_compact"))
    return made


# --------------------------------------------------------------------------
# architecture diagram
# --------------------------------------------------------------------------
# Drawing coordinates are hundredths of an inch, so one unit = 0.01 in and all
# font sizes below are the sizes the reader sees at the stated figure width.

ARCH_STYLE = {
    "inputs": {"box": "#dce9f7", "panel": "#f2f7fd", "edge": "#3f78ad", "text": "#123a5e"},
    "encoders": {"box": "#dff0dd", "panel": "#f3faf2", "edge": "#4e8f57", "text": "#1d4a25"},
    "arbitration": {"box": "#fde3c9", "panel": "#fef7f0", "edge": "#c07f34", "text": "#7a4708"},
    "output": {"box": "#e7e0f4", "panel": "#f7f4fc", "edge": "#7d66b0", "text": "#3b2a63"},
    "training": {"box": "#e9e9e9", "panel": "#f6f6f6", "edge": "#7d7d7d", "text": "#2f2f2f"},
}
LTP_COLOR = "#1f6fb2"
STI_COLOR = "#b8761c"
FLOW_COLOR = "#4a4a4a"
PERSIST_COLOR = "#6b4fa1"
PT = 100.0 / 72.0  # drawing units per typographic point


def _arch_canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    return fig, ax


def _arch_panel(ax, rect, group, number, title, fs=6.0):
    x, y, w, h = rect
    st = ARCH_STYLE[group]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5",
                                fc=st["panel"], ec=st["edge"], lw=0.7, zorder=0))
    r = fs * PT * 0.72
    cx, cy = x + 4 + r, y + h - 4 - r
    ax.add_patch(plt.Circle((cx, cy), r, fc=st["edge"], ec="none", zorder=3))
    ax.text(cx, cy, str(number), ha="center", va="center", color="white", fontsize=fs - 1.0,
            fontweight="bold", zorder=4)
    ax.text(cx + r + 3, cy, title, ha="left", va="center", fontsize=fs, fontweight="bold",
            color=st["text"], zorder=3)


def _arch_box(ax, rect, group, title, body="", shape="", fs=(5.6, 4.8, 4.3), pad=3.0):
    x, y, w, h = rect
    st = ARCH_STYLE[group]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3",
                                fc=st["box"], ec=st["edge"], lw=0.6, zorder=2))
    title_h = title.count("\n") + 1
    top = y + h - pad
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs[0], fontweight="bold",
            color=st["text"], linespacing=1.15, zorder=3)
    bottom = y + pad
    if shape:
        ax.text(x + w / 2, bottom, shape, ha="center", va="bottom", fontsize=fs[2], style="italic",
                color="#33383d", zorder=3)
        bottom += (shape.count("\n") + 1) * fs[2] * PT * 1.2
    if body:
        top -= title_h * fs[0] * PT * 1.25
        ax.text(x + w / 2, (top + bottom) / 2, body, ha="center", va="center", fontsize=fs[1],
                color="#1c1c1c", linespacing=1.3, zorder=3)


def _arch_arrow(ax, points, color=FLOW_COLOR, ls="-", lw=0.7, rad=0.0, zorder=5, head=1.0):
    style = f"-|>,head_length={0.55 * head},head_width={0.32 * head}"
    if len(points) == 2:
        arrow = FancyArrowPatch(points[0], points[1], arrowstyle=style, mutation_scale=10,
                                connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, zorder=zorder)
    else:
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        arrow = FancyArrowPatch(path=path, arrowstyle=style, mutation_scale=10, color=color,
                                lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(arrow)
    return arrow


def _arch_label(ax, x, y, text, fs=4.2, color=FLOW_COLOR, ha="center", va="center", rotation=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, rotation=rotation, zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def _arch_dims(cfg: Config | None = None) -> dict[str, object]:
    """Shapes and hyper-parameters shown in the figure.

    Tensor widths come from the implementation (``models.py`` / ``preprocess.py``)
    and everything tunable is read from ``cfg``; without one the repository
    configuration is loaded, so the figure never states dimensions that the run
    it accompanies did not use.
    """
    from .models import ACTION_WEIGHTS, N_FLAGS, N_GENRES

    cfg = cfg or load_config()
    d = int(cfg.hidden_dim)
    return {"d": d, "g": N_GENRES, "f": N_FLAGS, "rel_in": 4 * d + 2 * N_GENRES + N_FLAGS + 1,
            "act_in": 2 * d + 2 * N_GENRES + N_FLAGS + 1 + len(RELATIONSHIPS) + 5,
            "ltp_in": 3 * d, "sti_in": 5 * d,
            "max_history": int(cfg.max_history), "max_turns": int(cfg.max_context_turns),
            "cur_items": 10, "persistence_k": int(cfg.persistence_k),
            "top_k": ", ".join(str(k) for k in cfg.top_k),
            "lambda_rel": f"{float(cfg.lambda_rel):g}", "lambda_act": f"{float(cfg.lambda_act):g}",
            "action_weights": ", ".join(f"({w[0]:.2f}, {w[1]:.2f})".replace("0.", ".")
                                        for w in ACTION_WEIGHTS.tolist())}


def _architecture_wide(cfg: Config | None = None):
    """Two-column (``figure*``) variant: 7.16 in x 4.70 in, aspect ratio 1.52 : 1."""
    D = _arch_dims(cfg)
    d, g, f = D["d"], D["g"], D["f"]
    fig, ax = _arch_canvas(716, 470)
    fs = (5.4, 4.7, 4.2)
    dash_score = (0, (2.2, 1.4))
    dash_share = (0, (2.0, 1.6))
    dot_feedback = (0, (1.2, 1.4))

    _arch_panel(ax, (6, 74, 154, 374), "inputs", 1, "Inputs (one instance)")
    _arch_panel(ax, (168, 74, 132, 374), "encoders", 2, "Encoders")
    _arch_panel(ax, (324, 74, 260, 374), "arbitration", 3, "Arbitration")
    _arch_panel(ax, (596, 74, 114, 374), "output", 4, "Output")
    _arch_panel(ax, (6, 6, 704, 54), "training", 5, "Training objective")

    for rect, label in [((9, 302, 6, 128), "LTP (earlier sessions)"), ((9, 80, 6, 198), "STI (current dialogue)")]:
        bx, by, bw, bh = rect
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=2",
                                    fc=ARCH_STYLE["inputs"]["edge"], ec="none", zorder=2))
        ax.text(bx + bw / 2, by + bh / 2, label, rotation=90, ha="center", va="center",
                fontsize=4.0, color="white", zorder=3)

    _arch_box(ax, (18, 390, 138, 40), "inputs", "Cross-session liked-item history",
              "positional embedding, attention pooling",
              f"ids [B, {D['max_history']}] -> h_hist [B, {d}]", fs)
    _arch_box(ax, (18, 348, 138, 38), "inputs", "Seeker profile text",
              "preference statements from\nearlier sessions", "x_prof [B, d_t]", fs)
    _arch_box(ax, (18, 302, 138, 42), "inputs", "LTP genre prior",
              "genres of past liked items,\nrecency-decayed (gamma^delta)", f"g_LTP [B, {g}]", fs)
    _arch_box(ax, (18, 244, 138, 34), "inputs", "Dialogue context",
              f"up to {D['max_turns']} recent seeker turns", "x_ctx [B, d_t]", fs)
    _arch_box(ax, (18, 206, 138, 34), "inputs", "Last seeker utterance", "", "x_last [B, d_t]", fs)
    _arch_box(ax, (18, 168, 138, 34), "inputs", "In-dialogue liked items",
              "masked mean of item embeddings", f"ids [B, {D['cur_items']}]", fs)
    _arch_box(ax, (18, 130, 138, 34), "inputs", "STI genre cues",
              "genres named in this dialogue", f"g_STI [B, {g}]", fs)
    _arch_box(ax, (18, 80, 138, 46), "inputs", "Lexical flags",
              "override_sti, override_ltp, negation,\nrequest, cold_user, turn / item\ncounts, history length",
              f"phi_flag [B, {f}]", fs)

    _arch_box(ax, (174, 366, 122, 64), "encoders", "LTPEncoder",
              f"concat(h_hist, W x_prof, W g_LTP)\n-> MLP({D['ltp_in']} -> {d} -> {d});\nzeroed without past evidence",
              f"h_LTP [B, {d}]", fs)
    _arch_box(ax, (174, 324, 122, 34), "encoders", "LTP item scores",
              "s_LTP = h_LTP E^T + b", "[B, N+1]", fs)
    _arch_box(ax, (174, 224, 122, 84), "encoders", "Shared ItemTower",
              "E = E_id + W_c [title ; genres],\nitem bias b; used for the\nhistory and in-dialogue id\nlookups and for both scorings",
              f"E [N+1, {d}],  b [N+1]", fs)
    _arch_box(ax, (174, 126, 122, 64), "encoders", "STIEncoder",
              f"concat(W x_ctx, W x_last, W g_STI,\nW phi_flag, mean item emb.)\n-> MLP({D['sti_in']} -> {d} -> {d})",
              f"h_STI [B, {d}]", fs)
    _arch_box(ax, (174, 84, 122, 34), "encoders", "STI item scores",
              "s_STI = h_STI E^T + b", "[B, N+1]", fs)

    _arch_box(ax, (338, 366, 236, 64), "arbitration", "3a  RelationshipClassifier",
              "[h_LTP ; h_STI ; h_LTP * h_STI ; |h_LTP - h_STI| ;\n"
              f"g_LTP ; g_STI ; phi_flag ; cos]  ->  MLP({D['rel_in']} -> {d} -> {len(RELATIONSHIPS)})\n"
              + "  |  ".join(RELATIONSHIPS),
              f"p_rel [B, {len(RELATIONSHIPS)}];  confidence-weighted CE against weak-rule /\nsynthetic labels (weak rules are not human annotations)", fs)
    _arch_box(ax, (338, 268, 236, 76), "arbitration", "3b  CounterfactualDiagnostic",
              "interventions on the naive-fusion scores (s_LTP + s_STI) / 2:\n"
              "drop LTP -> s_STI,   drop STI -> s_LTP\n"
              "d_LTP, d_STI = 1 - Jaccard of the top-K lists; top-1 margins\n"
              "driver label: LTP- / STI- / Jointly- / Neither-driven",
              "phi_cf [B, 5]  (model-based diagnostic, not a causal effect)", fs)
    _arch_box(ax, (338, 168, 236, 78), "arbitration", "3c  ArbitrationPolicy",
              f"learned MLP({D['act_in']} -> {d} -> {len(ACTIONS)}) over\n"
              "[h_LTP ; h_STI ; g_LTP ; g_STI ; phi_flag ; cos ; p_rel ; phi_cf],\n"
              "or the rule policy (argmax p_rel + evidence overrides)\n"
              + "  |  ".join(ACTIONS),
              f"a [B, {len(ACTIONS)}]", fs)
    _arch_box(ax, (338, 100, 236, 52), "arbitration", "3d  Action-weighted mixing",
              f"(w_LTP, w_STI) = softmax(a) A,\nA = [{D['action_weights']}]",
              "w [B, 2];   s = w_LTP s_LTP + w_STI s_STI  [B, N+1]", fs)

    _arch_box(ax, (602, 344, 104, 86), "output", "Top-K recommendation",
              f"ranked catalogue items\n(K = {D['top_k']}; padding\ncolumn masked to -inf)", "top-K(s)", fs)
    _arch_box(ax, (602, 232, 104, 96), "output", "Clarification question",
              "English template that\ncontrasts the dominant\nLTP and STI genres;\nemitted when the action\nis Ask_Clarification", "", fs)
    _arch_box(ax, (602, 96, 104, 120), "output", "PersistenceTracker",
              "counts Prioritize_STI on\nthe same genre across\ndistinct sessions of one\n"
              f"seeker; >= k = {D['persistence_k']} sessions\nis a persistent shift,\n"
              "otherwise the override\nstays temporary", "", fs)

    for y0, y1 in [(410, 408), (367, 394), (323, 380)]:
        _arch_arrow(ax, [(156, y0), (172, y1)], LTP_COLOR, rad=0.08)
    for y0, y1 in [(261, 180), (223, 170), (185, 160), (147, 150), (103, 140)]:
        _arch_arrow(ax, [(156, y0), (172, y1)], STI_COLOR, rad=0.08)
    _arch_arrow(ax, [(235, 366), (235, 358)], LTP_COLOR)
    _arch_arrow(ax, [(235, 126), (235, 118)], STI_COLOR)
    _arch_arrow(ax, [(212, 308), (212, 324)], FLOW_COLOR, ls=dash_share, lw=0.6)
    _arch_arrow(ax, [(212, 224), (212, 118)], FLOW_COLOR, ls=dash_share, lw=0.6)

    # signal buses between the encoders and the arbitration stage
    for pts, color, ls in [
        ([(296, 398), (303, 398), (303, 412), (338, 412)], LTP_COLOR, "-"),
        ([(303, 398), (303, 200), (338, 200)], LTP_COLOR, "-"),
        ([(296, 341), (308, 341), (308, 306), (338, 306)], LTP_COLOR, dash_score),
        ([(308, 341), (308, 118), (338, 118)], LTP_COLOR, dash_score),
        ([(296, 158), (313, 158), (313, 382), (338, 382)], STI_COLOR, "-"),
        ([(313, 158), (313, 182), (338, 182)], STI_COLOR, "-"),
        ([(296, 101), (318, 101), (318, 286), (338, 286)], STI_COLOR, dash_score),
        ([(318, 101), (318, 110), (338, 110)], STI_COLOR, dash_score),
    ]:
        _arch_arrow(ax, pts, color, ls=ls, lw=0.65)
    for x, y, c in [(303, 398, LTP_COLOR), (308, 341, LTP_COLOR), (313, 158, STI_COLOR), (318, 101, STI_COLOR)]:
        ax.add_patch(plt.Circle((x, y), 1.6, fc=c, ec="none", zorder=6))

    _arch_arrow(ax, [(574, 382), (579, 382), (579, 214), (574, 214)], FLOW_COLOR, lw=0.65)
    _arch_label(ax, 579, 300, "p_rel", 4.0, rotation=90)
    _arch_arrow(ax, [(456, 268), (456, 246)], FLOW_COLOR)
    _arch_label(ax, 470, 257, "phi_cf", 4.0)
    _arch_arrow(ax, [(456, 168), (456, 152)], FLOW_COLOR)
    _arch_label(ax, 466, 160, "a", 4.0)
    _arch_arrow(ax, [(574, 126), (590, 126), (590, 388), (602, 388)], FLOW_COLOR, lw=0.8)
    _arch_label(ax, 590, 300, "s", 4.0)
    _arch_arrow(ax, [(574, 196), (602, 268)], FLOW_COLOR, ls=dash_score, lw=0.6, rad=-0.12)
    _arch_arrow(ax, [(574, 176), (602, 178)], FLOW_COLOR, ls=dash_score, lw=0.6)
    _arch_arrow(ax, [(654, 96), (654, 67), (164, 67), (164, 326), (156, 326)], PERSIST_COLOR,
                ls=dot_feedback, lw=0.7)
    _arch_label(ax, 400, 68, "persistent shift: the repeatedly prioritised genre is folded into the LTP prior (+gain)",
                4.0, PERSIST_COLOR)

    ax.text(6, 460, "AIPA-CRS: adaptive intent-preference arbitration (this implementation)",
            ha="left", va="center", fontsize=7.4, fontweight="bold", color="#1a1a1a")
    ax.text(710, 460, f"B: batch  |  N: catalogue items  |  d = {d}  |  d_t: text-embedding dimension",
            ha="right", va="center", fontsize=4.6, color="#3a3a3a")
    ax.text(210, 51, "L = L_rec + lambda_rel L_rel + lambda_act L_act      "
                     f"(lambda_rel = {D['lambda_rel']}, lambda_act = {D['lambda_act']})",
            ha="left", va="center", fontsize=5.4, fontweight="bold", color=ARCH_STYLE["training"]["text"])
    ax.text(16, 28,
            "L_rec: cross-entropy of the fused scores s against the held-out target item (stage 3d).       "
            "L_rel: confidence-weighted cross-entropy on stage 3a against the weak-rule /\n"
            "controlled-synthetic relationship labels.       "
            "L_act: confidence-weighted cross-entropy on stage 3c against the rule-derived action (learned policy only).",
            ha="left", va="center", fontsize=4.6, color="#1c1c1c", linespacing=1.4)
    ax.text(704, 12,
            "solid: hidden-state flow      dashed: item-score / diagnostic flow      dotted: cross-session feedback",
            ha="right", va="center", fontsize=4.2, color="#3a3a3a")
    return fig


def _architecture_compact(cfg: Config | None = None):
    """Single-column variant: 3.50 in x 5.16 in, aspect ratio 1 : 1.47."""
    D = _arch_dims(cfg)
    d, g, f = D["d"], D["g"], D["f"]
    fig, ax = _arch_canvas(350, 516)
    fs = (5.0, 4.5, 4.1)
    dash_score = (0, (2.2, 1.4))
    dash_share = (0, (2.0, 1.6))

    _arch_panel(ax, (4, 404, 334, 92), "inputs", 1, "Inputs", fs=5.2)
    _arch_panel(ax, (4, 308, 334, 88), "encoders", 2, "Encoders", fs=5.2)
    _arch_panel(ax, (4, 122, 334, 178), "arbitration", 3, "Arbitration", fs=5.2)
    _arch_panel(ax, (4, 48, 334, 66), "output", 4, "Output", fs=5.2)
    _arch_panel(ax, (4, 6, 334, 34), "training", 5, "Training", fs=5.2)

    _arch_box(ax, (10, 410, 158, 72), "inputs", "Cross-session (LTP)",
              "liked-item history with positional\nembedding and attention pooling;\nprofile text; recency-decayed\ngenre prior",
              f"ids [B,{D['max_history']}], x_prof [B,d_t], g_LTP [B,{g}]", fs)
    _arch_box(ax, (172, 410, 158, 72), "inputs", "Current dialogue (STI)",
              "context and last seeker utterance;\nin-dialogue liked items (masked\nmean); genre cues; lexical flags\n(override, negation, request, cold_user)",
              f"x_ctx, x_last [B,d_t], g_STI [B,{g}], phi_flag [B,{f}]", fs)

    _arch_box(ax, (10, 314, 104, 66), "encoders", "LTPEncoder",
              f"MLP({D['ltp_in']} -> {d} -> {d})", f"h_LTP [B,{d}]\ns_LTP = h_LTP E^T + b", fs)
    _arch_box(ax, (120, 314, 108, 66), "encoders", "Shared ItemTower",
              "E = E_id + W_c [title ; genres]\nwith item bias b; supplies the\nid lookups and both score\nvectors", f"E [N+1,{d}]", fs)
    _arch_box(ax, (234, 314, 96, 66), "encoders", "STIEncoder",
              f"MLP({D['sti_in']} -> {d} -> {d})", f"h_STI [B,{d}]\ns_STI = h_STI E^T + b", fs)

    _arch_box(ax, (10, 218, 158, 62), "arbitration", "3a  RelationshipClassifier",
              "[h_LTP ; h_STI ; h_LTP * h_STI ;\n|h_LTP - h_STI| ; g_LTP ; g_STI ;\nphi_flag ; cos] -> 5 classes\n"
              "Complement | Consistent | Conflict\n| Override | Uncertain", "p_rel [B,5]", fs)
    _arch_box(ax, (172, 218, 158, 62), "arbitration", "3b  CounterfactualDiagnostic",
              "drop LTP / drop STI on the\nnaive-fusion scores; d_LTP, d_STI\n= 1 - top-K Jaccard, plus margins\n-> LTP- / STI- / Jointly- /\nNeither-driven", "phi_cf [B,5]", fs)
    _arch_box(ax, (10, 166, 320, 46), "arbitration", "3c  ArbitrationPolicy (learned MLP or rule policy)",
              "Fuse  |  Prioritize_LTP  |  Prioritize_STI  |  Ask_Clarification", "a [B,4]", fs)
    _arch_box(ax, (10, 128, 320, 32), "arbitration", "3d  Action-weighted mixing",
              "(w_LTP, w_STI) = softmax(a) A;   s = w_LTP s_LTP + w_STI s_STI", "", fs)

    _arch_box(ax, (10, 54, 158, 42), "output", f"Top-K list (K = {D['top_k']})",
              "plus the template clarification\nquestion when the action is\nAsk_Clarification", "", fs)
    _arch_box(ax, (172, 54, 158, 42), "output", "PersistenceTracker",
              f"Prioritize_STI on the same genre in\n>= k = {D['persistence_k']} sessions is folded into the\n"
              "LTP genre prior (persistent shift)", "", fs)

    ax.text(58, 22, "L = L_rec + lambda_rel L_rel + lambda_act L_act   "
                    f"({D['lambda_rel']} / {D['lambda_act']}): item cross-entropy, plus confidence-weighted\n"
                    "cross-entropy on the relationship (weak-rule / synthetic labels) and on the rule-derived action",
            ha="left", va="center", fontsize=4.4, color="#1c1c1c", linespacing=1.4)

    _arch_arrow(ax, [(100, 410), (100, 380)], LTP_COLOR, rad=0.05)
    _arch_arrow(ax, [(282, 410), (282, 380)], STI_COLOR, rad=0.05)
    _arch_arrow(ax, [(120, 347), (114, 347)], FLOW_COLOR, ls=dash_share, lw=0.6)
    _arch_arrow(ax, [(228, 347), (234, 347)], FLOW_COLOR, ls=dash_share, lw=0.6)
    _arch_arrow(ax, [(100, 314), (100, 280)], LTP_COLOR)
    _arch_arrow(ax, [(282, 314), (282, 280)], STI_COLOR)
    _arch_arrow(ax, [(100, 218), (100, 212)], FLOW_COLOR)
    _arch_arrow(ax, [(251, 218), (251, 212)], FLOW_COLOR)
    _arch_arrow(ax, [(170, 166), (170, 160)], FLOW_COLOR)
    _arch_arrow(ax, [(100, 128), (100, 96)], FLOW_COLOR)
    _arch_arrow(ax, [(251, 128), (251, 96)], FLOW_COLOR, ls=dash_score, lw=0.6)
    _arch_arrow(ax, [(330, 75), (344, 75), (344, 490), (100, 490), (100, 482)], PERSIST_COLOR,
                ls=(0, (1.2, 1.4)), lw=0.7)
    _arch_label(ax, 300, 106, "persistent shift", 4.0, PERSIST_COLOR)

    ax.text(4, 508, "AIPA-CRS architecture (this implementation)", ha="left", va="center",
            fontsize=6.4, fontweight="bold", color="#1a1a1a")
    ax.text(346, 508, f"B: batch | N: items | d = {d}", ha="right", va="center", fontsize=4.2,
            color="#3a3a3a")
    return fig


def architecture_diagram(compact: bool = False, cfg: Config | None = None):
    """Architecture overview of the implemented model.

    ``compact=False`` returns the wide two-column (``figure*``) version,
    ``compact=True`` the single-column version.  Dimensions and hyper-parameters
    are read from ``cfg`` (the repository configuration when it is omitted).  The
    figure is returned, not saved; use :func:`save_architecture_diagrams` (or
    :func:`make_all`) to write ``fig00_architecture{,_compact}.{png,pdf}``.
    """
    return _architecture_compact(cfg) if compact else _architecture_wide(cfg)


def save_architecture_diagrams(out: str | Path = "outputs/figures",
                               cfg: Config | None = None) -> dict[str, Path]:
    out = Path(out)
    return {name: _save(architecture_diagram(compact=c, cfg=cfg), out, name)[0]
            for name, c in [("fig00_architecture", False), ("fig00_architecture_compact", True)]}
