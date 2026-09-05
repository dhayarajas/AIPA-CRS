"""Automatic Markdown + HTML experimental report generated from Results.

Every number in the report is read from the result tables; hypothesis verdicts
are computed from the statistical tests with fixed decision rules stated in the
text.  Nothing is typed by hand."""
from __future__ import annotations

from pathlib import Path

import markdown
import numpy as np
import pandas as pd

from . import RELATIONSHIPS
from .config import environment_report
from .experiments import PRIMARY, Results
from .models import BASELINE_NAMES

ALPHA = 0.05


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def _md_table(df: pd.DataFrame, nd: int = 3, max_rows: int = 60) -> str:
    if df is None or not len(df):
        return "_NOT RUN / no data for this table._\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: _fmt(v, nd))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v).replace("\n", " ").replace("|", "/") for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def _perf_table(d: pd.DataFrame, metrics=("Hit@10", "NDCG@10", "MRR@10", "Hit@20", "NDCG@20")) -> pd.DataFrame:
    if d is None or not len(d):
        return pd.DataFrame()
    out = pd.DataFrame({"model": d.model, "n": d.n})
    if "subset" in d:
        out.insert(0, "subset", d.subset.values)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {s:.3f} [{lo:.3f}, {hi:.3f}]" for a, s, lo, hi in
                  zip(d[f"{m}_mean"], d[f"{m}_std"], d[f"{m}_ci_low"], d[f"{m}_ci_high"])]
    return out


SIG_COLS = ["control", "n", "n_samples", "n_seeds", "mean_diff", "seed_std_diff", "t_p", "t_p_holm", "wilcoxon_p", "wilcoxon_p_holm",
            "perm_p", "perm_p_holm", "cohen_d", "cliffs_delta"]
CONFLICT_SUBSETS = ["conflict_natural_strict", "conflict_natural_broad", "conflict_synthetic"]


def _verdict(sig: pd.DataFrame, subset: str, metric: str, control: str) -> tuple[str, dict | None]:
    if sig is None or not len(sig):
        return "NOT RUN", None
    r = sig[(sig.subset == subset) & (sig.metric == metric) & (sig.control == control)]
    if not len(r):
        return "NOT RUN", None
    r = r.iloc[0]
    p = r.wilcoxon_p_holm if not np.isnan(r.wilcoxon_p_holm) else r.t_p_holm
    if np.isnan(p):
        return "inconclusive (no variance between systems)", r.to_dict()
    if p < ALPHA and r.mean_diff > 0:
        return "SUPPORTED", r.to_dict()
    if p < ALPHA and r.mean_diff < 0:
        return "CONTRADICTED", r.to_dict()
    return "NOT SUPPORTED (difference not significant)", r.to_dict()


def build_report(res: Results, figures: dict[str, Path], validation: pd.DataFrame | None = None,
                 ds_stats: pd.DataFrame | None = None) -> tuple[Path, Path]:
    cfg, T = res.cfg, res.tables
    out_dir = cfg.path("output_path") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = lambda k: f"../figures/{figures[k].name}" if k in figures else None  # noqa: E731
    env = environment_report()
    sig = T.get("significance", pd.DataFrame())
    n_nat = int((~res.labels.is_synthetic).sum())
    n_syn = int(res.labels.is_synthetic.sum())
    md: list[str] = []
    A = md.append
    A("# AIPA-CRS: Experimental Report\n")
    A(f"_Automatically generated. Run mode: **{cfg.run_mode}**; seeds: {cfg.seeds}; generated {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC._\n")
    A("> **Scope statement.** This is a research prototype evaluated on ReDial with derived (weak-rule) and controlled "
      "synthetic relationship labels. ReDial carries no native intent/preference relationship annotation; no human-verified "
      "labels were available for this run unless stated in Section 2. Counterfactual analyses are model-based "
      "interventions and must not be read as causal effects in the population. Approximate baselines are re-implementations, "
      "not reproductions of MRGE, DiffLSRec or any other published system; SASRec and KBRD-style are re-implementations on "
      "ReDial inputs (KBRD-style without the external knowledge graph), not the original code.\n")
    A("## 1. Research question and hypotheses\n")
    A("**RQ.** Does explicit intent-preference arbitration help specifically when current short-term intent (STI) conflicts with "
      "historical long-term preference (LTP)?\n")
    A("* **H1** - AIPA (full) improves ranking quality over the baselines (LTP-only, STI-only, fusion, GRU, conversation-aware, "
      "SASRec, KBRD-style) on the overall natural test set.\n"
      "* **H2** - The gain is concentrated on Conflict/Override instances (natural weak-labelled subset and controlled synthetic subset).\n"
      "* **H3** - Removing the relationship classifier, the counterfactual diagnostic, clarification or temporal persistence degrades performance.\n"
      "* **H4** - The relationship classifier recovers reference labels above chance and is reasonably calibrated.\n")
    A("Decision rule: a hypothesis is *supported* when the paired Wilcoxon test (Holm-corrected within a table) on per-instance "
      f"Hit@10 gives p < {ALPHA} in the hypothesised direction; *contradicted* when p < {ALPHA} in the opposite direction; otherwise "
      "*not supported*. Where a comparison could not be computed it is reported as NOT RUN. Paired differences are formed per "
      "instance *within* each seed and pooled over seeds (n = instances x seeds); a paired t-test and a sign-flip permutation test "
      "are reported alongside Wilcoxon, together with Cohen's d and Cliff's delta.\n")
    A("### Success criteria (computed automatically from `outputs/results/success_criteria.csv`)\n")
    A(_md_table(T.get("success_criteria"), nd=4))
    A("## 2. Data, labels and preprocessing\n")
    A(f"Dataset: **ReDial** (English conversational movie recommendation), source `{res.extra.get('dataset_source')}`. "
      "Item genres are joined from MovieLens `ml-latest` by normalised title + year (items without a match have empty genre lists).\n")
    if ds_stats is not None and len(ds_stats):
        A(_md_table(ds_stats.reset_index().rename(columns={"index": "statistic"}), nd=2))
    A(f"Instances: train={res.extra.get('n_train')}, valid={res.extra.get('n_valid')}, test natural={n_nat}, test synthetic={n_syn}. "
      "One instance = one new movie recommended by the recommender; LTP uses only the seeker's *earlier* sessions "
      "(lower `conversationId`), STI uses only earlier turns of the same session.\n")
    A("**Relationship label sources** (test set):\n")
    A(_md_table(T["label_distribution_test"]))
    A(f"Human-verified labels: {res.status.get('human_verified_labels')}.\n")
    if fig_rel("fig02_label_distribution"):
        A(f"![label distribution]({fig_rel('fig02_label_distribution')})\n")
    A("## 3. Overall performance (natural test instances)\n")
    A("Values are mean ± std over seeds with a 95% bootstrap CI over instances (pooled over seeds) in brackets. "
      "Recall@K equals Hit@K because each instance has exactly one target.\n")
    A(_md_table(_perf_table(T["overall_natural"])))
    if fig_rel("fig03_overall_natural"):
        A(f"![overall]({fig_rel('fig03_overall_natural')})\n")
    A("### Paired significance vs. baselines (natural, Hit@10)\n")
    s = sig[(sig.subset == "natural") & (sig.metric == "Hit@10")] if len(sig) else pd.DataFrame()
    A(_md_table(s[SIG_COLS] if len(s) else s, nd=4))
    A("## 4. Conflict-sensitive evaluation\n")
    A("Two natural subsets are evaluated: **strict** = weak-rule label in " + "/".join(cfg.conflict_strict_labels) +
      f"; **broad** (disagreement) = strict OR (weak-rule confidence >= {cfg.disagreement_conf_min} AND Jensen-Shannon divergence "
      f"between the LTP and STI genre distributions >= {cfg.disagreement_js_min}). Both rely on weak-rule labels, not human labels.\n")
    A(_md_table(T.get("conflict_subset_sizes")))
    A("### 4.1 Natural conflict subsets (weak-rule labels; noisy)\n")
    A(_md_table(_perf_table(T.get("conflict_natural"))))
    A("### 4.2 Natural non-disagreement subset (complement of the broad subset)\n")
    A(_md_table(_perf_table(T.get("nonconflict_natural"))))
    A("### 4.3 Controlled synthetic Conflict/Override subset\n")
    A("Targets on this subset are *sampled* items that match the injected intent; the numbers measure whether a system follows a "
      "clearly expressed short-term intent, not accuracy on human recommendations.\n")
    A(_md_table(_perf_table(T.get("conflict_synthetic"))))
    if fig_rel("fig04_conflict_vs_nonconflict"):
        A(f"![conflict]({fig_rel('fig04_conflict_vs_nonconflict')})\n")
    if fig_rel("fig05_relationship_subsets"):
        A(f"![subsets]({fig_rel('fig05_relationship_subsets')})\n")
    A("### 4.4 Synthetic conflict subset by injection intensity (mean ± std over seeds)\n")
    cbi = T.get("conflict_synthetic_by_intensity")
    if cbi is not None and len(cbi):
        A(_md_table(cbi[["intensity", "model", "n", "seeds", "Hit@10_mean", "Hit@10_std", "NDCG@10_mean", "NDCG@10_std"]]))
    A("### 4.5 Paired significance on conflict subsets (Hit@10)\n")
    s = sig[sig.subset.isin(CONFLICT_SUBSETS) & (sig.metric == "Hit@10")] if len(sig) else pd.DataFrame()
    A(_md_table(s[["subset"] + SIG_COLS] if len(s) else s, nd=4))
    A("## 5. Relationship classification, arbitration and clarification\n")
    rel = T.get("relationship", pd.DataFrame())
    if len(rel):
        A(_md_table(rel.groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").reset_index()))
    if fig_rel("fig06_relationship_confusion"):
        A(f"![confusion]({fig_rel('fig06_relationship_confusion')})\n")
    arb = T.get("arbitration", pd.DataFrame())
    if len(arb):
        A("### Arbitration and clarification metrics (mean over seeds)\n")
        A(_md_table(arb.groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").reset_index()))
    if fig_rel("fig07_actions_by_relationship"):
        A(f"![actions]({fig_rel('fig07_actions_by_relationship')})\n")
    cal = T.get("calibration", pd.DataFrame())
    if len(cal):
        A("### Calibration of the relationship classifier\n")
        A(_md_table(cal.groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").reset_index()))
    if fig_rel("fig11_calibration"):
        A(f"![calibration]({fig_rel('fig11_calibration')})\n")
    A("## 6. Counterfactual driver diagnostic (model-based)\n")
    A("LTP or STI encodings of the trained AIPA model are set to zero and the fused ranking is recomputed. "
      "Δ NDCG@10 and top-10 overlap quantify how much each signal drove the factual ranking; driver labels use a top-K "
      f"disagreement threshold τ = {cfg.cf_tau} and a dominance ratio of {cfg.cf_dominance} (a signal is the sole driver when its disruption is at least {cfg.cf_dominance}x the other; both above τ without dominance = jointly driven; both below τ = neither). This is an interventional diagnostic of the *model*, not an estimate of causal effects.\n")
    A(_md_table(T.get("counterfactual_by_relationship")))
    drv = T.get("drivers", pd.DataFrame())
    if len(drv):
        A(_md_table(drv.groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").reset_index()))
    if len(T.get("driver_action_agreement", [])):
        A("Driver-action agreement (share of instances where the diagnostic driver matches the chosen arbitration action):\n")
        A(_md_table(T["driver_action_agreement"]))
    if fig_rel("fig08_counterfactual"):
        A(f"![counterfactual]({fig_rel('fig08_counterfactual')})\n")
    A("## 7. Temporary override vs. persistent preference shift\n")
    ps_ = T.get("persistence_shifts", pd.DataFrame())
    A(f"The tracker is replayed in chronological order per seeker over the natural test dialogues (`conversationId`, then turn). "
      f"Persistent shifts detected on the test set (genre prioritised in ≥ {cfg.persistence_k} distinct sessions of a seeker): "
      f"**{len(ps_)}** across {ps_.seed.nunique() if len(ps_) else 0} seed(s).\n")
    A(_md_table(ps_.head(20)))
    pe = T.get("persistence_effect", pd.DataFrame())
    if len(pe):
        A(f"Effect of the tracker (AIPA (full) vs AIPA w/o persistence, Hit@10) on all natural instances, on instances of seekers with "
          f">= {cfg.persistence_min_sessions} test sessions, and on the instances whose LTP prior the tracker actually changed:\n")
        A(_md_table(pe, nd=4))
        aff = pe[pe.subset == "tracker_affected"].iloc[0]
        if aff.n == 0:
            A("**The tracker did not change any test instance in this run**, so AIPA (full) and AIPA w/o persistence are identical and "
              "no persistence effect can be claimed.\n")
        elif np.isnan(aff.get("perm_p", np.nan)) or aff.get("perm_p", 1.0) >= ALPHA:
            A(f"The tracker changed {int(aff.n)} instance(s) but the Hit@10 difference on that subset is not significant; "
              "no persistence effect is claimed.\n")
    sw = T.get("persistence_sweep", pd.DataFrame())
    if len(sw):
        A(f"`persistence_k` sweep over {cfg.persistence_k_grid} (validation split for selection; the test rows are reported for "
          f"transparency only and were not used to choose k = {cfg.persistence_k}):\n")
        cols = [c for c in ["split", "k", "seeds", "n_shifts_mean", "n_seekers_shifted_mean", "n_multi_session_mean", "n_affected_mean",
                            "hit10_multi_without_mean", "hit10_multi_with_mean", "hit10_multi_delta_mean", "hit10_affected_without_mean",
                            "hit10_affected_with_mean", "n_rank_changed_mean"] if c in sw]
        A(_md_table(sw[cols], nd=4))
    A("## 8. Sensitivity analyses\n")
    A("### History length (LTP) - AIPA (full), natural\n")
    A(_md_table(T.get("sens_history")))
    A("### History-length buckets - Hit@10 of every model (natural; mean ± std over seeds)\n")
    hb = T.get("history_buckets")
    if hb is not None and len(hb):
        A(_md_table(hb[["history_bucket", "model", "n", "seeds", "Hit@10_mean", "Hit@10_std", "NDCG@10_mean", "NDCG@10_std"]], max_rows=100))
    if fig_rel("fig09b_history_buckets"):
        A(f"![history buckets]({fig_rel('fig09b_history_buckets')})\n")
    A(f"### Target genre (top-{cfg.genre_breakdown_top}) - Hit@10 of every model (natural; mean ± std over seeds)\n")
    gb = T.get("genre_breakdown")
    if gb is not None and len(gb):
        A(_md_table(gb[["target_genre", "model", "n", "seeds", "Hit@10_mean", "Hit@10_std", "NDCG@10_mean", "NDCG@10_std"]], max_rows=120))
    if fig_rel("fig09c_genre_breakdown"):
        A(f"![genre breakdown]({fig_rel('fig09c_genre_breakdown')})\n")
    A("### STI context length - AIPA (full), natural\n")
    A(_md_table(T.get("sens_sti_length")))
    A("### Synthetic conflict intensity (Conflict/Override, Hit@10 on injected target)\n")
    si = T.get("sens_intensity")
    if si is not None and len(si):
        A(_md_table(si.pivot(index="model", columns="intensity", values="Hit@10").reset_index()))
    if fig_rel("fig09_sensitivity"):
        A(f"![sensitivity]({fig_rel('fig09_sensitivity')})\n")
    A("### Fixed fusion weight sweep\n")
    A(_md_table(T.get("alpha_sweep")))
    if fig_rel("fig10_alpha_sweep"):
        A(f"![alpha]({fig_rel('fig10_alpha_sweep')})\n")
    A("## 9. Ablations\n")
    ab = T["overall_natural"][T["overall_natural"].model.str.startswith("AIPA") | T["overall_natural"].model.isin(["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion"])]
    A(_md_table(_perf_table(ab, metrics=("Hit@10", "NDCG@10", "MRR@10"))))
    A("Per-ablation verdicts (H3), natural Hit@10, AIPA (full) vs. ablation:\n")
    rows = []
    for c in ["AIPA w/o relationship", "AIPA w/o counterfactual", "AIPA w/o clarification", "AIPA w/o persistence", "AIPA (rule policy)"]:
        v, r = _verdict(sig, "natural", "Hit@10", c)
        rows.append({"ablation": c, "verdict": v, "mean_diff": r["mean_diff"] if r else np.nan, "p_holm": (r["wilcoxon_p_holm"] if r else np.nan)})
    A(_md_table(pd.DataFrame(rows), nd=4))
    A("## 10. Computational efficiency\n")
    A(_md_table(T.get("efficiency")))
    if fig_rel("fig13_efficiency"):
        A(f"![efficiency]({fig_rel('fig13_efficiency')})\n")
    A("## 11. Error analysis\n")
    A(_md_table(T.get("error_analysis")))
    A("## 12. Qualitative case studies\n")
    cs = T.get("case_studies")
    if cs is not None and len(cs):
        for i, r in cs.iterrows():
            A(f"**Case {i + 1}** - `{r.sample_id}` ({'synthetic' if r.is_synthetic else 'natural'}; seeker {r.seeker_id})\n")
            A(f"* Dialogue excerpt: {r.dialogue_excerpt}\n* LTP profile (history={r.ltp_history_len}): {r.ltp_profile}\n* STI signal: {r.sti_signal}\n"
              f"* Reference relationship: {r.reference_relationship} ({r.reference_source}); predicted: {r.predicted_relationship} (conf {r.relationship_confidence})\n"
              f"* Arbitration: **{r.arbitration_action}** (w_LTP={r.w_ltp}, w_STI={r.w_sti}); counterfactual driver: {r.counterfactual_driver}\n"
              + (f"* Clarification: _{r.clarification}_\n" if r.clarification else "")
              + f"* Target: {r.target} (rank {r.target_rank}, hit@10={r['hit@10']}); top-5: {r.top5}\n")
    else:
        A("_NOT RUN._\n")
    A("## 13. Hypothesis verdicts\n")
    rows = []
    for c in BASELINE_NAMES:
        v, r = _verdict(sig, "natural", "Hit@10", c)
        rows.append({"hypothesis": "H1 (overall)", "comparison": f"{PRIMARY} vs {c}", "verdict": v, "mean_diff": r["mean_diff"] if r else np.nan, "p_holm": r["wilcoxon_p_holm"] if r else np.nan})
    for sub in CONFLICT_SUBSETS:
        for c in ["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion", "SASRec", "KBRD-style"]:
            v, r = _verdict(sig, sub, "Hit@10", c)
            rows.append({"hypothesis": f"H2 ({sub})", "comparison": f"{PRIMARY} vs {c}", "verdict": v, "mean_diff": r["mean_diff"] if r else np.nan, "p_holm": r["wilcoxon_p_holm"] if r else np.nan})
    if len(rel):
        rn = rel[(rel.model == PRIMARY) & (rel.subset == "natural")]
        rs = rel[(rel.model == PRIMARY) & (rel.subset == "synthetic")]
        chance = 1 / len(RELATIONSHIPS)
        for name, d in [("natural (weak labels)", rn), ("synthetic (controlled)", rs)]:
            if len(d):
                f1 = d.macro_f1.mean()
                rows.append({"hypothesis": "H4 (relationship classifier)", "comparison": f"macro-F1 on {name}", "verdict": "SUPPORTED" if f1 > chance else "NOT SUPPORTED",
                             "mean_diff": f1, "p_holm": np.nan})
    ver = pd.DataFrame(rows)
    A(_md_table(ver, nd=4))
    A("### Objective conclusion\n")
    A(_conclusion(ver, res))
    A("## 14. Limitations and threats to validity\n")
    A("* Natural relationship labels are weak heuristics (genre distributions + lexical markers); relationship metrics on the natural "
      "subset measure agreement with those heuristics, not with human judgement.\n"
      "* Synthetic Conflict/Override targets are sampled, popularity-weighted items of the injected genre; success on that subset "
      "shows intent-following, not recommendation accuracy.\n"
      "* ReDial seekers are crowd workers; cross-session history reflects worker behaviour across HITs, an implementation assumption "
      "standing in for real long-term preference. Ordering by `conversationId` is assumed chronological.\n"
      "* MovieLens genre joins by normalised title/year can mismatch remakes or same-titled films.\n"
      "* Baselines are approximate re-implementations; no claim of reproducing MRGE, DiffLSRec, SASRec, KBRD or other published "
      "systems is made (KBRD-style omits the external knowledge graph and response generator).\n"
      "* Counterfactual diagnostics are interventions on a trained model, not causal effects on users.\n"
      "* The novelty assessment in the accompanying design document is scoped; a broader literature search is needed before claiming "
      "AIPA-CRS is globally unprecedented.\n"
      f"* This run used run mode `{cfg.run_mode}`" + (" on a data subset with few epochs; results are indicative only and the `full` mode should be run before any publication claim.\n" if cfg.run_mode == "quick" else ".\n"))
    A("## 15. Reproducibility\n")
    A(_md_table(pd.DataFrame({"key": list(env), "value": list(env.values())})))
    te = res.extra.get("text_encoder") or {}
    if te:
        A(f"Text encoder: `{te.get('name')}` ({te.get('dim')}-d"
          + (", pretrained sentence-transformers" if te.get("pretrained") else ", TF-IDF + SVD fitted on the training split")
          + f"); encoding time {te.get('encode_seconds')} s, {te.get('n_newly_encoded')} strings newly encoded, "
          f"{te.get('n_cache_hits')} served from the on-disk cache."
          + (f" Requested `{te.get('requested')}` but fell back: {te.get('fallback_reason')}." if te.get("fallback_reason") else "")
          + "\n")
    A("Configuration (`configs/default.yaml`, effective values):\n")
    A(_md_table(pd.DataFrame({"key": list(cfg.to_dict()), "value": [str(v) for v in cfg.to_dict().values()]}), max_rows=200))
    A("Dataset file hashes (SHA-1):\n")
    fh = res.extra.get("file_hashes") or {}
    A(_md_table(pd.DataFrame({"file": list(fh), "sha1": list(fh.values())})))
    if validation is not None:
        A("## 16. Component validation\n")
        A(_md_table(validation))
    text = "\n".join(md)
    md_path = out_dir / "AIPA_CRS_Experimental_Report.md"
    md_path.write_text(text)
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html_doc = ("<!doctype html><html><head><meta charset='utf-8'><title>AIPA-CRS Experimental Report</title>"
                "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;line-height:1.45}"
                "table{border-collapse:collapse;font-size:12px;margin:1em 0}td,th{border:1px solid #ccc;padding:3px 6px}"
                "th{background:#f0f0f0}img{max-width:100%}blockquote{border-left:4px solid #c66;padding-left:1em;color:#444}</style>"
                f"</head><body>{html}</body></html>")
    html_path = out_dir / "AIPA_CRS_Experimental_Report.html"
    html_path.write_text(html_doc)
    return md_path, html_path


def _conclusion(ver: pd.DataFrame, res: Results) -> str:
    if not len(ver):
        return "No verdicts could be computed (NOT RUN).\n"
    h1 = ver[ver.hypothesis.str.startswith("H1")]
    h2 = ver[ver.hypothesis.str.startswith("H2")]
    n1s, n1 = (h1.verdict == "SUPPORTED").sum(), len(h1)
    n1c = (h1.verdict == "CONTRADICTED").sum()
    n2s, n2 = (h2.verdict == "SUPPORTED").sum(), len(h2)
    n2c = (h2.verdict == "CONTRADICTED").sum()
    parts = [f"H1 (overall improvement) is supported in {n1s}/{n1} baseline comparisons and contradicted in {n1c}."]
    parts.append(f"H2 (conflict-specific gain) is supported in {n2s}/{n2} comparisons and contradicted in {n2c}.")
    if n2s == 0 and n1s == 0:
        parts.append("On this run the evidence does **not** support the claim that explicit arbitration improves recommendation quality; "
                     "the differences between AIPA variants and fusion baselines are within noise.")
    elif n2s > 0 and n2s >= n1s:
        parts.append("The evidence is consistent with the central claim that arbitration helps *specifically* under conflict, "
                     "although it rests partly on weak or synthetic labels.")
    else:
        parts.append("The evidence is mixed/inconclusive: some comparisons favour AIPA, but the conflict-specific advantage is not "
                     "established at the chosen significance level.")
    if res.cfg.run_mode == "quick":
        parts.append("Quick mode uses a data subset and few epochs; these verdicts are provisional.")
    return " ".join(parts) + "\n"
