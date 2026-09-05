"""Render the LaTeX tables used by paper/main.tex from outputs/results/*.csv.

Run after `python -m aipa.pipeline` so the manuscript never carries hand-typed numbers:

    python scripts/build_paper_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "LTP-only", "STI-only", "Naive fusion", "Adaptive fusion", "Sequential (GRU)", "Conversation-aware",
    "AIPA w/o relationship", "AIPA w/o counterfactual", "AIPA w/o clarification", "AIPA w/o persistence",
    "AIPA (rule policy)", "AIPA (full)",
]
BASELINES = MODEL_ORDER[:6]


_ESC = {
    "\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "$": "\\$", "&": "\\&", "%": "\\%",
    "#": "\\#", "_": "\\_", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
}


def tex(s: str) -> str:
    return "".join(_ESC.get(c, c) for c in str(s))


def f3(x) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


def ranking_table(csv: str, name: str, models=MODEL_ORDER, ci: bool = True) -> None:
    d = pd.read_csv(RES / csv).set_index("model").loc[models]
    rows = []
    for m, r in d.iterrows():
        hit = f"{r['Hit@10_mean']:.3f}"
        if ci:
            hit += f" [{r['Hit@10_ci_low']:.3f}, {r['Hit@10_ci_high']:.3f}]"
        cells = [tex(m), hit, f3(r["NDCG@10_mean"]), f3(r["MRR@10_mean"]), f3(r["Hit@20_mean"]), f3(r["NDCG@20_mean"])]
        if m == "AIPA (full)":
            cells = [f"\\textbf{{{c}}}" for c in cells]
        rows.append(" & ".join(cells) + " \\\\")
        if m == BASELINES[-1]:
            rows.append("\\midrule")
    head = "Model & Hit@10" + (" [95\\% CI]" if ci else "") + " & NDCG@10 & MRR@10 & Hit@20 & NDCG@20 \\\\"
    write(name, "\\begin{tabular}{lccccc}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


ranking_table("table_overall_natural.csv", "overall_natural")
ranking_table("table_conflict_natural.csv", "conflict_natural")
ranking_table("table_conflict_synthetic.csv", "conflict_synthetic")
ranking_table("table_overall_synthetic.csv", "overall_synthetic", ci=False)

# significance: AIPA (full) vs every other model, Hit@10, three subsets
sig = pd.read_csv(RES / "table_significance.csv").query("treatment == 'AIPA (full)' and metric == 'Hit@10'")
SIG_SUBSETS = ["natural", "conflict_natural", "conflict_synthetic"]
sig_n = {s: int(sig.loc[sig.subset == s, "n"].iloc[0]) for s in SIG_SUBSETS}
rows = []
for m in MODEL_ORDER[:-1]:
    cells = [tex(m)]
    for sub in SIG_SUBSETS:
        r = sig[(sig.control == m) & (sig.subset == sub)]
        if r.empty or pd.isna(r.iloc[0]["t_p_holm"]):
            cells += ["--", "--", "--"]
        else:
            r = r.iloc[0]
            p = r["t_p_holm"]
            cells += [f"{r['mean_diff']:+.3f}", (f"{p:.3f}" if p >= 0.001 else "$<$0.001") + ("$^{*}$" if p < 0.05 else ""), f"{r['cohen_d']:+.2f}"]
    rows.append(" & ".join(cells) + " \\\\")
    if m == BASELINES[-1]:
        rows.append("\\midrule")
write("significance", "\\begin{tabular}{l ccc ccc ccc}\n\\toprule\n& \\multicolumn{3}{c}{Natural (all, $n$=" + str(sig_n["natural"]) + ")} & \\multicolumn{3}{c}{Natural conflict ($n$=" + str(sig_n["conflict_natural"]) + ")} & \\multicolumn{3}{c}{Synthetic conflict ($n$=" + str(sig_n["conflict_synthetic"]) + ")} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nControl & $\\Delta$ & $p_{\\text{Holm}}$ & $d$ & $\\Delta$ & $p_{\\text{Holm}}$ & $d$ & $\\Delta$ & $p_{\\text{Holm}}$ & $d$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# relationship classification (mean over seeds)
rel = pd.read_csv(RES / "table_relationship.csv")
rel = rel.groupby(["model", "subset"]).mean(numeric_only=True).reset_index()
rows = []
for m in [x for x in MODEL_ORDER if x.startswith("AIPA")]:
    for sub in ["natural", "synthetic"]:
        r = rel.query("model == @m and subset == @sub").iloc[0]
        absent = "--" if sub == "synthetic" else None
        rows.append(" & ".join([tex(m), sub, f3(r["accuracy"]), f3(r["macro_f1"]), f3(r["weighted_f1"]),
                                absent or f3(r["F1_Complement"]), f3(r["F1_Consistent"]), f3(r["F1_Conflict"]), f3(r["F1_Override"]), absent or f3(r["F1_Uncertain"])]) + " \\\\")
write("relationship", "\\begin{tabular}{llcccccccc}\n\\toprule\nModel & Subset & Acc. & Macro-F1 & W-F1 & Compl. & Consist. & Confl. & Overr. & Uncert. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# arbitration (mean over seeds, natural + synthetic)
arb = pd.read_csv(RES / "table_arbitration.csv")
arb = arb.groupby(["model", "subset"]).mean(numeric_only=True).reset_index()
rows = []
for m in [x for x in MODEL_ORDER if x.startswith("AIPA")]:
    for sub in ["natural", "synthetic"]:
        r = arb.query("model == @m and subset == @sub").iloc[0]
        rows.append(" & ".join([tex(m), sub, f3(r["arbitration_accuracy"]), f3(r["conflict_resolution_accuracy"]), f3(r["override_success_rate"]),
                                f3(r["clarification_rate"]), f3(r["clarification_precision"]), f3(r["unnecessary_clarification_rate"]), f3(r["wrong_override_rate"])]) + " \\\\")
write("arbitration", "\\begin{tabular}{llccccccc}\n\\toprule\nModel & Subset & Arb.\\ acc. & Confl.\\ res. & Overr.\\ succ. & Clar.\\ rate & Clar.\\ prec. & Unnec.\\ clar. & Wrong overr. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# counterfactual by relationship
cf = pd.read_csv(RES / "table_counterfactual_by_relationship.csv")
rows = []
for _, r in cf.iterrows():
    rows.append(" & ".join(["synthetic" if r["is_synthetic"] else "natural", r["relationship_label"], str(int(r["n"])),
                            f3(r["mean_abs_delta_ndcg_LTP"]), f3(r["mean_abs_delta_ndcg_STI"]), f3(r["overlap10_noLTP"]), f3(r["overlap10_noSTI"]),
                            f"{100 * r['STI_driven']:.0f}", f"{100 * r['LTP_driven']:.0f}", f"{100 * r['Jointly_driven']:.0f}", f"{100 * r['Neither_driven']:.0f}"]) + " \\\\")
write("counterfactual", "\\begin{tabular}{llr cc cc cccc}\n\\toprule\n& & & \\multicolumn{2}{c}{$|\\Delta$NDCG@10$|$} & \\multicolumn{2}{c}{Top-10 overlap} & \\multicolumn{4}{c}{Driver (\\%)} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-11}\nSource & Relationship & $n$ & no LTP & no STI & no LTP & no STI & STI & LTP & Joint & Neither \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# efficiency
eff = pd.read_csv(RES / "table_efficiency.csv").set_index("model").loc[MODEL_ORDER]
rows = [" & ".join([tex(m), f"{int(r['n_parameters']):,}", f"{r['model_size_mb']:.2f}", f"{r['train_time_s']:.1f}", f"{r['cpu_inference_ms_per_sample']:.3f}"]) + " \\\\" for m, r in eff.iterrows()]
write("efficiency", "\\begin{tabular}{lrrrr}\n\\toprule\nModel & Parameters & Size (MB) & Train (s) & CPU inf.\\ (ms/sample) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# label distribution
lab_tr = pd.read_csv(RES / "table_label_distribution_train.csv")
lab_te = pd.read_csv(RES / "table_label_distribution_test.csv")
lab = lab_tr.merge(lab_te, on=["relationship_source", "relationship_label"], how="outer", suffixes=("_train", "_test")).fillna(0)
lab = lab.sort_values(["relationship_source", "relationship_label"], ascending=[False, True])
rows = [" & ".join([tex(r["relationship_source"]), r["relationship_label"], str(int(r["count_train"])), str(int(r["count_test"]))]) + " \\\\" for _, r in lab.iterrows()]
write("labels", "\\begin{tabular}{llrr}\n\\toprule\nSource & Relationship & Train & Test \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# alpha sweep
al = pd.read_csv(RES / "table_alpha_sweep.csv")
rows = [" & ".join([f"{r['alpha_ltp']:.2f}", f3(r["Hit@10"]), f3(r["NDCG@10"]), f3(r["Hit@20"]), f3(r["Hit@10_synthetic"])]) + " \\\\" for _, r in al.iterrows()]
write("alpha", "\\begin{tabular}{ccccc}\n\\toprule\n$\\alpha_{\\text{LTP}}$ & Hit@10 & NDCG@10 & Hit@20 & Hit@10 (synthetic) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# case studies (natural, first 6)
cs = pd.read_csv(RES / "table_case_studies.csv")
rows = []
for _, r in cs.head(8).iterrows():
    exc = str(r["dialogue_excerpt"])
    exc = exc if len(exc) <= 170 else exc[:167] + "..."
    rows.append(" & ".join([
        "S" if r["is_synthetic"] else "N", tex(exc), tex(r["ltp_profile"]), tex(r["sti_signal"]),
        f"{r['reference_relationship']} / {r['predicted_relationship']}", tex(r["arbitration_action"]),
        f"{r['w_ltp']:.2f}/{r['w_sti']:.2f}", tex(r["counterfactual_driver"]).replace("-driven", ""), str(int(r["target_rank"]))]) + " \\\\ \\addlinespace")
write("cases", "\\begin{tabular}{@{}c p{5.6cm} p{2.6cm} p{2.6cm} p{2.0cm} l c l r@{}}\n\\toprule\n& Dialogue excerpt (seeker turns before the target) & LTP profile & STI signal & Ref./pred.\\ relationship & Action & $w_{\\text{LTP}}/w_{\\text{STI}}$ & Driver & Rank \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# dataset numbers as macros
meta = json.load(open(RES / "run_metadata.json"))
cfg = meta["config"]
inst = pd.read_csv(RES / "test_instances_meta.csv")
macros = {
    "nSeeds": len(cfg["seeds"]), "nEpochs": cfg["epochs"], "hiddenDim": cfg["hidden_dim"], "textDim": cfg["text_dim"],
    "subsetPct": int(round(100 * cfg["subset_fraction"])), "nBoot": cfg["bootstrap_samples"], "lr": cfg["learning_rate"],
    "batchSize": cfg["batch_size"], "maxHistory": cfg["max_history"], "maxContext": cfg["max_context_turns"],
    "cfTau": cfg["cf_tau"], "cfDom": cfg["cf_dominance"], "persistK": cfg["persistence_k"], "minHist": cfg["min_history_for_ltp"],
    "injRate": cfg["injection_rate"], "nTestNat": int((~inst["is_synthetic"]).sum()), "nTestSyn": int(inst["is_synthetic"].sum()),
    "nTrain": int(lab_tr["count"].sum()), "torchVersion": meta["environment"]["torch"], "pythonVersion": meta["environment"]["python"],
    "cpuCount": meta["environment"]["cpu_count"],
}
write("macros", "\n".join(f"\\newcommand{{\\{k}}}{{{tex(v)}}}" for k, v in macros.items()) + "\n")
