"""Generate AIPA_CRS_Research_Implementation.ipynb (run: python scripts/build_notebook.py)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

M("""
# AIPA-CRS: Adaptive Intent-Preference Arbitration for Conversational Recommendation

**Research implementation notebook** - reproducible experimental foundation for the AIPA-CRS study.

## Research design

| Item | Content |
|---|---|
| Problem | Conversational recommenders must reconcile a user's **long-term preference (LTP)** (earlier sessions) with the **short-term intent (STI)** expressed in the current dialogue. Naive fusion treats both as always compatible. |
| Research question | Does *explicit* intent-preference arbitration improve recommendation quality **specifically when STI conflicts with LTP**? |
| Hypotheses | H1 overall gain over LTP-only / STI-only / naive fusion; H2 gain concentrated on Conflict/Override instances; H3 every AIPA component contributes (ablations); H4 the relationship classifier is above chance and calibrated. |
| Relationship taxonomy | Complement, Consistent, Conflict, Override, Uncertain |
| Arbitration actions | Fuse, Prioritize_LTP, Prioritize_STI, Ask_Clarification |
| Dataset | **ReDial** (English, human-human movie recommendation dialogues; seekers recur across dialogues, which is used to build cross-session LTP). MovieLens `ml-latest` supplies English genre metadata. |
| Label sources | `weak_rule` (heuristic, noisy), `synthetic_controlled` (injected, explicitly marked), `human_verified` (optional external file; NOT RUN if absent). ReDial has **no native** relationship labels. |
| Baselines | LTP-only, STI-only, naive fusion, adaptive weighted fusion, sequential GRU baseline, conversation-aware baseline (approximate re-implementations, *not* reproductions of MRGE / DiffLSRec). |
| Statistics | multiple seeds, bootstrap 95% CIs, paired t / Wilcoxon tests with Holm correction, Cohen's d, Cliff's delta. |
| Counterfactual analysis | **Model-based interventional diagnostic** (LTP / STI encodings zeroed at inference) - not a causal-effect estimate. |

> **Scientific-integrity rules enforced by the code**: no fabricated numbers (every table is computed), synthetic instances are flagged in every table, unavailable analyses are reported as `NOT RUN`, and hypothesis verdicts are derived mechanically from the tests.

**Run modes**: `RUN_MODE = "quick"` (default; 25 % of dialogues, 2 seeds, CPU ~15 min) or `"full"` (all data, more epochs, 1000 bootstrap samples).
""")
C("""
RUN_MODE = "quick"   # "quick" | "full"
CLEAN_OUTPUTS = True  # remove outputs/ before running so no stale artefact survives

import os, sys, time, warnings, shutil
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd() if (Path.cwd() / "aipa").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["AIPA_RUN_MODE"] = RUN_MODE
import pandas as pd, numpy as np, torch
from IPython.display import display, Markdown, Image, HTML
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 120)

from aipa.config import load_config, environment_report, set_seed
cfg = load_config(RUN_MODE)
if CLEAN_OUTPUTS:
    shutil.rmtree(cfg.path("output_path"), ignore_errors=True)
for sub in ["figures", "tables", "results", "reports"]:
    (cfg.path("output_path") / sub).mkdir(parents=True, exist_ok=True)
set_seed(cfg.seed)
T0 = time.perf_counter()
print(f"RUN_MODE={cfg.run_mode} device={cfg.device}")
display(pd.DataFrame({"key": list(cfg.to_dict()), "value": [str(v) for v in cfg.to_dict().values()]}))
""")
M("""
## 1. Environment and reproducibility
""")
C("""
env = environment_report()
display(pd.DataFrame({"key": list(env), "value": list(env.values())}))
""")
M("""
## 2. Dataset acquisition and validation

The loader checks for local files, downloads ReDial (and MovieLens genres) if missing, validates sizes / hashes, caches the parsed corpus and reports source + version + hash. If the download is impossible the cell raises a clear error instead of silently substituting data.
""")
C("""
from aipa.data import dataset_status, download_dataset, needs_download, validate_dataset, load_dataset, dataset_statistics, per_seeker_frame, genre_frame, dialogue_frame, utterance_frame
status = dataset_status(cfg); display(status)
if needs_download(cfg):
    ok = download_dataset(cfg)
    if not ok:
        raise RuntimeError("A dataset file could not be downloaded; see the messages above. Place redial_dataset.zip under data/raw/redial/ and/or movies.csv under data/external/ml-latest/, then re-run.")
display(validate_dataset(cfg))
ds = load_dataset(cfg)
print("source:", ds.source)
display(pd.DataFrame({"file": list(ds.file_hashes), "sha1": list(ds.file_hashes.values())}))
""")
M("""
### 2.1 Exploratory data analysis (English corpus)
""")
C("""
stats = dataset_statistics(ds); display(stats)
seekers = per_seeker_frame(ds)
print(f"{len(seekers)} distinct seekers; {(seekers.dialogues >= 2).mean():.1%} appear in >= 2 dialogues (cross-session LTP available)")
display(seekers.describe().T)
display(genre_frame(ds).head(15))
display(dialogue_frame(ds).head(5))
display(utterance_frame(ds).sample(8, random_state=cfg.seed))
""")
M("""
## 3. Preprocessing: leak-free LTP / STI / target construction

* One instance per *new* movie recommended by the recommender at turn *t*.
* **LTP** = the seeker's liked / seen movies and genre distribution from **earlier dialogues only** (lower `conversationId`; treating ID order as collection order is an implementation assumption), plus profile sentences.
* **STI** = seeker utterances *before* turn *t* in the current dialogue (genre lexicon, negations, lexical intent markers such as "for a change", "tonight", "as always").
* Items: TF-IDF + SVD content embeddings of English titles/genres (a sentence-transformer can be substituted offline).
* **Weak-rule labels** are derived from LTP/STI genre divergence + markers; **controlled synthetic Conflict / Override / Consistent** instances are injected at `injection_rate` with explicit `is_synthetic=True` and full injection metadata.
""")
C("""
from aipa.experiments import prepare
from aipa.preprocess import instances_frame
from aipa.labeling import labels_frame
prepared = prepare(cfg, verbose=True)
ds, enc, index, inst, labels, X, human = prepared
meta_test = instances_frame(inst["test"])
lab_test = labels_frame(inst["test"], labels["test"])
print("Human-verified labels:", "loaded" if human is not None else "NOT RUN (data/annotations/human_verified.csv not provided)")
display(lab_test.groupby(["relationship_source", "relationship_label"]).size().rename("count").reset_index())
display(meta_test.describe().T)
""")
C("""
# Two natural and two synthetic examples, with their labels and rationales
ex = [i for i, x in enumerate(inst["test"]) if not x.is_synthetic][:2] + [i for i, x in enumerate(inst["test"]) if x.is_synthetic][:2]
for i in ex:
    x, l = inst["test"][i], labels["test"][i]
    print("=" * 100)
    print(f"{x.sample_id}  synthetic={x.is_synthetic}  seeker={x.seeker_id}  history_sessions={x.history_sessions}  history_items={len(x.history_items)}")
    print("LTP genres:", {k: round(v, 2) for k, v in sorted(x.ltp_genres.items(), key=lambda kv: -kv[1])[:4]})
    print("STI genres:", {k: round(v, 2) for k, v in x.sti_genres.items()}, " flags:", {k: v for k, v in x.sti_flags.items() if v})
    print("Seeker said:", x.seeker_recent_text[:300])
    if x.is_synthetic: print("Injection:", x.injection)
    print(f"LABEL: {l.relationship} -> {l.action}  [{l.source}, conf={l.confidence:.2f}]  {l.rationale}")
    print("TARGET:", ds.movie_titles[x.target], ds.movie_genres.get(x.target))
""")
M("""
## 4. Models

Architecture (see `outputs/figures/fig00_architecture.png` after the run):

* **LTP encoder** - attention over cross-session liked items + profile text + genre prior.
* **STI encoder** - dialogue text + in-session items + STI genre / flag features.
* **Relationship classifier** - 5-way head on (h_LTP, h_STI, interactions, evidence).
* **Counterfactual diagnostic** - top-K disruption when LTP or STI is neutralised (model-based, non-causal).
* **Arbitration policy** - learned (or rule-based) distribution over Fuse / Prioritize_LTP / Prioritize_STI / Ask_Clarification -> fusion weights.
* **Clarification module** - English question templates from LTP/STI genre disagreement.
* **Persistence tracker** - a genre prioritised in >= k sessions becomes a persistent shift folded into LTP; otherwise the override stays temporary.
* Baselines share the towers so differences are attributable to arbitration.
""")
C("""
from aipa.models import VARIANTS, build_model, clarification_question
from aipa.experiments import MODEL_ORDER
display(pd.DataFrame([{"model": k, **{f: getattr(v, f) for f in ["fusion", "use_rel", "use_cf", "use_clar", "use_persist", "learned_policy"]}} for k, v in VARIANTS.items()]))
m = build_model("AIPA (full)", index.content, cfg)
print(f"AIPA (full): {m.parameter_count():,} trainable parameters")
print(clarification_question({"Drama": 0.6, "Romance": 0.4}, {"Horror": 1.0}, "Conflict"))
""")
M("""
## 5. Training, evaluation and all experiments

`run_experiments` trains every model for every seed, then computes: ranking metrics with bootstrap CIs, paired tests, relationship / arbitration / clarification / calibration metrics, the counterfactual driver diagnostic, the temporary-vs-persistent tracker, conflict-subset analyses, sensitivity analyses (history length, STI length, conflict intensity), the alpha sweep, efficiency accounting, case studies and error analysis. Everything is saved under `outputs/results/`.
""")
C("""
from aipa.experiments import run_experiments
res = run_experiments(cfg, verbose=True, prepared=prepared)
print("status:", res.status)
display(res.training_curves.groupby(["model", "seed"]).agg(epochs=("epoch", "max"), best_valid_hit10=("valid_hit@10", "max")).reset_index())
""")
M("""
## 6. Results

### 6.1 Overall ranking quality (natural test instances; mean ± std over seeds, 95 % bootstrap CI)
""")
C("""
def perf(d, metrics=("Hit@10", "NDCG@10", "MRR@10", "Hit@20", "NDCG@20")):
    out = pd.DataFrame({"model": d.model, "n": d.n})
    if "subset" in d:
        out.insert(0, "subset", d.subset.values)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {s:.3f} [{lo:.3f}, {hi:.3f}]" for a, s, lo, hi in zip(d[f"{m}_mean"], d[f"{m}_std"], d[f"{m}_ci_low"], d[f"{m}_ci_high"])]
    return out
T = res.tables
display(perf(T["overall_natural"]))
sig = T["significance"]
display(sig[(sig.subset == "natural") & (sig.metric == "Hit@10")].reset_index(drop=True))
""")
M("""### 6.2 Conflict-sensitive evaluation

Two natural subsets are reported (`table_conflict_natural.csv`, column `subset`): **strict** = weak-rule label in `conflict_strict_labels`
(Conflict/Override), and **broad** ("disagreement") = strict OR (weak-rule confidence >= `disagreement_conf_min` AND Jensen-Shannon
divergence between the LTP and STI genre distributions >= `disagreement_js_min`). Both are derived from weak-rule labels, not human labels.
Synthetic Conflict/Override instances stay separate and are additionally broken down by injection intensity (1/2/3). Paired tests form
per-instance differences within each seed and pool them over seeds (paired t, Wilcoxon, sign-flip permutation; Holm-corrected; Cliff's delta).
""")
C("""
print("Natural conflict subsets (weak-rule labels; noisy):"); display(T["conflict_subset_sizes"]); display(perf(T["conflict_natural"]))
print("Natural non-disagreement:"); display(perf(T["nonconflict_natural"]))
print("Controlled synthetic Conflict/Override (targets are sampled intent-matching items):"); display(perf(T["conflict_synthetic"]))
display(T["conflict_synthetic_by_intensity"])
display(sig[sig.subset.isin(["conflict_natural_strict", "conflict_natural_broad", "conflict_synthetic"]) & (sig.metric == "Hit@10")].reset_index(drop=True))
print("Per-history-length bucket and per-target-genre Hit@10 (all models):"); display(T["history_buckets"]); display(T["genre_breakdown"])
print("Persistence tracker effect / k sweep:"); display(T["persistence_effect"]); display(T["persistence_sweep"])
print("Success criteria:"); display(T["success_criteria"])
""")
M("### 6.3 Relationship classification, arbitration, clarification and calibration")
C("""
display(T["relationship"].groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").round(3))
display(T["arbitration"].groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").round(3))
display(T["calibration"].groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").round(3))
display(T["action_by_relationship"])
""")
M("""
### 6.4 Counterfactual driver diagnostic (model-based; not a causal effect)
""")
C("""
display(T["counterfactual_by_relationship"].round(3))
display(T["drivers"].groupby(["model", "subset"]).mean(numeric_only=True).drop(columns="seed").round(3))
display(T["driver_action_agreement"])
""")
M("### 6.5 Temporary override vs. persistent preference shift")
C("""
ps = T["persistence_shifts"]
print(f"{len(ps)} persistent shifts detected (genre prioritised in >= {cfg.persistence_k} sessions of the same seeker)")
display(ps.head(20))
""")
M("### 6.6 Sensitivity analyses and fusion-weight sweep")
C("""
display(T["sens_history"]); display(T["sens_sti_length"])
display(T["sens_intensity"].pivot(index="model", columns="intensity", values="Hit@10").round(3))
display(T["alpha_sweep"].round(3))
""")
M("### 6.7 Ablations and efficiency")
C("""
ab = T["overall_natural"]
display(perf(ab[ab.model.str.startswith("AIPA") | ab.model.isin(["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion"])], metrics=("Hit@10", "NDCG@10", "MRR@10")))
display(T["efficiency"].round(3))
""")
M("""
## 7. Figures (publication quality; PNG + PDF under `outputs/figures/`)
""")
C("""
from aipa.figures import make_all
figs = make_all(res, stats, genre_frame(ds), per_seeker_frame(ds))
for k, p in figs.items():
    display(Markdown(f"**{k}**")); display(Image(filename=str(p), width=900))
""")
M("## 8. Qualitative case studies (>= 10) and automated error analysis")
C("""
cs = T["case_studies"]
for i, r in cs.iterrows():
    display(Markdown(f"**Case {i+1}** - `{r.sample_id}` ({'synthetic' if r.is_synthetic else 'natural'}; seeker {r.seeker_id})  \\n"
                     f"*Dialogue:* {r.dialogue_excerpt}  \\n*LTP* (history={r.ltp_history_len}): {r.ltp_profile}  \\n*STI:* {r.sti_signal}  \\n"
                     f"*Reference:* {r.reference_relationship} ({r.reference_source}) - *predicted:* {r.predicted_relationship} (conf {r.relationship_confidence})  \\n"
                     f"*Arbitration:* **{r.arbitration_action}** (w_LTP={r.w_ltp}, w_STI={r.w_sti}); *driver:* {r.counterfactual_driver}  \\n"
                     + (f"*Clarification:* _{r.clarification}_  \\n" if r.clarification else "")
                     + f"*Target:* {r.target} (rank {r.target_rank}, hit@10={r['hit@10']}); *top-5:* {r.top5}"))
display(T["error_analysis"].round(3))
""")
M("""
## 9. Tables, automatic report and hypothesis verdicts

All tables are written to `outputs/tables/` (CSV + Markdown) and the full report to `outputs/reports/AIPA_CRS_Experimental_Report.{md,html}`. Verdicts are derived mechanically from the paired tests (Holm-corrected Wilcoxon on per-instance Hit@10, alpha = 0.05).
""")
C("""
from aipa.pipeline import write_tables, validate_components
from aipa.report import build_report
write_tables(res)
md_path, html_path = build_report(res, figs, None, stats)
validation = validate_components(res, figs, (md_path, html_path))
md_path, html_path = build_report(res, figs, validation, stats)
validation.to_csv(cfg.path("output_path") / "results" / "component_validation.csv", index=False)
text = md_path.read_text()
start = text.index("## 13. Hypothesis verdicts"); end = text.index("## 14.")
display(Markdown(text[start:end]))
print("report:", md_path, html_path)
""")
M("## 10. Final validation (PASS / FAIL / NOT RUN per component)")
C("""
display(validation)
n_fail = int((validation.status == "FAIL").sum()); n_nr = int((validation.status == "NOT RUN").sum())
print(f"total runtime: {(time.perf_counter() - T0) / 60:.1f} min  |  PASS={int((validation.status == 'PASS').sum())} FAIL={n_fail} NOT RUN={n_nr}")
print("OVERALL:", "PASS" if n_fail == 0 else "FAIL")
""")
M("""
## 11. Limitations (read before citing any number)

1. **Labels** - natural relationship labels are weak heuristics; relationship metrics on natural data measure agreement with these heuristics, not with human judgement. Synthetic Conflict/Override instances are explicitly marked and evaluated separately; their targets are sampled intent-matching items.
2. **LTP proxy** - ReDial seekers are crowd workers; cross-session history across their dialogues stands in for genuine long-term preference, and `conversationId` order is assumed chronological.
3. **Baselines** - approximate re-implementations; no claim of reproducing MRGE, DiffLSRec or other published systems.
4. **Counterfactuals** - interventions on the trained model, not causal effects on users.
5. **Quick mode** - subset + few epochs; run `RUN_MODE = "full"` before drawing publication-level conclusions. Negative or inconclusive verdicts above are reported as such.
""")

nb["cells"] = cells
out = ROOT / "AIPA_CRS_Research_Implementation.ipynb"
nbf.write(nb, out)
print("wrote", out)
