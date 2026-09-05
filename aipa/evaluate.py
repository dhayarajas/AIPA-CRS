"""Metrics, bootstrap confidence intervals and paired statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import ACTIONS, RELATIONSHIPS

# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def per_sample_ranking(rank: np.ndarray, ks=(10, 20)) -> pd.DataFrame:
    """Per-sample metrics (one positive per instance, so Recall@K == Hit@K)."""
    d = {}
    for k in ks:
        hit = (rank <= k).astype(float)
        d[f"Hit@{k}"] = hit
        d[f"Recall@{k}"] = hit
        d[f"NDCG@{k}"] = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0)
        d[f"MRR@{k}"] = np.where(rank <= k, 1.0 / rank, 0.0)
    return pd.DataFrame(d)


def bootstrap_ci(x: np.ndarray, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def summarise(per_sample: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for col in per_sample.columns:
        m, lo, hi = bootstrap_ci(per_sample[col].values, n_boot, seed)
        rows.append({"metric": col, "mean": m, "ci_low": lo, "ci_high": hi, "n": len(per_sample)})
    return pd.DataFrame(rows)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of per-sample metric arrays a (treatment) vs b (control)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return {"n": len(d), "mean_diff": float(d.mean()) if len(d) else float("nan"), "t_p": float("nan"),
                "wilcoxon_p": float("nan"), "cohen_d": float("nan"), "cliffs_delta": float("nan")}
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        w_p = float("nan")
    sd = d.std(ddof=1)
    cohen = float(d.mean() / sd) if sd > 0 else float("nan")
    gt = (a[:, None] > b[None, :]).mean() if len(a) <= 2000 else np.mean([np.mean(a > bb) for bb in b[:2000]])
    lt = (a[:, None] < b[None, :]).mean() if len(a) <= 2000 else np.mean([np.mean(a < bb) for bb in b[:2000]])
    return {"n": len(d), "mean_diff": float(d.mean()), "t_p": t_p, "wilcoxon_p": w_p, "cohen_d": cohen,
            "cliffs_delta": float(gt - lt)}


def holm_bonferroni(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i]) if not np.isnan(p[i]) else np.nan
        running = max(running, val) if not np.isnan(val) else running
        adj[i] = running if not np.isnan(val) else np.nan
    return adj.tolist()


# --------------------------------------------------------------------------
# relationship classification
# --------------------------------------------------------------------------

def relationship_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(RELATIONSHIPS)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        **{f"F1_{RELATIONSHIPS[i]}": per[i] for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


# --------------------------------------------------------------------------
# arbitration / clarification
# --------------------------------------------------------------------------

def arbitration_metrics(act_true: np.ndarray, act_pred: np.ndarray, rel_true: np.ndarray, conf_true: np.ndarray,
                        hit: np.ndarray, clar_threshold: float = 0.6) -> dict:
    A = {a: i for i, a in enumerate(ACTIONS)}
    R = {r: i for i, r in enumerate(RELATIONSHIPS)}
    conflict = np.isin(rel_true, [R["Conflict"], R["Override"]])
    override = rel_true == R["Override"]
    asked = act_pred == A["Ask_Clarification"]
    # clarification is "necessary" when the reference label is Uncertain or a low-confidence Conflict
    necessary = (rel_true == R["Uncertain"]) | (conflict & (conf_true < clar_threshold))
    sti_pred = act_pred == A["Prioritize_STI"]
    wrong_override = sti_pred & np.isin(rel_true, [R["Consistent"]])
    return {
        "arbitration_accuracy": float((act_true == act_pred).mean()) if len(act_true) else np.nan,
        "conflict_resolution_accuracy": float((act_true[conflict] == act_pred[conflict]).mean()) if conflict.any() else np.nan,
        "conflict_arbitration_f1": float(f1_score(act_true[conflict], act_pred[conflict], average="macro", zero_division=0)) if conflict.any() else np.nan,
        "override_success_rate": float(hit[override & sti_pred].mean()) if (override & sti_pred).any() else np.nan,
        "clarification_rate": float(asked.mean()) if len(asked) else np.nan,
        "clarification_precision": float(necessary[asked].mean()) if asked.any() else np.nan,
        "clarification_efficiency": float(asked[necessary].mean()) if necessary.any() else np.nan,
        "unnecessary_clarification_rate": float((asked & ~necessary).mean()) if len(asked) else np.nan,
        "wrong_override_rate": float(wrong_override.mean()) if len(asked) else np.nan,
        "n": int(len(act_true)),
        "n_conflict": int(conflict.sum()),
        "n_asked": int(asked.sum()),
    }


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame, float]:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append({"bin_low": lo, "bin_high": hi, "confidence": conf[m].mean(), "accuracy": correct[m].mean(), "count": int(m.sum())})
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    brier = float(np.mean([brier_score_loss((y_true == c).astype(int), probs[:, c]) for c in range(probs.shape[1]) if (y_true == c).any()]))
    return float(ece), pd.DataFrame(rows), brier


# --------------------------------------------------------------------------
# counterfactual driver summary
# --------------------------------------------------------------------------

def driver_summary(drivers: list[str], d_ltp: np.ndarray, d_sti: np.ndarray, overlap_ltp: np.ndarray,
                   overlap_sti: np.ndarray) -> dict:
    s = pd.Series(drivers)
    return {
        "STI_driven_rate": float((s == "STI-driven").mean()),
        "LTP_driven_rate": float((s == "LTP-driven").mean()),
        "Jointly_driven_rate": float((s == "Jointly-driven").mean()),
        "Neither_driven_rate": float((s == "Neither-driven").mean()),
        "mean_abs_delta_LTP": float(np.abs(d_ltp).mean()),
        "mean_abs_delta_STI": float(np.abs(d_sti).mean()),
        "mean_topk_overlap_noLTP": float(overlap_ltp.mean()),
        "mean_topk_overlap_noSTI": float(overlap_sti.mean()),
        "n": int(len(s)),
    }
