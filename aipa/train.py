"""Training loop with multi-task losses and efficiency accounting."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import ACT2ID, REL2ID, RELATIONSHIPS
from .config import Config, set_seed
from .labeling import REL2ACTION, Label
from .models import AIPA, build_model

REC_LOSSES = ("softmax", "sampled_softmax", "bpr")
CONFLICT_REL_IDS = (REL2ID["Conflict"], REL2ID["Override"])
REL2ACT_ID = torch.tensor([ACT2ID[REL2ACTION[r]] for r in RELATIONSHIPS])


def label_tensors(labels: list[Label]) -> dict[str, torch.Tensor]:
    return {
        "rel": torch.tensor([REL2ID[lab.relationship] for lab in labels]),
        "act": torch.tensor([ACT2ID[lab.action] for lab in labels]),
        "conf": torch.tensor([lab.confidence for lab in labels], dtype=torch.float32),
        "synthetic": torch.tensor([lab.source == "synthetic_controlled" for lab in labels]),
        "weak": torch.tensor([lab.source == "weak_rule" for lab in labels]),
    }


def _slice(t: dict[str, torch.Tensor], idx: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    return {k: v[idx].to(device) for k, v in t.items()}


def _sample_negatives(target: torch.Tensor, n_items: int, n_uniform: int) -> torch.Tensor:
    """[B, B + n_uniform] candidate columns: in-batch targets + uniform items (padding column 0 excluded)."""
    B = target.shape[0]
    uni = torch.randint(1, n_items, (B, n_uniform), device=target.device) if n_uniform > 0 else target.new_empty((B, 0))
    return torch.cat([target.unsqueeze(0).expand(B, B), uni], 1)


def rec_loss(scores: torch.Tensor, target: torch.Tensor, mode: str = "softmax", n_negatives: int = 256) -> torch.Tensor:
    """Per-instance recommendation loss [B].

    * ``softmax``          - full cross-entropy over the catalogue;
    * ``sampled_softmax``  - cross-entropy restricted to the positive, the other
                             in-batch targets and ``n_negatives`` uniform items
                             (accidental hits of the positive are masked);
    * ``bpr``              - mean pairwise -log sigmoid(s_pos - s_neg) over the
                             same negative set.
    """
    if mode == "softmax":
        return F.cross_entropy(scores, target, reduction="none")
    if mode not in REC_LOSSES:
        raise ValueError(f"rec_loss must be one of {REC_LOSSES}, got {mode!r}")
    pos = scores.gather(1, target.unsqueeze(1))  # [B,1]
    neg_idx = _sample_negatives(target, scores.shape[1], n_negatives)
    neg = scores.gather(1, neg_idx)
    accidental = neg_idx == target.unsqueeze(1)
    if mode == "sampled_softmax":
        logits = torch.cat([pos, neg.masked_fill(accidental, -1e9)], 1)
        return F.cross_entropy(logits, torch.zeros_like(target), reduction="none")
    valid = (~accidental).float()
    pair = -F.logsigmoid(pos - neg) * valid
    return pair.sum(1) / valid.sum(1).clamp(min=1.0)


def instance_weights(rel: torch.Tensor, conflict_weight: float) -> torch.Tensor:
    """Recommendation-loss weights: Conflict/Override instances get ``conflict_weight``
    (natural weak-rule or synthetic), everything else 1; normalised to mean 1 so
    the overall loss scale is unchanged."""
    w = torch.ones_like(rel, dtype=torch.float32)
    if conflict_weight != 1.0:
        is_conf = torch.zeros_like(rel, dtype=torch.bool)
        for r in CONFLICT_REL_IDS:
            is_conf |= rel == r
        w = torch.where(is_conf, torch.full_like(w, float(conflict_weight)), w)
        w = w / w.mean().clamp(min=1e-8)
    return w


@torch.no_grad()
def self_train_relabel(model, train: dict[str, torch.Tensor], original: dict[str, torch.Tensor], cfg: Config,
                       min_conf: float, threshold: float) -> tuple[dict[str, torch.Tensor], int]:
    """Confidence-filtered self-training for the relationship head.

    Weak-rule labels whose rule confidence is below ``min_conf`` are replaced
    by the model's own prediction when its softmax max-probability is at least
    ``threshold``.  Decisions are always taken against the *original* weak
    labels, so relabelling does not drift across epochs.  Synthetic and
    human-verified labels are never touched; the action label follows
    ``REL2ACTION`` of the new relationship.
    """
    model.eval()
    n = original["rel"].shape[0]
    preds, probs = [], []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(i + 1024, n))
        out = model(_slice(train, idx, cfg.device))
        p = out["rel_logits"].softmax(-1)
        c, k = p.max(-1)
        preds.append(k.cpu())
        probs.append(c.cpu())
    pred, prob = torch.cat(preds), torch.cat(probs)
    eligible = original["weak"] & (original["conf"] < min_conf) & (prob >= threshold)
    new = {k: v.clone() for k, v in original.items()}
    new["rel"] = torch.where(eligible, pred, original["rel"])
    new["act"] = torch.where(eligible, REL2ACT_ID[pred], original["act"])
    new["conf"] = torch.where(eligible, prob, original["conf"])
    model.train()
    return new, int(eligible.sum())


def train_model(
    name: str,
    content: torch.Tensor,
    train: dict[str, torch.Tensor],
    train_labels: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    cfg: Config,
    seed: int,
    lambda_rel: float = 0.5,
    lambda_act: float = 0.3,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict]:
    set_seed(seed)
    device = cfg.device
    v = cfg.values
    loss_mode = v.get("rec_loss", "softmax")
    n_neg = int(v.get("n_negatives", 256))
    conflict_w = float(v.get("conflict_loss_weight", 1.0))
    smoothing = float(v.get("rel_label_smoothing", 0.0))
    self_train = bool(v.get("self_train", False))
    model = build_model(name, content, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    n = train["target"].shape[0]
    is_aipa = isinstance(model, AIPA) and model.variant.fusion == "aipa"
    original_labels = train_labels
    history = []
    best_state, best_val, patience = None, -1.0, 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        n_relabelled = 0
        if is_aipa and self_train and epoch >= int(v.get("self_train_start_epoch", 3)):
            train_labels, n_relabelled = self_train_relabel(
                model, train, original_labels, cfg, float(v.get("self_train_min_conf", 0.6)),
                float(v.get("self_train_threshold", 0.9)))
        model.train()
        perm = torch.randperm(n)
        tot = {"rec": 0.0, "rel": 0.0, "act": 0.0}
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            b = _slice(train, idx, device)
            lb = _slice(train_labels, idx, device)
            out = model(b)
            per_inst = rec_loss(out["scores"], b["target"], loss_mode, n_neg)
            loss_rec = (per_inst * instance_weights(lb["rel"], conflict_w)).mean()
            loss = loss_rec
            tot["rec"] += loss_rec.item() * len(idx)
            if is_aipa:
                w = lb["conf"]
                loss_rel = (F.cross_entropy(out["rel_logits"], lb["rel"], reduction="none", label_smoothing=smoothing) * w).mean()
                loss = loss + lambda_rel * loss_rel
                tot["rel"] += loss_rel.item() * len(idx)
                if model.variant.learned_policy:
                    act_t = lb["act"]
                    if not model.variant.use_clar:
                        act_t = torch.where(act_t == ACT2ID["Ask_Clarification"], torch.full_like(act_t, ACT2ID["Fuse"]), act_t)
                    loss_act = (F.cross_entropy(out["act_logits"], act_t, reduction="none") * w).mean()
                    loss = loss + lambda_act * loss_act
                    tot["act"] += loss_act.item() * len(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = quick_hit(model, valid, cfg, k=10)
        history.append({"epoch": epoch + 1, **{k: x / n for k, x in tot.items()}, "valid_hit@10": val,
                        "n_self_train_relabelled": n_relabelled})
        if verbose:
            print(f"[{name} seed={seed}] epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in history[-1].items() if k != "epoch"))
        if val > best_val:
            best_val, patience = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    n_params = model.parameter_count()
    eff = {
        "model": name,
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "epochs_run": len(history),
        "n_parameters": n_params,
        "model_size_mb": round(n_params * 4 / 1e6, 3),
        "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else float("nan"),
    }
    return model, {"history": history, "efficiency": eff}


@torch.no_grad()
def quick_hit(model, data: dict[str, torch.Tensor], cfg: Config, k: int = 10) -> float:
    model.eval()
    hits = 0
    n = data["target"].shape[0]
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(i + 1024, n))
        b = _slice(data, idx, cfg.device)
        top = model(b)["scores"].topk(k, -1).indices
        hits += (top == b["target"].unsqueeze(-1)).any(-1).sum().item()
    return hits / max(n, 1)


@torch.no_grad()
def predict(model, data: dict[str, torch.Tensor], cfg: Config, ltp_scale: float = 1.0, sti_scale: float = 1.0,
            ltp_override: torch.Tensor | None = None, max_k: int = 20,
            fixed_alpha: float | None = None) -> dict[str, np.ndarray]:
    """Run inference; returns top-K indices, target ranks and auxiliary outputs."""
    model.eval()
    n = data["target"].shape[0]
    res: dict[str, list] = {"topk": [], "rank": [], "target_score_norm": []}
    is_aipa = isinstance(model, AIPA)
    t0 = time.perf_counter()
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(i + 1024, n))
        b = _slice(data, idx, cfg.device)
        if ltp_override is not None:
            b["ltp_genres"] = ltp_override[idx].to(cfg.device)
        out = model(b, ltp_scale=ltp_scale, sti_scale=sti_scale, fixed_alpha=fixed_alpha) if is_aipa else model(b)
        s = out["scores"]
        tgt = b["target"]
        ts = s.gather(1, tgt.unsqueeze(-1))
        rank = (s > ts).sum(-1) + 1
        res["topk"].append(s.topk(max_k, -1).indices.cpu())
        res["rank"].append(rank.cpu())
        res["target_score_norm"].append(s.softmax(-1).gather(1, tgt.unsqueeze(-1)).squeeze(-1).cpu())
        for key in ["rel_logits", "act_logits", "w_ltp", "w_sti", "cf"]:
            if key in out:
                res.setdefault(key, []).append(out[key].cpu())
    infer = time.perf_counter() - t0
    packed = {k: torch.cat(v).numpy() for k, v in res.items()}
    packed["inference_time_s"] = infer
    packed["inference_ms_per_sample"] = 1000 * infer / max(n, 1)
    return packed
