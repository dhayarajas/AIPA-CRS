"""Training loop with multi-task losses and efficiency accounting."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import ACT2ID, REL2ID
from .config import Config, set_seed
from .labeling import Label
from .models import AIPA, build_model


def label_tensors(labels: list[Label]) -> dict[str, torch.Tensor]:
    return {
        "rel": torch.tensor([REL2ID[lab.relationship] for lab in labels]),
        "act": torch.tensor([ACT2ID[lab.action] for lab in labels]),
        "conf": torch.tensor([lab.confidence for lab in labels], dtype=torch.float32),
        "synthetic": torch.tensor([lab.source == "synthetic_controlled" for lab in labels]),
    }


def _slice(t: dict[str, torch.Tensor], idx: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    return {k: v[idx].to(device) for k, v in t.items()}


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
    model = build_model(name, content, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    n = train["target"].shape[0]
    is_aipa = isinstance(model, AIPA) and model.variant.fusion == "aipa"
    history = []
    best_state, best_val, patience = None, -1.0, 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = {"rec": 0.0, "rel": 0.0, "act": 0.0}
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            b = _slice(train, idx, device)
            lb = _slice(train_labels, idx, device)
            out = model(b)
            loss_rec = F.cross_entropy(out["scores"], b["target"])
            loss = loss_rec
            tot["rec"] += loss_rec.item() * len(idx)
            if is_aipa:
                w = lb["conf"]
                loss_rel = (F.cross_entropy(out["rel_logits"], lb["rel"], reduction="none") * w).mean()
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
        history.append({"epoch": epoch + 1, **{k: v / n for k, v in tot.items()}, "valid_hit@10": val})
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
