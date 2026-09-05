"""Neural components: item tower, LTP / STI encoders, relationship classifier,
arbitration policies, clarification generator, counterfactual driver
diagnostic and all baselines.

Every recommender exposes ``forward(batch) -> dict`` with at least
``scores`` ([B, n_items], padding column 0 masked to -inf).  AIPA variants also
return ``rel_logits`` [B,5], ``act_logits`` [B,4], ``w_ltp``/``w_sti`` [B],
``s_ltp``/``s_sti`` component scores and ``h_ltp``/``h_sti`` encodings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ACT2ID, ACTIONS, RELATIONSHIPS
from .preprocess import FLAG_KEYS, GENRES

N_GENRES = len(GENRES)
N_FLAGS = len(FLAG_KEYS)

# fixed fusion weights implied by each action (w_ltp, w_sti)
ACTION_WEIGHTS = torch.tensor(
    [[0.5, 0.5],  # Fuse
     [0.85, 0.15],  # Prioritize_LTP
     [0.15, 0.85],  # Prioritize_STI
     [0.5, 0.5]]  # Ask_Clarification (recommendation still produced, flagged)
)


@dataclass
class Variant:
    """Architecture switches.  ``fusion`` in {ltp, sti, naive, adaptive, aipa}."""

    name: str
    fusion: str = "aipa"
    use_rel: bool = True  # relationship classifier feeds arbitration
    use_cf: bool = True  # counterfactual driver features feed arbitration
    use_clar: bool = True  # Ask_Clarification action enabled
    use_persist: bool = True  # temporal persistence tracker at inference
    learned_policy: bool = True  # learned vs rule arbitration
    extra: dict = field(default_factory=dict)


VARIANTS: dict[str, Variant] = {
    "LTP-only": Variant("LTP-only", fusion="ltp", use_rel=False, use_cf=False, use_clar=False, use_persist=False),
    "STI-only": Variant("STI-only", fusion="sti", use_rel=False, use_cf=False, use_clar=False, use_persist=False),
    "Naive fusion": Variant("Naive fusion", fusion="naive", use_rel=False, use_cf=False, use_clar=False, use_persist=False),
    "Adaptive fusion": Variant("Adaptive fusion", fusion="adaptive", use_rel=False, use_cf=False, use_clar=False, use_persist=False),
    "AIPA w/o relationship": Variant("AIPA w/o relationship", use_rel=False),
    "AIPA w/o counterfactual": Variant("AIPA w/o counterfactual", use_cf=False),
    "AIPA w/o clarification": Variant("AIPA w/o clarification", use_clar=False),
    "AIPA w/o persistence": Variant("AIPA w/o persistence", use_persist=False),
    "AIPA (rule policy)": Variant("AIPA (rule policy)", learned_policy=False),
    "AIPA (full)": Variant("AIPA (full)"),
}
BASELINE_NAMES = ["LTP-only", "STI-only", "Naive fusion", "Adaptive fusion", "Sequential (GRU)", "Conversation-aware",
                  "SASRec", "KBRD-style"]


def mlp(i: int, h: int, o: int, p: float = 0.1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class ItemTower(nn.Module):
    def __init__(self, content: torch.Tensor, hidden: int):
        super().__init__()
        self.register_buffer("content", content)
        self.emb = nn.Embedding(content.shape[0], hidden, padding_idx=0)
        self.proj = nn.Linear(content.shape[1], hidden)
        self.bias = nn.Parameter(torch.zeros(content.shape[0]))

    def all_items(self) -> torch.Tensor:
        return self.emb.weight + self.proj(self.content)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids) + self.proj(self.content[ids])


def masked_mean(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    m = (ids > 0).float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def score_items(h: torch.Tensor, items: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Dot-product scores against every item plus item bias; padding column 0 masked."""
    s = h @ items.T + bias
    s[:, 0] = -1e9
    return s


def left_align(ids: torch.Tensor) -> torch.Tensor:
    """Move padding (0) to the front of each row, keeping the order of real ids."""
    order = torch.argsort((ids > 0).long(), dim=1, stable=True)
    return ids.gather(1, order)


class LTPEncoder(nn.Module):
    """Cross-session preference: profile text + recency-weighted history + genre prior."""

    def __init__(self, items: ItemTower, text_dim: int, hidden: int, max_history: int):
        super().__init__()
        self.items = items
        self.text = nn.Linear(text_dim, hidden)
        self.genre = nn.Linear(N_GENRES, hidden)
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.pos = nn.Embedding(max_history, hidden)
        self.out = mlp(3 * hidden, hidden, hidden)

    def forward(self, batch: dict) -> torch.Tensor:
        h_ids = batch["history"]
        mask = h_ids > 0
        e = self.items.lookup(h_ids) + self.pos.weight[: h_ids.shape[1]].unsqueeze(0)
        att = (e @ self.query).masked_fill(~mask, -1e9).softmax(-1) * mask.any(1, keepdim=True).float()
        h_hist = (att.unsqueeze(-1) * e).sum(1)
        z = torch.cat([h_hist, self.text(batch["profile"]), self.genre(batch["ltp_genres"])], -1)
        h = self.out(z)
        has = (mask.any(1) | (batch["ltp_genres"].sum(1) > 0) | (batch["profile"].abs().sum(1) > 0)).float()
        return h * has.unsqueeze(-1)


class STIEncoder(nn.Module):
    """Current-session intent: recent seeker text, last utterance, in-dialogue liked items, genre cues, flags."""

    def __init__(self, items: ItemTower, text_dim: int, hidden: int):
        super().__init__()
        self.items = items
        self.ctx = nn.Linear(text_dim, hidden)
        self.last = nn.Linear(text_dim, hidden)
        self.genre = nn.Linear(N_GENRES, hidden)
        self.flags = nn.Linear(N_FLAGS, hidden)
        self.out = mlp(5 * hidden, hidden, hidden)

    def forward(self, batch: dict) -> torch.Tensor:
        cur = masked_mean(self.items.lookup(batch["cur_items"]), batch["cur_items"])
        z = torch.cat([self.ctx(batch["context"]), self.last(batch["last"]), self.genre(batch["sti_genres"]),
                       self.flags(batch["flags"]), cur], -1)
        return self.out(z)


class RelationshipClassifier(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = mlp(4 * hidden + 2 * N_GENRES + N_FLAGS + 1, hidden, len(RELATIONSHIPS))

    @staticmethod
    def features(h_ltp, h_sti, batch) -> torch.Tensor:
        cos = F.cosine_similarity(h_ltp, h_sti, dim=-1, eps=1e-6).unsqueeze(-1)
        return torch.cat([h_ltp, h_sti, h_ltp * h_sti, (h_ltp - h_sti).abs(), batch["ltp_genres"],
                          batch["sti_genres"], batch["flags"], cos], -1)

    def forward(self, h_ltp, h_sti, batch) -> torch.Tensor:
        return self.net(self.features(h_ltp, h_sti, batch))


class CounterfactualDiagnostic(nn.Module):
    """Model-based interventional diagnostic (not a causal-effect estimator).

    Given component scores it computes, per instance, how much the fused top-1
    score margin changes when LTP or STI is neutralised (set to zero), and the
    Jaccard overlap between the counterfactual and factual top-K lists.
    """

    def __init__(self, k: int = 10, tau: float = 0.1, dominance: float = 1.5):
        super().__init__()
        self.k, self.tau, self.dominance = k, tau, dominance

    @torch.no_grad()
    def features(self, s_full, s_no_ltp, s_no_sti) -> torch.Tensor:
        def margin(s):
            top2 = s.topk(2, dim=-1).values
            return (top2[:, 0] - top2[:, 1])

        def overlap(a, b):
            ta, tb = a.topk(self.k, -1).indices, b.topk(self.k, -1).indices
            inter = (ta.unsqueeze(-1) == tb.unsqueeze(1)).any(-1).float().sum(-1)
            return inter / (2 * self.k - inter)

        d_ltp = 1 - overlap(s_full, s_no_ltp)  # effect of removing LTP
        d_sti = 1 - overlap(s_full, s_no_sti)
        return torch.stack([d_ltp, d_sti, margin(s_full), margin(s_no_ltp), margin(s_no_sti)], -1)

    def driver(self, feats: torch.Tensor) -> list[str]:
        d_ltp, d_sti = feats[:, 0], feats[:, 1]
        out = []
        for a, b in zip(d_ltp.tolist(), d_sti.tolist()):
            if a < self.tau and b < self.tau:
                out.append("Neither-driven")
            elif b >= self.tau and b >= self.dominance * a:
                out.append("STI-driven")
            elif a >= self.tau and a >= self.dominance * b:
                out.append("LTP-driven")
            else:
                out.append("Jointly-driven")
        return out


class ArbitrationPolicy(nn.Module):
    """Learned policy: action distribution from relationship posterior, evidence, and CF features."""

    def __init__(self, hidden: int, use_rel: bool, use_cf: bool, use_clar: bool):
        super().__init__()
        self.use_rel, self.use_cf, self.use_clar = use_rel, use_cf, use_clar
        d = 2 * hidden + 2 * N_GENRES + N_FLAGS + 1 + (len(RELATIONSHIPS) if use_rel else 0) + (5 if use_cf else 0)
        self.net = mlp(d, hidden, len(ACTIONS))

    def forward(self, h_ltp, h_sti, batch, rel_probs=None, cf=None) -> torch.Tensor:
        cos = F.cosine_similarity(h_ltp, h_sti, dim=-1, eps=1e-6).unsqueeze(-1)
        parts = [h_ltp, h_sti, batch["ltp_genres"], batch["sti_genres"], batch["flags"], cos]
        if self.use_rel:
            parts.append(rel_probs)
        if self.use_cf:
            parts.append(cf)
        logits = self.net(torch.cat(parts, -1))
        if not self.use_clar:
            logits = logits.clone()
            logits[:, ACT2ID["Ask_Clarification"]] = -1e9
        return logits


def rule_policy(rel_probs: torch.Tensor, batch: dict, threshold: float, use_clar: bool) -> torch.Tensor:
    """Deterministic policy: argmax relationship -> action, with evidence overrides."""
    from .labeling import REL2ACTION

    conf, rel = rel_probs.max(-1)
    cold = batch["flags"][:, FLAG_KEYS.index("cold_user")] > 0.5
    no_sti = batch["sti_genres"].sum(-1) == 0
    out = torch.full_like(rel, ACT2ID["Fuse"])
    for i in range(rel.shape[0]):
        a = REL2ACTION[RELATIONSHIPS[int(rel[i])]]
        if conf[i] < threshold and use_clar:
            a = "Ask_Clarification"
        if cold[i] and not no_sti[i]:
            a = "Prioritize_STI"
        if no_sti[i] and not cold[i]:
            a = "Prioritize_LTP" if not use_clar else a
        if not use_clar and a == "Ask_Clarification":
            a = "Fuse"
        out[i] = ACT2ID[a]
    return F.one_hot(out, len(ACTIONS)).float() * 20.0  # sharp logits


class AIPA(nn.Module):
    def __init__(self, content: torch.Tensor, text_dim: int, hidden: int, max_history: int, variant: Variant,
                 rel_threshold: float = 0.5, cf_tau: float = 0.1, cf_dominance: float = 1.5, top_k: int = 10):
        super().__init__()
        self.variant = variant
        self.items = ItemTower(content, hidden)
        self.ltp = LTPEncoder(self.items, text_dim, hidden, max_history)
        self.sti = STIEncoder(self.items, text_dim, hidden)
        self.rel = RelationshipClassifier(hidden) if variant.fusion == "aipa" else None
        self.cf = CounterfactualDiagnostic(k=top_k, tau=cf_tau, dominance=cf_dominance)
        self.rel_threshold = rel_threshold
        if variant.fusion == "adaptive":
            self.gate = mlp(2 * hidden + 2 * N_GENRES + N_FLAGS, hidden, 1)
        if variant.fusion == "aipa" and variant.learned_policy:
            self.policy = ArbitrationPolicy(hidden, variant.use_rel, variant.use_cf, variant.use_clar)
        self.register_buffer("action_weights", ACTION_WEIGHTS.clone())

    def _score(self, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return score_items(h, items, self.items.bias)

    def forward(self, batch: dict, ltp_scale: float = 1.0, sti_scale: float = 1.0,
                fixed_alpha: float | None = None) -> dict:
        items = self.items.all_items()
        h_ltp = self.ltp(batch) * ltp_scale
        h_sti = self.sti(batch) * sti_scale
        s_ltp, s_sti = self._score(h_ltp, items), self._score(h_sti, items)
        B = h_ltp.shape[0]
        out: dict = {"h_ltp": h_ltp, "h_sti": h_sti, "s_ltp": s_ltp, "s_sti": s_sti}
        v = self.variant
        if fixed_alpha is not None:
            w = torch.tensor([fixed_alpha, 1.0 - fixed_alpha], device=h_ltp.device).expand(B, 2)
        elif v.fusion == "ltp":
            w = torch.tensor([1.0, 0.0], device=h_ltp.device).expand(B, 2)
        elif v.fusion == "sti":
            w = torch.tensor([0.0, 1.0], device=h_ltp.device).expand(B, 2)
        elif v.fusion == "naive":
            w = torch.tensor([0.5, 0.5], device=h_ltp.device).expand(B, 2)
        elif v.fusion == "adaptive":
            a = torch.sigmoid(self.gate(torch.cat([h_ltp, h_sti, batch["ltp_genres"], batch["sti_genres"], batch["flags"]], -1)))
            w = torch.cat([a, 1 - a], -1)
        else:
            rel_logits = self.rel(h_ltp, h_sti, batch)
            rel_probs = rel_logits.softmax(-1)
            out["rel_logits"] = rel_logits
            s_naive = 0.5 * (s_ltp + s_sti)
            cf = self.cf.features(s_naive, s_sti, s_ltp) if v.use_cf else None
            out["cf"] = cf if cf is not None else self.cf.features(s_naive, s_sti, s_ltp)
            if v.learned_policy:
                act_logits = self.policy(h_ltp, h_sti, batch, rel_probs.detach() if v.use_rel else None, cf)
            else:
                act_logits = rule_policy(rel_probs.detach(), batch, self.rel_threshold, v.use_clar)
            out["act_logits"] = act_logits
            w = act_logits.softmax(-1) @ self.action_weights
        out["w_ltp"], out["w_sti"] = w[:, 0], w[:, 1]
        out["scores"] = w[:, :1] * s_ltp + w[:, 1:] * s_sti
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequentialBaseline(nn.Module):
    """GRU over the cross-session liked-item sequence plus in-dialogue items
    (GRU4Rec-style; this is NOT a reproduction of any published CRS model)."""

    def __init__(self, content: torch.Tensor, text_dim: int, hidden: int, max_history: int, variant: Variant | None = None):
        super().__init__()
        self.variant = variant or Variant("Sequential (GRU)", fusion="seq")
        self.items = ItemTower(content, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = mlp(2 * hidden, hidden, hidden)

    def forward(self, batch: dict, **_) -> dict:
        seq = torch.cat([batch["history"], batch["cur_items"]], 1)
        e = self.items.lookup(seq)
        lengths = (seq > 0).sum(1)
        _, h = self.gru(e)
        h = h[0] * (lengths > 0).float().unsqueeze(-1)
        z = self.out(torch.cat([h, masked_mean(e, seq)], -1))
        s = z @ self.items.all_items().T + self.items.bias
        s[:, 0] = -1e9
        return {"scores": s}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConversationBaseline(nn.Module):
    """Conversation-only recommender: dialogue text + items mentioned in the
    current dialogue (in the spirit of ReDial / KBRD text+entity baselines; an
    approximate re-implementation, not the original code)."""

    def __init__(self, content: torch.Tensor, text_dim: int, hidden: int, max_history: int, variant: Variant | None = None):
        super().__init__()
        self.variant = variant or Variant("Conversation-aware", fusion="conv")
        self.items = ItemTower(content, hidden)
        self.text = nn.Linear(2 * text_dim, hidden)
        self.out = mlp(2 * hidden, hidden, hidden)

    def forward(self, batch: dict, **_) -> dict:
        cur = masked_mean(self.items.lookup(batch["cur_items"]), batch["cur_items"])
        z = self.out(torch.cat([self.text(torch.cat([batch["context"], batch["last"]], -1)), cur], -1))
        s = z @ self.items.all_items().T + self.items.bias
        s[:, 0] = -1e9
        return {"scores": s}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SASRecBaseline(nn.Module):
    """Self-attentive sequential recommender (SASRec; Kang & McAuley, "Self-Attentive
    Sequential Recommendation", ICDM 2018) over the cross-session liked-item history
    followed by the in-dialogue liked items.  Stacked causal transformer blocks with
    learned absolute positions; the representation at the last real position scores
    every item through the shared ``ItemTower``.  Dialogue text is *not* used, so this
    is the strongest history-only comparison point.  Approximate re-implementation on
    ReDial, not the original code."""

    def __init__(self, content: torch.Tensor, text_dim: int, hidden: int, max_history: int, variant: Variant | None = None,
                 n_blocks: int = 2, n_heads: int = 2, dropout: float = 0.1, max_cur_items: int = 10):
        super().__init__()
        self.variant = variant or Variant("SASRec", fusion="sasrec")
        self.items = ItemTower(content, hidden)
        self.max_len = max_history + max_cur_items
        self.pos = nn.Embedding(self.max_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def states(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position hidden states ``[B, L, H]`` and the real-token mask ``[B, L]``."""
        seq = left_align(seq)[:, -self.max_len:]
        B, L = seq.shape
        real = (seq > 0)
        e = self.items.lookup(seq) + self.pos.weight[:L].unsqueeze(0)
        e = self.drop(e) * real.float().unsqueeze(-1)
        causal = torch.ones(L, L, dtype=torch.bool, device=seq.device).triu(1)
        pad = ~real.unsqueeze(1).expand(B, L, L)
        mask = (causal.unsqueeze(0) | pad) & ~torch.eye(L, dtype=torch.bool, device=seq.device).unsqueeze(0)
        mask = mask.repeat_interleave(self.blocks.layers[0].self_attn.num_heads, 0)
        h = self.blocks(e, mask=mask)
        return self.norm(h) * real.float().unsqueeze(-1), real

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        h, real = self.states(seq)
        return h[:, -1] * real.any(1).float().unsqueeze(-1)

    def forward(self, batch: dict, **_) -> dict:
        h = self.encode(torch.cat([batch["history"], batch["cur_items"]], 1))
        return {"scores": score_items(h, self.items.all_items(), self.items.bias)}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class KBRDBaseline(nn.Module):
    """KBRD-style conversation-aware recommender (Chen et al., "Towards Knowledge-Based
    Recommender Dialog System", EMNLP 2019).  Entity-centric dialogue model: the items
    and genre "entities" mentioned in the current dialogue are embedded and pooled
    (self-attentive pooling as in KBRD, or a mean), a text encoder reads the dialogue
    context and last utterance, the two are fused with a learned gate, and every item
    is scored through the shared ``ItemTower`` plus a genre-aware bias derived from the
    dialogue genre cues.

    This is an approximate re-implementation on ReDial without the external
    knowledge graph (DBpedia entity linking and R-GCN propagation are replaced by the
    item tower and MovieLens genre features) and without the response generator; it is
    not the original code and its numbers are not comparable to the published ones."""

    def __init__(self, content: torch.Tensor, text_dim: int, hidden: int, max_history: int, variant: Variant | None = None,
                 pooling: str = "attention"):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.variant = variant or Variant("KBRD-style", fusion="kbrd")
        self.pooling = pooling
        self.items = ItemTower(content, hidden)
        self.genre_ent = nn.Embedding(N_GENRES, hidden)
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.text = mlp(2 * text_dim, hidden, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.genre_bias = nn.Linear(N_GENRES, N_GENRES, bias=False)

    def entities(self, batch: dict) -> torch.Tensor:
        cur = batch["cur_items"]
        g = batch["sti_genres"]
        e = torch.cat([self.items.lookup(cur), self.genre_ent.weight.unsqueeze(0).expand(cur.shape[0], -1, -1) * g.unsqueeze(-1)], 1)
        present = torch.cat([cur > 0, g > 0], 1)
        if self.pooling == "mean":
            return masked_mean(e, present.long())
        att = (e @ self.query).masked_fill(~present, -1e9).softmax(-1) * present.any(1, keepdim=True).float()
        return (att.unsqueeze(-1) * e).sum(1)

    def forward(self, batch: dict, **_) -> dict:
        h_ent = self.entities(batch)
        h_txt = self.text(torch.cat([batch["context"], batch["last"]], -1))
        gate = torch.sigmoid(self.gate(torch.cat([h_ent, h_txt], -1)))
        h = gate * h_ent + (1 - gate) * h_txt
        item_genres = self.items.content[:, -N_GENRES:]
        s = score_items(h, self.items.all_items(), self.items.bias) + self.genre_bias(batch["sti_genres"]) @ item_genres.T
        return {"scores": s}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name: str, content: torch.Tensor, cfg) -> nn.Module:
    kw = dict(content=content, text_dim=content.shape[1] - N_GENRES, hidden=cfg.hidden_dim, max_history=cfg.max_history)
    if name == "Sequential (GRU)":
        return SequentialBaseline(**kw)
    if name == "Conversation-aware":
        return ConversationBaseline(**kw)
    if name == "SASRec":
        return SASRecBaseline(n_blocks=cfg.values.get("sasrec_blocks", 2), n_heads=cfg.values.get("sasrec_heads", 2),
                              dropout=cfg.values.get("sasrec_dropout", 0.1), **kw)
    if name == "KBRD-style":
        return KBRDBaseline(pooling=cfg.values.get("kbrd_pooling", "attention"), **kw)
    return AIPA(variant=VARIANTS[name], rel_threshold=cfg.relationship_threshold, cf_tau=cfg.cf_tau,
                cf_dominance=cfg.cf_dominance,
                top_k=min(cfg.top_k), **kw)


# --------------------------------------------------------------------------
# clarification (template-based, English)
# --------------------------------------------------------------------------

def clarification_question(ltp_genres: dict[str, float], sti_genres: dict[str, float], rel: str) -> str:
    lt = max(ltp_genres, key=ltp_genres.get) if ltp_genres else None
    st = max(sti_genres, key=sti_genres.get) if sti_genres else None
    if lt and st and lt != st:
        return (f"Earlier you enjoyed {lt.lower()} movies, but right now you seem interested in "
                f"{st.lower()}. Should I focus on {st.lower()} for tonight, or find something that fits both?")
    if not st and lt:
        return f"You usually go for {lt.lower()} movies. Would you like something similar, or are you in the mood for a change?"
    if st and not lt:
        return f"Just to confirm, are you looking for {st.lower()} specifically, or are you open to other genres?"
    return "Could you tell me a bit more about what you are in the mood for tonight?"


# --------------------------------------------------------------------------
# temporary override vs persistent preference shift
# --------------------------------------------------------------------------

class PersistenceTracker:
    """Tracks per-seeker repeated STI-prioritised genres across consecutive
    sessions.  A genre that wins arbitration in >= `k` distinct sessions is
    treated as a persistent shift and folded into the LTP genre prior
    (mass `gain`); otherwise the override remains temporary."""

    def __init__(self, k: int = 2, gain: float = 0.3):
        self.k, self.gain = k, gain
        self.counts: dict[tuple[str, str], set[int]] = {}
        self.shifts: list[dict] = []

    def adjust(self, seeker: str, ltp_genres: torch.Tensor) -> torch.Tensor:
        g = ltp_genres.clone()
        for (s, genre), sess in self.counts.items():
            if s == seeker and len(sess) >= self.k:
                g[GENRES.index(genre)] += self.gain
        return g / g.sum() if g.sum() > 0 else g

    def observe(self, seeker: str, conv_id: int, action: str, sti_genres: dict[str, float]) -> None:
        if action != "Prioritize_STI" or not sti_genres:
            return
        top = max(sti_genres, key=sti_genres.get)
        key = (seeker, top)
        before = len(self.counts.get(key, set()))
        self.counts.setdefault(key, set()).add(conv_id)
        if before < self.k <= len(self.counts[key]):
            self.shifts.append({"seeker_id": seeker, "genre": top, "conv_id": conv_id})
