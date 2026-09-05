"""Relationship labels for LTP/STI pairs.

ReDial has **no** native intent-preference relationship annotation.  Every
label produced here therefore carries an explicit ``relationship_source``:

* ``weak_rule``            - heuristic rule over genre distributions and
                             lexical markers (noisy; used for training only and
                             reported as such);
* ``synthetic_controlled`` - the dialogue context was modified by inserting an
                             English seeker utterance that expresses a controlled
                             relationship (conflict / override / consistent /
                             complement / uncertain) and, for conflict/override/
                             complement, the target item is replaced by an item
                             matching the injected intent (uncertain keeps the
                             original target).  These
                             samples are flagged ``is_synthetic`` and are never
                             mixed silently with natural samples;
* ``human_verified``       - reserved for manual annotation loaded from
                             ``data/annotations/human_verified.csv``.  No such
                             file ships with the repository, so the human-verified
                             set is reported as NOT RUN unless provided.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ACTIONS, RELATIONSHIPS
from .config import Config
from .data import GENRE2ID, GENRES, ReDial
from .preprocess import (
    OVERRIDE_LTP_MARKERS,
    OVERRIDE_STI_MARKERS,
    TENSION_PAIRS,
    Instance,
    genre_hits,
    marker_hits,
    negated_genres,
)

# relationship -> default arbitration action (rule policy)
REL2ACTION = {
    "Consistent": "Fuse",
    "Complement": "Fuse",
    "Conflict": "Prioritize_STI",
    "Override": "Prioritize_STI",
    "Uncertain": "Ask_Clarification",
}

GENRE_PHRASES = {
    "Action": ["an action movie", "something with a lot of action", "an action packed film"],
    "Adventure": ["an adventure movie", "a big adventure film"],
    "Animation": ["an animated movie", "a cartoon", "something animated"],
    "Children": ["a kids movie", "something for children", "a family friendly movie"],
    "Comedy": ["a comedy", "something funny", "something that makes me laugh"],
    "Crime": ["a crime movie", "a gangster film", "a heist movie"],
    "Documentary": ["a documentary", "a documentary about real events"],
    "Drama": ["a drama", "a serious drama", "an emotional drama"],
    "Fantasy": ["a fantasy movie", "something with magic and dragons"],
    "Horror": ["a horror movie", "something scary", "a really scary film"],
    "Musical": ["a musical", "something with singing"],
    "Mystery": ["a mystery", "a whodunit"],
    "Romance": ["a romance", "a romantic movie", "a love story"],
    "Sci-Fi": ["a sci-fi movie", "some science fiction", "something set in space"],
    "Thriller": ["a thriller", "something suspenseful"],
    "War": ["a war movie", "a world war film"],
    "Western": ["a western", "a cowboy movie"],
}
INJECTION_TEMPLATES = {
    # (relationship, intensity) -> templates with {sti} = injected genre phrase, {ltp} = habitual genre name
    ("Conflict", 1): ["Maybe {sti} could be good too.", "I was also thinking about {sti}."],
    ("Conflict", 2): ["I want {sti}.", "Can you recommend {sti}?"],
    ("Conflict", 3): ["I really want {sti}, not a {ltp} movie.", "Please only {sti}, I am not into {ltp} anymore."],
    ("Override", 1): ["Tonight maybe {sti} for a change.", "For tonight I was thinking of {sti}."],
    ("Override", 2): ["Tonight I am in the mood for {sti}.", "For a change I want {sti} this time."],
    ("Override", 3): [
        "Tonight I only want {sti}, no {ltp} this time.",
        "This time I am watching with my kids so I just want {sti}, anything but {ltp}.",
    ],
    ("Consistent", 1): ["Something {ltp_adj} as usual would be nice.", "A {ltp} movie like always."],
    ("Consistent", 2): ["As always I want a {ltp} movie.", "My usual, a {ltp} film please."],
    ("Consistent", 3): ["Stick with {ltp} like I always do, that is my favorite kind.", "As usual, only {ltp} for me."],
    # Complement: {sti} is a genre that co-occurs with the habitual genre and does not contradict it
    ("Complement", 1): ["Maybe {sti} would also be nice.", "I could also go for {sti}."],
    ("Complement", 2): ["I still like {ltp}, but {sti} would be a nice addition.",
                        "Something {ltp_adj} is great, and {sti} would go well with that."],
    ("Complement", 3): ["Ideally {sti} that is also {ltp_adj}, the best of both.",
                        "I want {sti} with a {ltp} feel to it, mixing both would be perfect."],
    # Uncertain: vague, no genre cue; target unchanged
    ("Uncertain", 1): ["I am open to anything really.", "Not sure, anything you like."],
    ("Uncertain", 2): ["Anything is fine, surprise me.", "Whatever you think is good, I do not mind."],
    ("Uncertain", 3): ["Honestly I have no idea what I want tonight, just pick something.",
                       "I really cannot decide, I will watch anything at all."],
}
DEFAULT_INJECTION_RELATIONSHIPS = ["Conflict", "Override", "Consistent", "Complement", "Uncertain"]
GENRE_ADJ = {g: g.lower() for g in GENRES}
GENRE_ADJ.update({"Sci-Fi": "sci-fi", "Film-Noir": "noir", "Children": "kid friendly", "Comedy": "funny",
                  "Horror": "scary", "Romance": "romantic", "Animation": "animated", "Musical": "musical"})


@dataclass
class Label:
    relationship: str
    action: str
    source: str
    confidence: float
    rationale: str


def _vec(d: dict[str, float]) -> np.ndarray:
    v = np.zeros(len(GENRES))
    for g, x in d.items():
        v[GENRE2ID[g]] = x
    return v


def _top(d: dict[str, float], k: int = 2, min_mass: float = 0.15) -> list[str]:
    return [g for g, v in sorted(d.items(), key=lambda kv: -kv[1])[:k] if v >= min_mass]


def weak_rule_label(x: Instance, cfg: Config) -> Label:
    L, S = x.ltp_genres, x.sti_genres
    text = x.seeker_recent_text
    has_ltp = bool(L) and not x.sti_flags["cold_user"]
    if not S and not has_ltp:
        return Label("Uncertain", "Ask_Clarification", "weak_rule", 0.9, "no STI genre evidence and no usable LTP")
    if not S:
        return Label("Uncertain", "Ask_Clarification", "weak_rule", 0.6, "no STI genre evidence in recent seeker turns")
    if not has_ltp:
        return Label("Uncertain", "Prioritize_STI", "weak_rule", 0.6, "cold seeker: no cross-session LTP, STI available")
    lv, sv = _vec(L), _vec(S)
    sim = float(lv @ sv / (np.linalg.norm(lv) * np.linalg.norm(sv) + 1e-9))
    topL, topS = _top(L), _top(S)
    ltp_markers = marker_hits(text, OVERRIDE_LTP_MARKERS)
    sti_markers = marker_hits(text, OVERRIDE_STI_MARKERS)
    neg = negated_genres(text)
    neg_habit = [g for g in neg if L.get(g, 0) >= 0.2]
    tension = any(frozenset((s, g)) in TENSION_PAIRS for s in topS for g in topL if L.get(s, 0) < 0.1)
    new_dim = any(L.get(s, 0) < 0.05 for s in topS)
    if ltp_markers:
        return Label("Consistent", "Fuse", "weak_rule", 0.8, f"explicit habitual marker {ltp_markers[0]!r}")
    if neg_habit:
        rel = "Override" if sti_markers else "Conflict"
        return Label(rel, "Prioritize_STI", "weak_rule", 0.8, f"rejects habitual genre {neg_habit[0]}")
    if tension and sim < 0.25:
        rel = "Override" if sti_markers else "Conflict"
        conf = 0.75 if sti_markers else 0.55 + 0.2 * (0.25 - sim) / 0.25
        return Label(rel, "Prioritize_STI", "weak_rule", round(conf, 3), f"opposing genres (sim={sim:.2f})")
    if sim >= 0.5:
        return Label("Consistent", "Fuse", "weak_rule", round(0.5 + 0.5 * sim, 3), f"aligned genres (sim={sim:.2f})")
    if new_dim:
        conf = 0.5 + 0.3 * (1 - sim)
        return Label("Complement", "Fuse", "weak_rule", round(conf, 3), f"STI adds new genre (sim={sim:.2f})")
    if sim >= 0.25:
        return Label("Consistent", "Fuse", "weak_rule", 0.5, f"partially aligned (sim={sim:.2f})")
    return Label("Uncertain", "Ask_Clarification", "weak_rule", 0.5, f"weak, ambiguous evidence (sim={sim:.2f})")


# --------------------------------------------------------------------------
# controlled synthetic injection
# --------------------------------------------------------------------------

def _item_pool(ds: ReDial, instances: list[Instance]) -> dict[str, tuple[list[int], np.ndarray]]:
    pop: dict[int, int] = {}
    for x in instances:
        pop[x.target] = pop.get(x.target, 0) + 1
    pools = {}
    for g in GENRES:
        items = [m for m, gl in ds.movie_genres.items() if g in gl and m in pop]
        if items:
            w = np.array([pop[m] for m in items], float)
            pools[g] = (items, w / w.sum())
    return pools


def _pair_pool(ds: ReDial, instances: list[Instance]) -> dict[frozenset, tuple[list[int], np.ndarray]]:
    """Popularity-weighted pools of training targets tagged with *both* genres of a pair."""
    pop: dict[int, int] = {}
    for x in instances:
        pop[x.target] = pop.get(x.target, 0) + 1
    pools: dict[frozenset, tuple[list[int], np.ndarray]] = {}
    acc: dict[frozenset, list[int]] = {}
    for m in pop:
        gl = ds.movie_genres.get(m, [])
        for i, a in enumerate(gl):
            for b in gl[i + 1:]:
                acc.setdefault(frozenset((a, b)), []).append(m)
    for k, items in acc.items():
        w = np.array([pop[m] for m in items], float)
        pools[k] = (items, w / w.sum())
    return pools


def complement_genres(ltp_top: str, ltp_genres: dict[str, float], pair_pools: dict[frozenset, tuple], min_pair: int = 3) -> list[str]:
    """Genres that co-occur with the habitual genre on >= ``min_pair`` training
    targets, are not in tension with it and are not already part of the LTP prior."""
    return [g for g in GENRES
            if g != ltp_top and frozenset((g, ltp_top)) not in TENSION_PAIRS and ltp_genres.get(g, 0) < 0.05
            and len(pair_pools.get(frozenset((g, ltp_top)), ([], None))[0]) >= min_pair]


def inject_controlled(
    instances: list[Instance], ds: ReDial, cfg: Config, train_pool: list[Instance], seed: int
) -> list[Instance]:
    """Return synthetic copies of a random subset of `instances`."""
    rng = np.random.RandomState(seed)
    pools = _item_pool(ds, train_pool)
    pair_pools = _pair_pool(ds, train_pool)
    out = []
    candidates = [x for x in instances if x.ltp_genres and not x.sti_flags["cold_user"]]
    n = int(round(cfg.injection_rate * len(instances)))
    if not candidates or n == 0:
        return out
    picks = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    kinds = list(cfg.values.get("injection_relationships", DEFAULT_INJECTION_RELATIONSHIPS))
    unknown = set(kinds) - set(DEFAULT_INJECTION_RELATIONSHIPS)
    if unknown:
        raise ValueError(f"injection_relationships has no templates for {sorted(unknown)}")
    for pi in picks:
        base = candidates[pi]
        rel = kinds[rng.randint(len(kinds))]
        intensity = int(rng.choice(cfg.injection_intensities))
        ltp_top = max(base.ltp_genres.items(), key=lambda kv: kv[1])[0]
        if rel in ("Consistent", "Uncertain"):
            sti_g = ltp_top if rel == "Consistent" else ""
        elif rel == "Complement":
            opts = complement_genres(ltp_top, base.ltp_genres, pair_pools)
            if not opts:
                continue
            sti_g = opts[rng.randint(len(opts))]
        else:
            opts = [g for g in GENRES if frozenset((g, ltp_top)) in TENSION_PAIRS and base.ltp_genres.get(g, 0) < 0.05 and g in pools]
            if not opts:
                continue
            sti_g = opts[rng.randint(len(opts))]
        tmpl = INJECTION_TEMPLATES[(rel, intensity)][rng.randint(len(INJECTION_TEMPLATES[(rel, intensity)]))]
        phr = GENRE_PHRASES.get(sti_g, [f"a {sti_g.lower()} movie"])
        utt = tmpl.format(sti=phr[rng.randint(len(phr))], ltp=ltp_top.lower(), ltp_adj=GENRE_ADJ[ltp_top])
        y = copy.deepcopy(base)
        y.is_synthetic = True
        y.sample_id = f"syn/{rel[:3].lower()}{intensity}/{base.sample_id}"
        y.context = y.context + [{"role": "Seeker", "text": utt, "movies": []}]
        y.seeker_recent_text = (y.seeker_recent_text + " " + utt).strip()
        y.last_seeker_text = utt
        if rel == "Uncertain":
            # the vague utterance withdraws any earlier genre cue; the target is unchanged
            y.sti_genres = {}
        else:
            hits = genre_hits(y.seeker_recent_text) if rel not in ("Consistent", "Complement") else {}
            boost = 2.0 * intensity
            sti = {g: v for g, v in base.sti_genres.items()} if rel in ("Consistent", "Complement") else {}
            if rel == "Complement":
                sti[ltp_top] = sti.get(ltp_top, 0.0) + 0.5 * boost
            sti[sti_g] = sti.get(sti_g, 0.0) + boost
            for g, v in hits.items():
                if g != sti_g:
                    sti[g] = sti.get(g, 0.0) + 0.25 * v
            tot = sum(sti.values())
            y.sti_genres = {g: v / tot for g, v in sti.items()}
        y.sti_flags = dict(base.sti_flags)
        y.sti_flags["override_sti"] = float(bool(marker_hits(utt, OVERRIDE_STI_MARKERS)))
        y.sti_flags["override_ltp"] = float(bool(marker_hits(utt, OVERRIDE_LTP_MARKERS)))
        y.sti_flags["negation"] = float(bool(negated_genres(utt)))
        y.sti_flags["request"] = 1.0
        if rel == "Complement":
            items, w = pair_pools[frozenset((sti_g, ltp_top))]
            y.target = int(items[rng.choice(len(items), p=w)])
        elif rel in ("Conflict", "Override"):
            items, w = pools[sti_g]
            y.target = int(items[rng.choice(len(items), p=w)])
        y.injection = {
            "relationship": rel, "intensity": intensity, "injected_genre": sti_g, "habitual_genre": ltp_top,
            "utterance": utt, "original_target": base.target, "original_sample_id": base.sample_id,
        }
        out.append(y)
    return out


def synthetic_label(x: Instance) -> Label:
    inj = x.injection
    rel = inj["relationship"]
    return Label(rel, REL2ACTION[rel], "synthetic_controlled", 1.0,
                 f"injected {rel} (intensity {inj['intensity']}): {inj['injected_genre']} vs habitual {inj['habitual_genre']}")


def load_human_verified(cfg: Config) -> pd.DataFrame | None:
    p = cfg.path("dataset_path").parent.parent / "annotations" / "human_verified.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"sample_id", "relationship_label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} must contain columns {need}")
    return df


def label_all(instances: list[Instance], cfg: Config, human: pd.DataFrame | None = None) -> list[Label]:
    hv = {} if human is None else dict(zip(human.sample_id, human.relationship_label))
    out = []
    for x in instances:
        if x.is_synthetic:
            out.append(synthetic_label(x))
        elif x.sample_id in hv and hv[x.sample_id] in RELATIONSHIPS:
            r = hv[x.sample_id]
            out.append(Label(r, REL2ACTION[r], "human_verified", 1.0, "manual annotation"))
        else:
            out.append(weak_rule_label(x, cfg))
    return out


def labels_frame(instances: list[Instance], labels: list[Label]) -> pd.DataFrame:
    rows = []
    for x, lab in zip(instances, labels):
        rows.append(
            {
                "sample_id": x.sample_id,
                "seeker_id": x.seeker_id,
                "dialogue_id": x.conv_id,
                "split": x.split,
                "relationship_label": lab.relationship,
                "gold_action": lab.action,
                "relationship_source": lab.source,
                "confidence": lab.confidence,
                "rationale": lab.rationale,
                "is_synthetic": x.is_synthetic,
                "intensity": x.injection.get("intensity", 0),
                "ltp_signal": "; ".join(f"{g}:{v:.2f}" for g, v in sorted(x.ltp_genres.items(), key=lambda kv: -kv[1])[:3]),
                "sti_signal": "; ".join(f"{g}:{v:.2f}" for g, v in sorted(x.sti_genres.items(), key=lambda kv: -kv[1])[:3]),
                "original_context": " | ".join(
                    f"{c['role']}: {c['text']}" for c in (x.context[:-1] if x.is_synthetic else x.context)[-4:]
                ),
                "modified_context": x.injection.get("utterance", ""),
            }
        )
    return pd.DataFrame(rows)


assert set(REL2ACTION) == set(RELATIONSHIPS) and set(REL2ACTION.values()) <= set(ACTIONS)
