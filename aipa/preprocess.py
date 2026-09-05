"""Leak-free construction of recommendation instances, LTP / STI signals and
item representations from ReDial.

Temporal separation
-------------------
For an instance created at recommender turn *t* of dialogue *c* (seeker *u*):

* **LTP** is built only from dialogues of *u* whose ``conversationId`` is
  smaller than *c* (ReDial ids are assigned sequentially during collection; we
  treat them as the collection order - an *implementation assumption*).  From
  those sessions we take the movies the seeker annotated as liked / seen
  (genuine ReDial labels) and the seeker's explicit preference statements.
* **STI** is built only from messages of *c* that precede turn *t*.
* The target is the movie mentioned by the recommender at turn *t* that had not
  been mentioned before in *c*.

Nothing after *t* - in the same dialogue or in later dialogues - is used.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .data import GENRE2ID, GENRES, SPLITS, Dialogue, ReDial

# --------------------------------------------------------------------------
# lexicons (interpretable STI features; NOT ground truth)
# --------------------------------------------------------------------------
GENRE_LEXICON: dict[str, list[str]] = {
    "Action": ["action", "explosions", "fight", "superhero", "marvel", "dc comics"],
    "Adventure": ["adventure", "quest", "epic journey", "treasure"],
    "Animation": ["animated", "animation", "cartoon", "pixar", "disney", "anime", "dreamworks"],
    "Children": ["kids", "kid", "children", "child", "family friendly", "family movie", "for the family",
                 "my son", "my daughter", "little ones", "for my nephew", "for my niece", "toddler"],
    "Comedy": ["comedy", "comedies", "funny", "laugh", "hilarious", "humor", "humour", "light hearted",
               "lighthearted", "spoof", "parody", "rom com", "romcom"],
    "Crime": ["crime", "heist", "gangster", "mafia", "mob", "detective", "cop"],
    "Documentary": ["documentary", "documentaries", "true story", "real events", "based on a true"],
    "Drama": ["drama", "dramas", "emotional", "tearjerker", "sad movie", "serious", "cry"],
    "Fantasy": ["fantasy", "magic", "wizard", "dragon", "fairy tale", "harry potter", "lord of the rings"],
    "Film-Noir": ["noir", "film noir"],
    "Horror": ["horror", "scary", "scare", "creepy", "slasher", "haunted", "ghost", "zombie", "gore", "gory",
               "terrifying", "frightening", "paranormal", "jump scares"],
    "Musical": ["musical", "musicals", "singing", "broadway", "songs"],
    "Mystery": ["mystery", "mysteries", "whodunit", "twist ending", "puzzle"],
    "Romance": ["romance", "romantic", "love story", "date night", "chick flick", "rom com", "romcom"],
    "Sci-Fi": ["sci fi", "sci-fi", "scifi", "science fiction", "space", "aliens", "alien", "futuristic",
               "time travel", "robots", "star wars", "star trek"],
    "Thriller": ["thriller", "thrillers", "suspense", "suspenseful", "edge of my seat", "psychological",
                 "intense"],
    "War": ["war movie", "war film", "war movies", "world war", "wwii", "vietnam", "soldiers", "military"],
    "Western": ["western", "westerns", "cowboy", "cowboys", "wild west"],
}
# explicit precedence markers (temporary contextual override, STI dominates)
OVERRIDE_STI_MARKERS = [
    "tonight", "this time", "for tonight", "for a change", "instead", "for once", "right now",
    "for my kids", "with my kids", "with the kids", "for the kids", "with my children", "for my children",
    "for my parents", "with my parents", "with my mom", "with my dad", "with my wife", "with my husband",
    "with my girlfriend", "with my boyfriend", "date night", "family night", "movie night with",
    "something different", "anything but", "don't care about", "do not care about", "only want", "just want",
    "in the mood for", "i need something", "this weekend", "this evening", "for a party", "for halloween",
    "for christmas", "for a sleepover",
]
# explicit statements that the stable preference should govern (LTP dominates)
OVERRIDE_LTP_MARKERS = [
    "as usual", "like always", "as always", "my usual", "my favorite kind", "my favourite kind",
    "like i always", "what i usually", "my go to", "my go-to", "i always watch", "i always like",
    "stick with", "stick to", "same kind", "same type", "more like the ones", "more of the same",
    "my all time favorite", "my all-time favorite",
]
NEGATION_PATTERNS = [
    r"\b(?:not|no|don't|dont|do not|never|hate|dislike|can't stand|cannot stand|sick of|tired of|"
    r"too|nothing|anything but|not into|not a fan of|not really into|not big on)\b[^.!?]{0,40}?\b(%s)\b",
]
PREFERENCE_STATEMENT = re.compile(
    r"\b(i (?:really |also |just |usually |always )?(?:like|love|enjoy|prefer|am into|adore|am a fan of)"
    r"|my favou?rite|i'm into|im into|big fan of)\b",
    re.I,
)
REQUEST_PATTERNS = re.compile(
    r"\b(recommend|suggest|looking for|any (?:good|great)|what (?:about|else)|something (?:like|similar)|"
    r"can you|could you|do you know|ideas|in the mood)\b",
    re.I,
)
# genre pairs whose joint request rarely co-occurs and that pull the ranking
# in opposite directions; used by the weak Conflict rule (an implementation
# assumption, not a dataset label).
TENSION_PAIRS = {
    frozenset(p)
    for p in [
        ("Horror", "Children"), ("Horror", "Animation"), ("Horror", "Romance"), ("Horror", "Musical"),
        ("Horror", "Documentary"), ("Horror", "Comedy"), ("Children", "Thriller"), ("Children", "Crime"),
        ("Children", "War"), ("Children", "Film-Noir"), ("Animation", "Crime"), ("Animation", "War"),
        ("Animation", "Thriller"), ("Comedy", "War"), ("Comedy", "Documentary"), ("Romance", "War"),
        ("Romance", "Action"), ("Musical", "Action"), ("Musical", "Crime"), ("Documentary", "Fantasy"),
        ("Documentary", "Sci-Fi"), ("Documentary", "Action"), ("Drama", "Comedy"), ("Western", "Sci-Fi"),
        ("Western", "Animation"),
    ]
}

_MENTION_RE = re.compile(r"@(\d+)")


@dataclass
class Instance:
    sample_id: str
    conv_id: int
    seeker_id: str
    split: str
    turn: int
    target: int
    context: list[dict]  # messages before `turn` (role, text with titles substituted)
    seeker_recent_text: str
    last_seeker_text: str
    cur_liked_items: list[int]
    cur_mentioned_items: list[int]
    sti_genres: dict[str, float]
    sti_flags: dict[str, float]
    history_items: list[int]  # cross-session liked/seen items, chronological
    history_sessions: int
    profile_sentences: list[str]
    ltp_genres: dict[str, float]
    is_synthetic: bool = False
    injection: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def substitute_titles(text: str, titles: dict[int, str]) -> str:
    return _MENTION_RE.sub(lambda m: titles.get(int(m.group(1)), "a movie"), text)


def genre_hits(text: str) -> dict[str, float]:
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
    hits: dict[str, float] = {}
    for g, kws in GENRE_LEXICON.items():
        n = sum(low.count(" " + k + " ") for k in kws)
        if n:
            hits[g] = float(n)
    return hits


def negated_genres(text: str) -> set[str]:
    low = text.lower()
    out = set()
    for g, kws in GENRE_LEXICON.items():
        alts = "|".join(re.escape(k) for k in kws)
        for pat in NEGATION_PATTERNS:
            if re.search(pat % alts, low):
                out.add(g)
                break
    return out


def marker_hits(text: str, markers: list[str]) -> list[str]:
    low = " " + re.sub(r"[^a-z0-9' -]+", " ", text.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    return [m for m in markers if (" " + m + " ") in low]


def genre_distribution(items: list[int], ds: ReDial, weights: list[float] | None = None) -> dict[str, float]:
    acc = np.zeros(len(GENRES))
    weights = weights or [1.0] * len(items)
    for it, w in zip(items, weights):
        gl = ds.movie_genres.get(it) or []
        for g in gl:
            acc[GENRE2ID[g]] += w / len(gl)
    if acc.sum() > 0:
        acc = acc / acc.sum()
    return {GENRES[i]: float(v) for i, v in enumerate(acc) if v > 0}


# --------------------------------------------------------------------------
# cross-session seeker memory
# --------------------------------------------------------------------------

class SeekerMemory:
    """Per-seeker chronological record of sessions (by conversationId)."""

    def __init__(self, ds: ReDial):
        self.sessions: dict[str, list[tuple[int, list[int], list[str]]]] = {}
        for d in ds.all_dialogues():
            liked = [m for m, lab in d.seeker_labels.items() if lab.get("liked") == 1 or lab.get("seen") == 1]
            prefs = [
                substitute_titles(m["text"], ds.movie_titles)
                for m in d.messages
                if m["role"] == "Seeker" and PREFERENCE_STATEMENT.search(m["text"]) and len(m["text"]) < 200
            ]
            self.sessions.setdefault(d.seeker_id, []).append((d.conv_id, liked, prefs))
        for s in self.sessions.values():
            s.sort(key=lambda x: x[0])

    def before(self, seeker_id: str, conv_id: int) -> tuple[list[int], list[str], int]:
        items, prefs, n = [], [], 0
        for cid, liked, p in self.sessions.get(seeker_id, []):
            if cid >= conv_id:
                break
            n += 1
            items.extend(liked)
            prefs.extend(p)
        return items, prefs, n


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------

def build_instances(ds: ReDial, cfg: Config, memory: SeekerMemory | None = None) -> dict[str, list[Instance]]:
    memory = memory or SeekerMemory(ds)
    rng = np.random.RandomState(cfg.seed)
    out: dict[str, list[Instance]] = {}
    for split in SPLITS:
        dl = ds.dialogues[split]
        if cfg.subset_fraction < 1.0:
            keep = rng.rand(len(dl)) < cfg.subset_fraction
            dl = [d for d, k in zip(dl, keep) if k]
        inst: list[Instance] = []
        for d in dl:
            inst.extend(_instances_from_dialogue(d, ds, cfg, memory))
        out[split] = inst
    return out


def _instances_from_dialogue(d: Dialogue, ds: ReDial, cfg: Config, memory: SeekerMemory) -> list[Instance]:
    hist_items, prefs, n_sessions = memory.before(d.seeker_id, d.conv_id)
    hist_items = hist_items[-cfg.max_history:]
    ltp_genres = genre_distribution(
        hist_items, ds, weights=[cfg.history_temporal_decay ** (len(hist_items) - 1 - i) for i in range(len(hist_items))]
    )
    out = []
    seen: set[int] = set()
    for t, m in enumerate(d.messages):
        if m["role"] == "Recommender":
            for target in dict.fromkeys(m["movies"]):
                if target in seen or target not in ds.movie_titles or t == 0:
                    continue
                out.append(_make_instance(d, ds, cfg, t, target, hist_items, prefs, n_sessions, ltp_genres))
        seen.update(m["movies"])
    return out


def _make_instance(d, ds, cfg, t, target, hist_items, prefs, n_sessions, ltp_genres) -> Instance:
    context = [
        {"role": x["role"], "text": substitute_titles(x["text"], ds.movie_titles), "movies": x["movies"]}
        for x in d.messages[:t]
    ]
    seeker_msgs = [x for x in context if x["role"] == "Seeker"]
    recent = seeker_msgs[-cfg.max_context_turns:]
    recent_text = " ".join(x["text"] for x in recent)
    last_text = recent[-1]["text"] if recent else ""
    cur_mentioned = [mid for x in seeker_msgs for mid in x["movies"]]
    cur_liked = [mid for mid in cur_mentioned if d.seeker_labels.get(mid, {}).get("liked") == 1]
    sti_genres = genre_hits(recent_text)
    for g, v in genre_distribution(cur_liked, ds).items():
        sti_genres[g] = sti_genres.get(g, 0.0) + v
    tot = sum(sti_genres.values())
    sti_genres = {g: v / tot for g, v in sti_genres.items()} if tot else {}
    flags = {
        "override_sti": float(bool(marker_hits(recent_text, OVERRIDE_STI_MARKERS))),
        "override_ltp": float(bool(marker_hits(recent_text, OVERRIDE_LTP_MARKERS))),
        "negation": float(bool(negated_genres(recent_text))),
        "request": float(bool(REQUEST_PATTERNS.search(last_text))),
        "n_seeker_turns": float(len(seeker_msgs)),
        "n_cur_items": float(len(cur_mentioned)),
        "cold_user": float(len(hist_items) < cfg.min_history_for_ltp),
        "history_len": float(len(hist_items)),
    }
    return Instance(
        sample_id=f"{d.conv_id}/{t}/{target}",
        conv_id=d.conv_id,
        seeker_id=d.seeker_id,
        split=d.split,
        turn=t,
        target=target,
        context=context,
        seeker_recent_text=recent_text,
        last_seeker_text=last_text,
        cur_liked_items=cur_liked,
        cur_mentioned_items=cur_mentioned,
        sti_genres=sti_genres,
        sti_flags=flags,
        history_items=list(hist_items),
        history_sessions=n_sessions,
        profile_sentences=prefs[-10:],
        ltp_genres=ltp_genres,
    )


# --------------------------------------------------------------------------
# text encoder
# --------------------------------------------------------------------------

class TextEncoder:
    """Lightweight TF-IDF + SVD encoder (default) or a sentence-transformers model."""

    def __init__(self, cfg: Config):
        self.name = cfg.embedding_model
        self.dim = cfg.text_dim
        self._st = None
        if self.name != "tfidf-svd":
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.name, device=cfg.device)
                self.dim = self._st.get_sentence_embedding_dimension()
            except Exception as exc:
                print(f"sentence-transformers model {self.name!r} unavailable ({exc!r}); falling back to tfidf-svd")
                self.name = "tfidf-svd"
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        self.svd = TruncatedSVD(n_components=self.dim, random_state=cfg.seed)

    def fit(self, texts: list[str]) -> TextEncoder:
        if self._st is None:
            X = self.vec.fit_transform(texts)
            self.svd.fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._st is not None:
            return self._st.encode(texts, batch_size=64, show_progress_bar=False).astype(np.float32)
        empty = np.array([not t.strip() for t in texts])
        Z = self.svd.transform(self.vec.transform(texts)).astype(np.float32)
        Z[empty] = 0.0
        norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        return Z / norm


# --------------------------------------------------------------------------
# tensorisation
# --------------------------------------------------------------------------

@dataclass
class ItemIndex:
    ids: list[int]  # position -> ReDial id ; position 0 is padding
    id2pos: dict[int, int]
    content: torch.Tensor  # [n_items+1, text_dim + n_genres]

    @property
    def n(self) -> int:
        return len(self.ids)


def build_item_index(ds: ReDial, encoder: TextEncoder) -> ItemIndex:
    ids = [0] + sorted(ds.movie_titles)
    id2pos = {mid: i for i, mid in enumerate(ids)}
    titles = ["" if i == 0 else ds.movie_titles[i] for i in ids]
    T = encoder.encode(titles)
    G = np.zeros((len(ids), len(GENRES)), dtype=np.float32)
    for pos, mid in enumerate(ids):
        for g in ds.movie_genres.get(mid, []):
            G[pos, GENRE2ID[g]] = 1.0
    G[0] = 0
    return ItemIndex(ids=ids, id2pos=id2pos, content=torch.tensor(np.concatenate([T, G], axis=1)))


FLAG_KEYS = ["override_sti", "override_ltp", "negation", "request", "n_seeker_turns", "n_cur_items",
             "cold_user", "history_len"]


def tensorise(instances: list[Instance], encoder: TextEncoder, index: ItemIndex, cfg: Config) -> dict[str, torch.Tensor]:
    n = len(instances)
    prof_text = encoder.encode([" ".join(x.profile_sentences) for x in instances])
    ctx_text = encoder.encode([x.seeker_recent_text for x in instances])
    last_text = encoder.encode([x.last_seeker_text for x in instances])
    ltp_g = np.zeros((n, len(GENRES)), np.float32)
    sti_g = np.zeros((n, len(GENRES)), np.float32)
    flags = np.zeros((n, len(FLAG_KEYS)), np.float32)
    hist = np.zeros((n, cfg.max_history), np.int64)
    cur = np.zeros((n, 10), np.int64)
    tgt = np.zeros(n, np.int64)
    for i, x in enumerate(instances):
        for g, v in x.ltp_genres.items():
            ltp_g[i, GENRE2ID[g]] = v
        for g, v in x.sti_genres.items():
            sti_g[i, GENRE2ID[g]] = v
        flags[i] = [x.sti_flags[k] for k in FLAG_KEYS]
        h = [index.id2pos[m] for m in x.history_items if m in index.id2pos][-cfg.max_history:]
        if h:
            hist[i, -len(h):] = h
        c = [index.id2pos[m] for m in x.cur_liked_items if m in index.id2pos][-10:]
        if c:
            cur[i, -len(c):] = c
        tgt[i] = index.id2pos[x.target]
    flags[:, FLAG_KEYS.index("n_seeker_turns")] /= 10.0
    flags[:, FLAG_KEYS.index("n_cur_items")] /= 5.0
    flags[:, FLAG_KEYS.index("history_len")] /= float(cfg.max_history)
    return {
        "profile": torch.tensor(prof_text),
        "ltp_genres": torch.tensor(ltp_g),
        "history": torch.tensor(hist),
        "context": torch.tensor(ctx_text),
        "last": torch.tensor(last_text),
        "sti_genres": torch.tensor(sti_g),
        "flags": torch.tensor(flags),
        "cur_items": torch.tensor(cur),
        "target": torch.tensor(tgt),
    }


def instances_frame(instances: list[Instance]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": x.sample_id,
            "conv_id": x.conv_id,
            "seeker_id": x.seeker_id,
            "split": x.split,
            "turn": x.turn,
            "target": x.target,
            "history_len": len(x.history_items),
            "history_sessions": x.history_sessions,
            "profile_sentences": len(x.profile_sentences),
            "n_sti_genres": len(x.sti_genres),
            "n_ltp_genres": len(x.ltp_genres),
            "seeker_turns": len([c for c in x.context if c["role"] == "Seeker"]),
            "cold_user": bool(x.sti_flags["cold_user"]),
            "is_synthetic": x.is_synthetic,
        }
        for x in instances
    )


def save_instances(instances: dict[str, list[Instance]], cfg: Config, name: str) -> None:
    p = cfg.path("processed_path") / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(instances, f)


def load_instances(cfg: Config, name: str) -> dict[str, list[Instance]] | None:
    p = cfg.path("processed_path") / f"{name}.pkl"
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return None
