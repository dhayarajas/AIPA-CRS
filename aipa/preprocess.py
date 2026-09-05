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

import hashlib
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

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

TFIDF = "tfidf-svd"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """On-disk store of sentence embeddings keyed by the SHA-1 of the text
    (``<interim>/text_cache_<model>_<id-hash>.npz``) so repeated runs and seeds never
    re-encode the same string."""

    def __init__(self, path: Path | None, dim: int):
        self.path, self.dim = path, dim
        self.store: dict[str, np.ndarray] = {}
        self.dirty = False
        if path is not None and path.exists():
            with np.load(path) as z:
                keys, emb = z["keys"], z["emb"]
            if emb.ndim == 2 and emb.shape[1] == dim:
                self.store = dict(zip(keys.tolist(), emb.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.store)

    def get(self, key: str) -> np.ndarray | None:
        return self.store.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self.store[key] = vec.astype(np.float32)
        self.dirty = True

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.store), dtype=str)
        emb = np.stack(list(self.store.values())) if self.store else np.zeros((0, self.dim), np.float32)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, keys=keys, emb=emb)
        tmp.replace(self.path)
        self.dirty = False


class TextEncoder:
    """Text -> fixed-size vector.  ``cfg.embedding_model`` selects the back-end:

    * ``"tfidf-svd"``: word n-gram TF-IDF + truncated SVD (``cfg.text_dim``) fitted on
      the training texts; runs in seconds on CPU.
    * a sentence-transformers model id (e.g. ``sentence-transformers/all-MiniLM-L6-v2``):
      pretrained encoder whose own dimension (384 for MiniLM) becomes ``self.dim``;
      embeddings are L2-normalised, encoded in batches of ``cfg.encoder_batch_size`` and
      cached on disk (:class:`EmbeddingCache`) when ``cfg.text_cache`` is true.

    Empty strings encode to the zero vector under both back-ends so that downstream
    presence checks (``x.abs().sum() > 0``) are encoder-independent.  If the pretrained
    model cannot be loaded and ``cfg.embedding_fallback`` is true the encoder falls back
    to TF-IDF + SVD and records why in ``self.fallback_reason``; otherwise the error is raised.
    """

    def __init__(self, cfg: Config):
        self.requested = cfg.embedding_model
        self.name = cfg.embedding_model
        self.dim = int(cfg.text_dim)
        self.batch_size = int(cfg.values.get("encoder_batch_size", 128))
        self.fallback_reason: str | None = None
        self.encode_seconds = 0.0
        self.n_encoded = 0
        self.n_cache_hits = 0
        self._st = None
        self._cache: EmbeddingCache | None = None
        if self.name != TFIDF:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.name, device=cfg.device)
                self.dim = int(self._st.get_embedding_dimension())
            except Exception as exc:
                reason = f"{self.name!r} unavailable ({exc.__class__.__name__}: {exc})"
                if not cfg.values.get("embedding_fallback", True):
                    raise RuntimeError(f"sentence-transformers model {reason}; set embedding_fallback: true to use {TFIDF}") from exc
                self.fallback_reason = reason
                print(f"sentence-transformers model {self.fallback_reason}; falling back to {TFIDF}")
                self.name = TFIDF
                self._st = None
        if self._st is not None:
            cache_path = None
            if cfg.values.get("text_cache", True):
                slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.name)
                cache_path = cfg.path("interim_path") / f"text_cache_{slug}_{_sha1(self.name)[:8]}.npz"
            self._cache = EmbeddingCache(cache_path, self.dim)
        else:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
            self.svd = TruncatedSVD(n_components=self.dim, random_state=cfg.seed)

    @property
    def is_pretrained(self) -> bool:
        return self._st is not None

    def fit(self, texts: list[str]) -> TextEncoder:
        if self._st is None:
            X = self.vec.fit_transform(texts)
            self.svd.fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        t0 = time.perf_counter()
        Z = self._encode_pretrained(texts) if self._st is not None else self._encode_tfidf(texts)
        self.encode_seconds += time.perf_counter() - t0
        return Z

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        empty = np.array([not t.strip() for t in texts])
        Z = self.svd.transform(self.vec.transform(texts)).astype(np.float32)
        Z[empty] = 0.0
        norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        return Z / norm

    def _encode_pretrained(self, texts: list[str]) -> np.ndarray:
        assert self._st is not None and self._cache is not None
        clean = [t.strip() for t in texts]
        keys = [_sha1(t) for t in clean]
        todo: dict[str, str] = {}
        for t, k in zip(clean, keys):
            if t and self._cache.get(k) is None:
                todo.setdefault(k, t)
            elif t:
                self.n_cache_hits += 1
        if todo:
            uniq = list(todo)
            emb = self._st.encode([todo[k] for k in uniq], batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
            for k, v in zip(uniq, np.asarray(emb, dtype=np.float32)):
                self._cache.put(k, v)
            self.n_encoded += len(uniq)
            self._cache.save()
        Z = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, (t, k) in enumerate(zip(clean, keys)):
            if t:
                Z[i] = self._cache.get(k)
        return Z

    def summary(self) -> dict:
        return {
            "requested": self.requested,
            "name": self.name,
            "dim": self.dim,
            "pretrained": self.is_pretrained,
            "fallback_reason": self.fallback_reason,
            "batch_size": self.batch_size if self.is_pretrained else None,
            "encode_seconds": round(self.encode_seconds, 2),
            "n_newly_encoded": self.n_encoded,
            "n_cache_hits": self.n_cache_hits,
            "cache_size": len(self._cache) if self._cache is not None else None,
            "cache_path": str(self._cache.path) if self._cache is not None and self._cache.path else None,
        }


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


def item_text(ds: ReDial, mid: int, with_genres: bool = True) -> str:
    """Content string of an item: the ReDial title (which already carries the year),
    optionally enriched with its MovieLens genres, e.g.
    ``"Toy Story (1995). Genres: Animation, Comedy"``."""
    title = ds.movie_titles[mid].strip()
    genres = ds.movie_genres.get(mid) or []
    if with_genres and genres:
        return f"{title}. Genres: {', '.join(genres)}"
    return title


def item_texts(ds: ReDial, cfg: Config) -> list[str]:
    with_genres = bool(cfg.values.get("item_text_genres", True))
    return [item_text(ds, mid, with_genres) for mid in sorted(ds.movie_titles)]


def build_item_index(ds: ReDial, encoder: TextEncoder, cfg: Config | None = None) -> ItemIndex:
    ids = [0] + sorted(ds.movie_titles)
    id2pos = {mid: i for i, mid in enumerate(ids)}
    with_genres = bool(cfg.values.get("item_text_genres", True)) if cfg is not None else True
    titles = ["" if i == 0 else item_text(ds, i, with_genres) for i in ids]
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
