"""ReDial acquisition, validation, loading and EDA statistics.

Primary dataset: ReDial (Li et al., NeurIPS 2018), the public English
conversational movie-recommendation corpus distributed by its authors at
https://github.com/ReDialData/website (branch ``data``).  Item genre metadata
is taken from the public MovieLens ``ml-latest`` release (GroupLens) and
joined on normalised title + year.

Nothing in this module fabricates data.  When files are missing and cannot be
downloaded the loader raises ``DatasetUnavailable`` and the notebook reports
the affected experiments as NOT RUN.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import Config

REDIAL_URL = "https://github.com/ReDialData/website/raw/data/redial_dataset.zip"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
REDIAL_FILES = {
    "train_data.jsonl": 28_762_643,
    "test_data.jsonl": 3_790_112,
    "movies_with_mentions.csv": 226_703,
}
MOVIELENS_FILES = ["movies.csv"]
SPLITS = ["train", "valid", "test"]
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
GENRE2ID = {g: i for i, g in enumerate(GENRES)}


class DatasetUnavailable(RuntimeError):
    pass


@dataclass
class Dialogue:
    conv_id: int
    seeker_id: str
    recommender_id: str
    split: str
    messages: list[dict]  # {"text", "sender", "role": "Seeker"|"Recommender", "movies": [item ids]}
    mentions: dict[int, str]  # item id -> title
    seeker_labels: dict[int, dict]  # item id -> {"suggested", "seen", "liked"} (seeker-side annotation)
    recommender_labels: dict[int, dict]


@dataclass
class ReDial:
    dialogues: dict[str, list[Dialogue]]
    movie_titles: dict[int, str]  # ReDial movie id -> title (with year)
    movie_genres: dict[int, list[str]]  # ReDial movie id -> MovieLens genres ([] when unmatched)
    movie_year: dict[int, int]
    source: str
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.movie_titles)

    def all_dialogues(self):
        for s in SPLITS:
            yield from self.dialogues[s]


# ----------------------------------------------------------------------------
# acquisition / validation
# ----------------------------------------------------------------------------

def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_status(cfg: Config) -> pd.DataFrame:
    rows = []
    root = cfg.path("dataset_path")
    ml = cfg.path("external_path") / "ml-latest"
    for name, size in REDIAL_FILES.items():
        p = root / name
        rows.append({"source": "ReDial", "file": name, "present": p.exists(),
                     "bytes": p.stat().st_size if p.exists() else 0, "expected_bytes": size,
                     "size_ok": p.exists() and p.stat().st_size == size})
    for name in MOVIELENS_FILES:
        p = ml / name
        rows.append({"source": "MovieLens", "file": name, "present": p.exists(),
                     "bytes": p.stat().st_size if p.exists() else 0, "expected_bytes": None,
                     "size_ok": p.exists() and p.stat().st_size > 1_000_000})
    return pd.DataFrame(rows)


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as r, dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as exc:  # network failures are reported, never masked
        print(f"Download of {url} failed: {exc!r}")
        return False


def download_dataset(cfg: Config, force: bool = False) -> bool:
    """Fetch ReDial (required) and MovieLens genres (optional metadata)."""
    status = dataset_status(cfg)
    ok = True
    root = cfg.path("dataset_path")
    if force or not status.loc[status.source == "ReDial", "present"].all():
        z = root / "redial_dataset.zip"
        if _download(REDIAL_URL, z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)
        else:
            ok = False
    if force or not status.loc[status.source == "MovieLens", "present"].all():
        z = cfg.path("external_path") / "ml-latest.zip"
        if _download(MOVIELENS_URL, z):
            with zipfile.ZipFile(z) as zf:
                zf.extract("ml-latest/movies.csv", cfg.path("external_path"))
            z.unlink(missing_ok=True)
        else:
            print("MovieLens genres unavailable; items will have empty genre lists.")
    return ok


def validate_dataset(cfg: Config) -> pd.DataFrame:
    status = dataset_status(cfg)
    redial = status[status.source == "ReDial"]
    if not redial["present"].all():
        raise DatasetUnavailable(
            f"ReDial files missing under {cfg.path('dataset_path')}: "
            f"{redial.loc[~redial.present, 'file'].tolist()}. Download from {REDIAL_URL}"
        )
    return status


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

_TITLE_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")
_ARTICLE_RE = re.compile(r"^(?P<t>.*), (?P<a>the|a|an)$", re.I)


def normalise_title(title: str) -> tuple[str, int | None]:
    title = title.strip()
    year = None
    m = _TITLE_RE.match(title)
    if m:
        title, year = m.group("title"), int(m.group("year"))
    t = title.strip().lower()
    am = _ARTICLE_RE.match(t)
    if am:  # MovieLens style "Matrix, The" -> "the matrix"
        t = f"{am.group('a')} {am.group('t')}"
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, year


def _load_movielens_genres(cfg: Config) -> dict[tuple[str, int | None], list[str]]:
    p = cfg.path("external_path") / "ml-latest" / "movies.csv"
    out: dict[tuple[str, int | None], list[str]] = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalise_title(row["title"])
            genres = [g for g in row["genres"].split("|") if g in GENRE2ID]
            out.setdefault(key, genres)
            out.setdefault((key[0], None), genres)
    return out


def _labels(obj) -> dict[int, dict]:
    return {int(k): v for k, v in obj.items()} if isinstance(obj, dict) else {}


def _parse_dialogue(raw: dict, split: str) -> Dialogue:
    seeker = int(raw["initiatorWorkerId"])
    mm = raw["movieMentions"] if isinstance(raw["movieMentions"], dict) else {}
    mentions = {int(k): v for k, v in mm.items() if v}
    msgs = []
    for m in raw["messages"]:
        ids = [int(x) for x in re.findall(r"@(\d+)", m["text"]) if int(x) in mentions]
        msgs.append({
            "text": m["text"],
            "sender": int(m["senderWorkerId"]),
            "role": "Seeker" if int(m["senderWorkerId"]) == seeker else "Recommender",
            "movies": ids,
        })
    return Dialogue(
        conv_id=int(raw["conversationId"]),
        seeker_id=str(seeker),
        recommender_id=str(raw["respondentWorkerId"]),
        split=split,
        messages=msgs,
        mentions=mentions,
        seeker_labels=_labels(raw.get("initiatorQuestions")),
        recommender_labels=_labels(raw.get("respondentQuestions")),
    )


def load_dataset(cfg: Config, use_cache: bool = True, valid_fraction: float = 0.1) -> ReDial:
    """Load ReDial, carve a validation split out of train (by dialogue), join genres, cache."""
    validate_dataset(cfg)
    root = cfg.path("dataset_path")
    cache = cfg.path("interim_path") / "redial_parsed.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)

    movie_titles: dict[int, str] = {}
    with (root / "movies_with_mentions.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie_titles[int(row["movieId"])] = row["movieName"].strip()

    train_raw = [json.loads(line) for line in (root / "train_data.jsonl").open(encoding="utf-8")]
    test_raw = [json.loads(line) for line in (root / "test_data.jsonl").open(encoding="utf-8")]
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(train_raw))
    n_valid = int(len(train_raw) * valid_fraction)
    valid_idx = set(perm[:n_valid].tolist())
    dialogues = {"train": [], "valid": [], "test": []}
    for i, raw in enumerate(train_raw):
        split = "valid" if i in valid_idx else "train"
        dialogues[split].append(_parse_dialogue(raw, split))
    dialogues["test"] = [_parse_dialogue(r, "test") for r in test_raw]
    for s in SPLITS:
        dialogues[s].sort(key=lambda d: d.conv_id)

    # titles mentioned in dialogues but absent from movies_with_mentions.csv
    for d in (x for s in SPLITS for x in dialogues[s]):
        for mid, title in d.mentions.items():
            movie_titles.setdefault(mid, title)

    ml = _load_movielens_genres(cfg)
    movie_genres, movie_year = {}, {}
    for mid, title in movie_titles.items():
        key = normalise_title(title)
        movie_year[mid] = key[1] or 0
        movie_genres[mid] = ml.get(key) or ml.get((key[0], None)) or []

    ds = ReDial(
        dialogues=dialogues,
        movie_titles=movie_titles,
        movie_genres=movie_genres,
        movie_year=movie_year,
        source=f"{REDIAL_URL} ; genres: {MOVIELENS_URL}",
        file_hashes={n: _sha1(root / n) for n in REDIAL_FILES},
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(ds, f)
    return ds


# ----------------------------------------------------------------------------
# EDA frames
# ----------------------------------------------------------------------------

def dataset_statistics(ds: ReDial) -> pd.DataFrame:
    rows = []
    for split in SPLITS + ["all"]:
        dl = list(ds.all_dialogues()) if split == "all" else ds.dialogues[split]
        utter = [m for d in dl for m in d.messages]
        movies = {mid for d in dl for mid in d.mentions}
        rec = sum(1 for d in dl for m in d.messages if m["role"] == "Recommender" for _ in m["movies"])
        genres_cov = np.mean([bool(ds.movie_genres.get(m)) for m in movies]) if movies else 0.0
        rows.append({
            "split": split,
            "seekers": len({d.seeker_id for d in dl}),
            "recommenders": len({d.recommender_id for d in dl}),
            "dialogues": len(dl),
            "utterances": len(utter),
            "movie_mentions": sum(len(d.mentions) for d in dl),
            "recommender_mentions": rec,
            "unique_movies": len(movies),
            "genre_coverage": float(genres_cov),
            "avg_words_per_utterance": float(np.mean([len(m["text"].split()) for m in utter])),
            "avg_turns_per_dialogue": float(np.mean([len(d.messages) for d in dl])),
        })
    return pd.DataFrame(rows)


def per_seeker_frame(ds: ReDial) -> pd.DataFrame:
    agg: dict[str, dict] = {}
    for d in ds.all_dialogues():
        a = agg.setdefault(d.seeker_id, {"dialogues": 0, "liked": 0, "seen": 0, "mentions": 0})
        a["dialogues"] += 1
        a["mentions"] += len(d.mentions)
        for lab in d.seeker_labels.values():
            a["liked"] += int(lab.get("liked") == 1)
            a["seen"] += int(lab.get("seen") == 1)
    return pd.DataFrame([{"seeker_id": k, **v} for k, v in agg.items()])


def utterance_frame(ds: ReDial) -> pd.DataFrame:
    return pd.DataFrame(
        {"split": d.split, "role": m["role"], "words": len(m["text"].split()), "conv_id": d.conv_id}
        for d in ds.all_dialogues() for m in d.messages
    )


def dialogue_frame(ds: ReDial) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": d.split,
            "conv_id": d.conv_id,
            "seeker_id": d.seeker_id,
            "turns": len(d.messages),
            "mentions": len(d.mentions),
            "recommendations": sum(len(m["movies"]) for m in d.messages if m["role"] == "Recommender"),
        }
        for d in ds.all_dialogues()
    )


def genre_frame(ds: ReDial) -> pd.DataFrame:
    counts = {g: 0 for g in GENRES}
    for gl in ds.movie_genres.values():
        for g in gl:
            counts[g] += 1
    return pd.DataFrame({"genre": list(counts), "movies": list(counts.values())}).sort_values("movies", ascending=False)
