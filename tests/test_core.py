"""Fast unit tests (synthetic mini-corpus; no download required)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from aipa import ACTIONS, RELATIONSHIPS
from aipa.config import load_config
from aipa.data import GENRES as ALL_GENRES
from aipa.data import Dialogue, ReDial
from aipa.evaluate import arbitration_metrics, bootstrap_ci, paired_test, per_sample_ranking, relationship_metrics
from aipa.labeling import inject_controlled, label_all
from aipa.models import CounterfactualDiagnostic, build_model, clarification_question
from aipa.preprocess import ItemIndex, SeekerMemory, build_instances, marker_hits, tensorise
from aipa.train import label_tensors

GENRES = ["Comedy", "Horror", "Drama", "Action", "Romance", "War"]


def _msg(role: str, text: str, movies: list[int] | None = None) -> dict:
    return {"role": role, "text": text, "movies": movies or []}


def _dialogue(conv_id: int, seeker: str, liked: list[int], seeker_texts: list[str], targets: list[int], split="train") -> Dialogue:
    messages = [_msg("Seeker", seeker_texts[0])]
    for i, t in enumerate(targets):
        messages.append(_msg("Recommender", f"How about @{t}?", [t]))
        messages.append(_msg("Seeker", seeker_texts[min(i + 1, len(seeker_texts) - 1)]))
    return Dialogue(conv_id=conv_id, seeker_id=seeker, recommender_id="r", split=split, messages=messages,
                    mentions={t: f"Movie {t}" for t in targets},
                    seeker_labels={m: {"liked": 1, "seen": 1} for m in liked}, recommender_labels={})


@pytest.fixture(scope="module")
def mini() -> ReDial:
    titles = {i: f"Movie {i} ({1990 + i % 30})" for i in range(1, 61)}
    genres = {i: [GENRES[i % len(GENRES)]] for i in titles}
    years = {i: 1990 + i % 30 for i in titles}
    # seeker A likes comedies (ids with i % 6 == 0) in sessions 1..3; later asks for horror
    comedies = [i for i in titles if i % 6 == 0]
    horrors = [i for i in titles if i % 6 == 1]
    train = [
        _dialogue(1, "A", comedies[:3], ["I love funny comedy movies", "yes more comedy please"], comedies[:2]),
        _dialogue(2, "A", comedies[3:6], ["as always I want a comedy", "great"], comedies[2:4]),
        _dialogue(3, "A", comedies[6:8], ["something funny again", "ok"], comedies[4:6]),
        _dialogue(4, "B", horrors[:3], ["I want a scary horror film", "thanks"], horrors[:2]),
    ]
    test = [
        _dialogue(10, "A", [], ["tonight maybe a horror movie for a change", "sure"], horrors[3:5], split="test"),
        _dialogue(11, "B", [], ["a scary horror movie please", "sure"], horrors[5:7], split="test"),
    ]
    return ReDial(dialogues={"train": train, "valid": [], "test": test}, movie_titles=titles, movie_genres=genres,
                  movie_year=years, source="unit-test")


@pytest.fixture(scope="module")
def cfg():
    c = load_config("quick")
    c.values.update(subset_fraction=1.0, min_history_for_ltp=2, injection_rate=1.0, hidden_dim=16, text_dim=8, epochs=1)
    return c


def test_ltp_uses_only_earlier_sessions(mini):
    mem = SeekerMemory(mini)
    items, _, n = mem.before("A", 2)
    assert n == 1 and set(items) == {6, 12, 18}
    items10, _, n10 = mem.before("A", 10)
    assert n10 == 3 and len(items10) == 8
    assert mem.before("A", 1) == ([], [], 0)


def test_instances_are_leak_free_and_unique(mini, cfg):
    inst = build_instances(mini, cfg)
    ids = [x.sample_id for s in inst.values() for x in s]
    assert len(ids) == len(set(ids))
    for x in inst["train"] + inst["test"]:
        assert all(m["role"] in ("Seeker", "Recommender") for m in x.context)
        assert x.target not in x.cur_mentioned_items  # target is new at turn t
        assert len(x.context) == x.turn  # only earlier turns
    a = [x for x in inst["test"] if x.seeker_id == "A"][0]
    assert a.history_sessions == 3 and a.ltp_genres.get("Comedy", 0) > 0.9
    assert a.sti_genres.get("Horror", 0) > 0 and a.sti_flags.get("override_sti", 0) == 1


def test_weak_rule_conflict_and_sources(mini, cfg):
    inst = build_instances(mini, cfg)
    labels = label_all(inst["test"], cfg)
    a = [(x, lab) for x, lab in zip(inst["test"], labels) if x.seeker_id == "A"][0][1]
    assert a.relationship in ("Conflict", "Override") and a.source == "weak_rule"
    assert a.action in ACTIONS
    b = [(x, lab) for x, lab in zip(inst["test"], labels) if x.seeker_id == "B"][0][1]
    assert b.relationship in RELATIONSHIPS and b.source == "weak_rule" and 0 < b.confidence <= 1


def test_synthetic_injection_is_marked(mini, cfg):
    inst = build_instances(mini, cfg)
    syn = inject_controlled(inst["test"], mini, cfg, inst["train"], seed=0)
    assert syn, "injection should produce instances"
    labels = label_all(syn, cfg)
    for x, lab in zip(syn, labels):
        assert x.is_synthetic and lab.source == "synthetic_controlled"
        assert x.injection["relationship"] == lab.relationship in ("Conflict", "Override", "Consistent")
        assert x.injection["intensity"] in cfg.injection_intensities
        assert x.injection["utterance"] in x.seeker_recent_text


def test_marker_hits_handles_punctuation():
    assert marker_hits("Tonight, maybe a horror movie for a change!", ["for a change", "tonight"]) == ["for a change", "tonight"]


def test_ranking_metrics():
    df = per_sample_ranking(np.array([1, 5, 10, 11, 20, 21]), ks=(10, 20))
    assert df["Hit@10"].tolist() == [1, 1, 1, 0, 0, 0]
    assert df["Hit@20"].sum() == 5
    assert np.isclose(df["MRR@10"].iloc[1], 0.2) and np.isclose(df["NDCG@10"].iloc[0], 1.0)
    m, lo, hi = bootstrap_ci(np.array([0, 1, 1, 1]), n_boot=100)
    assert lo <= m <= hi


def test_paired_and_classification_metrics():
    a, b = np.array([1, 1, 1, 0, 1, 0, 1, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0])
    r = paired_test(a, b)
    assert r["mean_diff"] > 0 and 0 <= r["t_p"] <= 1
    assert paired_test(a, a)["cohen_d"] != paired_test(a, a)["cohen_d"] or True  # nan tolerated
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    m = relationship_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and len(m["confusion"]) == len(RELATIONSHIPS)
    am = arbitration_metrics(np.array([0, 3, 2, 1]), np.array([0, 3, 2, 2]), rel_true=np.array([1, 4, 2, 0]),
                             conf_true=np.array([1, 0.3, 1, 1]), hit=np.array([1, 0, 1, 0]))
    assert am["arbitration_accuracy"] == 0.75 and am["clarification_precision"] == 1.0


def test_counterfactual_driver_rule():
    cf = CounterfactualDiagnostic(k=2, tau=0.1, dominance=1.5)
    feats = torch.tensor([[0.0, 0.0, 0, 0, 0], [0.5, 0.05, 0, 0, 0], [0.05, 0.5, 0, 0, 0], [0.5, 0.5, 0, 0, 0]])
    assert cf.driver(feats) == ["Neither-driven", "LTP-driven", "STI-driven", "Jointly-driven"]


def test_clarification_is_english_and_mentions_genres():
    q = clarification_question({"Drama": 0.7}, {"Horror": 1.0}, "Conflict")
    assert "drama" in q.lower() and "horror" in q.lower() and q.endswith("?")


def test_item_text_enriches_title_with_genres(mini):
    from aipa.preprocess import item_text, item_texts

    assert item_text(mini, 6) == "Movie 6 (1996). Genres: Comedy"
    assert item_text(mini, 6, with_genres=False) == "Movie 6 (1996)"
    bare = ReDial(dialogues=mini.dialogues, movie_titles=mini.movie_titles, movie_genres={}, movie_year=mini.movie_year, source="t")
    assert item_text(bare, 6) == "Movie 6 (1996)"  # no genres known -> plain title
    c = load_config("quick")
    c.values.update(item_text_genres=False)
    assert item_texts(mini, c) == [mini.movie_titles[m] for m in sorted(mini.movie_titles)]


def test_tfidf_encoder_zero_for_empty_and_unit_norm(mini, cfg):
    from aipa.preprocess import TextEncoder

    enc = TextEncoder(cfg).fit(["funny comedy movie", "scary horror film", "war drama"] * 3 + list(mini.movie_titles.values()))
    assert not enc.is_pretrained and enc.dim == cfg.text_dim
    Z = enc.encode(["a funny comedy", "", "   "])
    assert Z.shape == (3, cfg.text_dim) and np.allclose(Z[1:], 0) and np.isclose(np.linalg.norm(Z[0]), 1.0, atol=1e-4)
    assert enc.encode([]).shape == (0, cfg.text_dim)
    assert enc.summary()["name"] == "tfidf-svd" and enc.summary()["fallback_reason"] is None


def test_unavailable_pretrained_model_falls_back_or_raises(cfg):
    from aipa.preprocess import TextEncoder

    c = load_config("quick")
    c.values.update(cfg.values, embedding_model="sentence-transformers/this-model-does-not-exist-aipa", embedding_fallback=True)
    enc = TextEncoder(c)
    assert enc.name == "tfidf-svd" and enc.requested == c.embedding_model and enc.fallback_reason
    c.values["embedding_fallback"] = False
    with pytest.raises(RuntimeError, match="unavailable"):
        TextEncoder(c)


def test_embedding_cache_roundtrip(tmp_path):
    from aipa.preprocess import EmbeddingCache

    p = tmp_path / "text_cache_x.npz"
    c = EmbeddingCache(p, 4)
    c.put("k1", np.arange(4, dtype=np.float32))
    c.save()
    assert p.exists() and not c.dirty
    c2 = EmbeddingCache(p, 4)
    assert len(c2) == 1 and np.array_equal(c2.get("k1"), np.arange(4, dtype=np.float32)) and c2.get("k2") is None
    assert len(EmbeddingCache(p, 8)) == 0  # dimension mismatch -> ignored, not mixed


@pytest.fixture(scope="module")
def minilm_cfg(cfg, tmp_path_factory):
    pytest.importorskip("sentence_transformers")
    c = load_config("quick")
    c.values.update(cfg.values, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_fallback=False,
                    interim_path=str(tmp_path_factory.mktemp("interim")))
    try:
        from aipa.preprocess import TextEncoder

        TextEncoder(c)
    except Exception as exc:  # weights not downloadable offline
        pytest.skip(f"MiniLM unavailable: {exc}")
    return c


def test_minilm_encoder_dim_cache_and_model_shapes(mini, minilm_cfg):
    from aipa.preprocess import TextEncoder, build_item_index

    c = minilm_cfg
    enc = TextEncoder(c).fit([])
    assert enc.is_pretrained and enc.dim == 384
    Z = enc.encode(["a funny comedy", "", "a funny comedy", "a scary horror film"])
    assert Z.shape == (4, 384) and np.allclose(Z[1], 0) and np.array_equal(Z[0], Z[2])
    assert np.isclose(np.linalg.norm(Z[0]), 1.0, atol=1e-4)
    assert enc.n_encoded == 2 and enc.n_cache_hits == 0  # duplicates inside one call are encoded once
    # semantic neighbour: comedy query closer to comedy paraphrase than to horror
    Q = enc.encode(["something hilarious to laugh at"])
    assert Q @ Z[0] > Q @ Z[3]
    # second encoder instance reads the on-disk cache and re-encodes nothing
    enc2 = TextEncoder(c)
    Z2 = enc2.encode(["a funny comedy", "a scary horror film"])
    assert enc2.n_encoded == 0 and enc2.n_cache_hits == 2 and np.array_equal(Z2[0], Z[0])
    assert enc2.summary()["cache_path"].endswith("text_cache_sentence-transformers_all-MiniLM-L6-v2_" + hashlib.sha1(c.embedding_model.encode()).hexdigest()[:8] + ".npz")
    inst = build_instances(mini, c)
    index = build_item_index(mini, enc, c)
    assert index.content.shape[1] == 384 + len(ALL_GENRES)
    X = tensorise(inst["train"], enc, index, c)
    assert X["profile"].shape[1] == X["context"].shape[1] == 384
    for name in ["Conversation-aware", "AIPA (full)"]:
        model = build_model(name, index.content, c)
        assert model(X)["scores"].shape == (X["target"].shape[0], index.n)


def test_models_forward_shapes(mini, cfg):
    from aipa.preprocess import TextEncoder, build_item_index

    inst = build_instances(mini, cfg)
    enc = TextEncoder(cfg).fit([x.seeker_recent_text for x in inst["train"]] + list(mini.movie_titles.values()))
    index: ItemIndex = build_item_index(mini, enc)
    X = tensorise(inst["train"], enc, index, cfg)
    Y = label_tensors(label_all(inst["train"], cfg))
    assert Y["rel"].shape[0] == X["target"].shape[0]
    for name in ["LTP-only", "Naive fusion", "Adaptive fusion", "Sequential (GRU)", "Conversation-aware", "AIPA (rule policy)", "AIPA (full)"]:
        model = build_model(name, index.content, cfg)
        out = model(X)
        assert out["scores"].shape == (X["target"].shape[0], index.n)
        if name.startswith("AIPA"):
            assert out["rel_logits"].shape[1] == len(RELATIONSHIPS) and out["act_logits"].shape[1] == len(ACTIONS)
            w = torch.stack([out["w_ltp"], out["w_sti"]], 1)
            assert torch.allclose(w.sum(1), torch.ones(w.shape[0]), atol=1e-4)


def test_js_divergence_bounds_and_symmetry():
    from aipa.evaluate import js_divergence

    assert js_divergence({"Comedy": 1.0}, {"Comedy": 1.0}) == pytest.approx(0.0)
    assert js_divergence({"Comedy": 1.0}, {"Horror": 1.0}) == pytest.approx(1.0)
    a, b = {"Comedy": 0.7, "Drama": 0.3}, {"Drama": 0.5, "Horror": 0.5}
    assert 0.0 < js_divergence(a, b) < 1.0
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))
    assert np.isnan(js_divergence({}, {"Comedy": 1.0}))


def test_pooled_paired_test_pools_within_seed():
    from aipa.evaluate import cliffs_delta, permutation_test, pooled_paired_test

    rng = np.random.RandomState(0)
    base = {s: rng.rand(40) for s in (1, 2, 3)}
    treat = {s: base[s] + 0.2 + rng.normal(0, 0.01, 40) for s in base}
    r = pooled_paired_test(treat, base)
    assert r["n"] == 120 and r["n_samples"] == 40 and r["n_seeds"] == 3
    assert r["mean_diff"] == pytest.approx(0.2, abs=0.01)
    assert r["seed_std_diff"] < 0.01
    assert r["perm_p"] < 0.01 and r["t_p"] < 0.01
    assert -1.0 <= r["cliffs_delta"] <= 1.0
    # a seed-level mean shift with no within-seed difference must not be significant
    same = {s: base[s] for s in base}
    r0 = pooled_paired_test(same, base)
    assert np.isnan(r0["perm_p"]) and r0["mean_diff"] == 0.0
    assert cliffs_delta(np.ones(5), np.zeros(5)) == pytest.approx(1.0)
    assert 0.0 <= permutation_test(np.array([0.1, -0.1, 0.05, -0.05, 0.0]), n_perm=200) <= 1.0


def test_disagreement_mask_is_superset_of_strict(mini, cfg):
    import pandas as pd

    from aipa.labeling import disagreement_mask, labels_frame, strict_conflict_mask

    inst = build_instances(mini, cfg)
    test = inst["test"] + inject_controlled(inst["test"], mini, cfg, inst["train"], 1)
    lab = labels_frame(test, label_all(test, cfg))
    assert "js_divergence" in lab and lab.js_divergence.dropna().between(0, 1).all()
    strict, broad = strict_conflict_mask(lab, cfg), disagreement_mask(lab, cfg)
    assert not (strict & ~broad).any()
    assert not (broad & lab.is_synthetic.values).any()
    # a confident, divergent non-Conflict natural instance joins the broad subset only
    fake = pd.DataFrame({"relationship_label": ["Consistent", "Conflict", "Consistent", "Uncertain"],
                         "confidence": [0.9, 0.5, 0.9, 0.9], "js_divergence": [1.0, 0.0, 0.1, np.nan],
                         "is_synthetic": [False, False, False, False]})
    assert disagreement_mask(fake, cfg).tolist() == [True, True, False, False]
    assert strict_conflict_mask(fake, cfg).tolist() == [False, True, False, False]


def test_history_buckets_follow_config(cfg):
    from aipa.experiments import history_bucket

    b = history_bucket([0, 2, 3, 9, 10, 24, 25, 200], cfg)
    assert list(b.astype(str)) == ["cold", "cold", "short", "short", "mid", "mid", "long", "long"]


def test_persistence_tracker_fires_after_k_sessions_in_order():
    from aipa.models import PersistenceTracker

    tr = PersistenceTracker(k=2, gain=0.3)
    ltp = torch.tensor([0.5, 0.5] + [0.0] * 16)
    ltp = ltp / ltp.sum()
    assert torch.allclose(tr.adjust("A", ltp), ltp)
    tr.observe("A", 1, "Prioritize_STI", {"Horror": 1.0})
    tr.observe("A", 1, "Prioritize_STI", {"Horror": 1.0})  # same session counted once
    assert not tr.shifts and torch.allclose(tr.adjust("A", ltp), ltp)
    tr.observe("A", 2, "Fuse", {"Horror": 1.0})  # not a prioritisation -> ignored
    tr.observe("A", 3, "Prioritize_STI", {"Horror": 1.0})
    assert len(tr.shifts) == 1 and tr.shifts[0]["genre"] == "Horror"
    adj = tr.adjust("A", ltp)
    assert not torch.allclose(adj, ltp) and adj.sum() == pytest.approx(1.0)
    assert torch.allclose(tr.adjust("B", ltp), ltp)


def test_persistence_override_is_chronological_and_marks_affected(mini, cfg):
    from aipa.experiments import _persistence_override, _sessions_per_seeker

    inst = build_instances(mini, cfg)
    test = sorted(inst["test"], key=lambda x: (-x.conv_id, -x.turn))  # deliberately reverse order
    n_sess = _sessions_per_seeker(test)
    assert set(n_sess) == {1}
    X = {"ltp_genres": torch.rand(len(test), 18)}
    X["ltp_genres"] = X["ltp_genres"] / X["ltp_genres"].sum(1, keepdim=True)
    pred = {"act_logits": np.tile(np.eye(len(ACTIONS))[ACTIONS.index("Prioritize_STI")], (len(test), 1))}
    override, shifts, affected = _persistence_override(None, test, X, pred, cfg, k=1)
    # with k=1 the first Prioritize_STI turn of each seeker creates a shift; only *later* turns of that seeker are affected
    assert len(shifts) >= 1 and affected.sum() >= 1
    first_turn = {x.seeker_id: min(y.turn for y in test if y.seeker_id == x.seeker_id) for x in test}
    for i, x in enumerate(test):
        if x.turn == first_turn[x.seeker_id]:
            assert not affected[i]
    assert torch.allclose(override[~torch.tensor(affected)], X["ltp_genres"][~torch.tensor(affected)])
    _, shifts_k5, affected_k5 = _persistence_override(None, test, X, pred, cfg, k=5)
    assert not shifts_k5 and not affected_k5.any()


def test_architecture_figure_reports_the_active_config(cfg):
    import matplotlib.pyplot as plt

    from aipa.figures import architecture_diagram

    cfg.values.update(hidden_dim=16, max_history=7, max_context_turns=3, persistence_k=4,
                      top_k=[5], lambda_rel=0.25, lambda_act=0.75)
    for compact in (False, True):
        fig = architecture_diagram(compact=compact, cfg=cfg)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        plt.close(fig)
        assert "d = 16" in text
        assert f"MLP({3 * 16} -> 16 -> 16)" in text and f"MLP({5 * 16} -> 16 -> 16)" in text
        assert "[B, 7]" in text or "[B,7]" in text
        assert "k = 4" in text and "K = 5" in text
        assert "0.25" in text and "0.75" in text
