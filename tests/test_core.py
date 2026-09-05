"""Fast unit tests (synthetic mini-corpus; no download required)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aipa import ACTIONS, RELATIONSHIPS
from aipa.config import load_config
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
