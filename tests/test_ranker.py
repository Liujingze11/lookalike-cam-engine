from engine.retrievers.base import Candidate
from engine.rankers.simple_ranker import SimpleRanker
from engine.config import Preset, SafetyConfig


def make_candidate(char_id: str, sim: float) -> Candidate:
    return Candidate(
        character_id=char_id,
        embedding_type="global",
        similarity=sim,
        index=0,
    )


def test_simple_ranker_sorts_by_similarity():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("mario", 0.65),
        make_candidate("luigi", 0.92),
        make_candidate("peach", 0.78),
    ]
    names = {"mario": "Mario", "luigi": "Luigi", "peach": "Peach"}
    safety_map = {"mario": True, "luigi": True, "peach": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
        top_k=5,
    )

    assert len(results) == 3
    assert results[0].character_id == "luigi"
    assert results[0].score == 0.92
    assert results[1].character_id == "peach"
    assert results[2].character_id == "mario"


def test_simple_ranker_deduplicates():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("mario", 0.9),
        make_candidate("mario", 0.7),
        make_candidate("mario", 0.8),
    ]
    names = {"mario": "Mario"}
    safety_map = {"mario": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
    )

    assert len(results) == 1
    assert results[0].score == 0.9


def test_simple_ranker_filters_unsafe():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("safe", 0.9),
        make_candidate("unsafe", 0.95),
    ]
    names = {"safe": "Safe", "unsafe": "Unsafe"}
    safety_map = {"safe": True, "unsafe": False}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
    )

    assert len(results) == 1
    assert results[0].character_id == "safe"


def test_simple_ranker_respects_min_score():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("a", 0.3),
        make_candidate("b", 0.8),
    ]
    names = {"a": "A", "b": "B"}
    safety_map = {"a": True, "b": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(min_final_score=0.5),
    )

    assert len(results) == 1
    assert results[0].character_id == "b"
