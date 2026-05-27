from app.schemas.requests import MatchRequest
from app.schemas.responses import MatchResponse, MatchResultItem


def test_match_request_defaults():
    req = MatchRequest(image_base64="aaa")
    assert req.preset == "default"
    assert req.top_k == 5


def test_match_request_validation():
    req = MatchRequest(image_base64="bbb", preset="anime_vibe", top_k=10)
    assert req.preset == "anime_vibe"
    assert req.top_k == 10


def test_match_response_serialization():
    resp = MatchResponse(
        query_id="q_001",
        results=[
            MatchResultItem(
                rank=1,
                character_id="mario",
                name="Mario",
                score=0.87,
                feature_scores={"global": 0.87},
            )
        ],
    )
    data = resp.model_dump()
    assert data["query_id"] == "q_001"
    assert len(data["results"]) == 1
    assert data["results"][0]["character_id"] == "mario"
