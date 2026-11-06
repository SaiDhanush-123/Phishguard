from phishguard.detection import determine_risk_band
from phishguard.similarity import SimilarityResult


def _similarity(match_type: str, score: float) -> SimilarityResult:
    return SimilarityResult(
        match_type=match_type,
        closest_reference_domain="example.com" if match_type != "no_close_match" else None,
        closest_brand="Example" if match_type != "no_close_match" else None,
        diagnostic_reference_domain="example.com",
        closeness_score=score,
        component_scores={},
        evidence=[],
        threshold=0.4,
        registry_version="test",
    )


def test_exact_reference_prevents_similarity_or_ml_false_alarm():
    assert determine_risk_band(0.9, 0.2, 0.1, _similarity("exact_reference", 100), 0.36) == "Lower observed risk"


def test_risk_policy_keeps_scores_independent():
    assert determine_risk_band(0.8, 0.2, 0.1, _similarity("no_close_match", 10), 0.36) == "High risk"
    assert determine_risk_band(0.01, 0.2, 0.1, _similarity("lookalike", 50), 0.36) == "High risk"
    assert determine_risk_band(0.05, 0.2, 0.1, _similarity("no_close_match", 37), 0.36) == "Caution"
    assert determine_risk_band(0.01, 0.2, 0.1, _similarity("no_close_match", 20), 0.36) == "Lower observed risk"
