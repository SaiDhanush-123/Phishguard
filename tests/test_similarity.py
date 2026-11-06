import pytest

from phishguard.reference import ReferenceDomain
from phishguard.similarity import SimilarityEngine, character_ngram_similarity, confusable_skeleton


REFERENCES = [
    ReferenceDomain("REF0001", "PayPal", "paypal.com", "paypal", "com", "finance", "global"),
    ReferenceDomain("REF0002", "Google", "google.com", "google", "com", "technology", "global"),
]


def test_exact_reference_is_not_a_lookalike():
    result = SimilarityEngine(REFERENCES).compare("https://paypal.com/login")

    assert result.match_type == "exact_reference"
    assert result.closeness_score == 100


def test_ascii_confusable_is_explained():
    result = SimilarityEngine(REFERENCES, threshold=0.75).compare("paypa1.com")

    assert result.match_type == "lookalike"
    assert result.closest_reference_domain == "paypal.com"
    assert "confusable character substitution" in result.evidence


def test_brand_hidden_in_deceptive_subdomain_is_detected():
    result = SimilarityEngine(REFERENCES, threshold=0.75).compare("paypal.security-check.example.com")

    assert result.match_type == "lookalike"
    assert "genuine brand appears only in a deceptive subdomain" in result.evidence


def test_unrelated_domain_is_not_forced_to_a_reference():
    result = SimilarityEngine(REFERENCES, threshold=0.80).compare("garden-tools.net")

    assert result.match_type == "no_close_match"
    assert result.closest_reference_domain is None
    assert result.diagnostic_reference_domain is not None


def test_similarity_primitives_and_weight_validation():
    assert confusable_skeleton("paypa1") == "paypal"
    assert character_ngram_similarity("paypal", "paypal") == 1
    with pytest.raises(ValueError, match="sum to 1"):
        SimilarityEngine(REFERENCES, weights={"levenshtein": 1, "jaro_winkler": 1, "ngram": 1, "confusable": 1})
