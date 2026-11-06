from phishguard.reference import ReferenceDomain
import pandas as pd
import pytest

from phishguard.synthetic import _mutations, validate_synthetic_frame


def test_mutations_cover_required_attack_types():
    reference = ReferenceDomain("REF0001", "PayPal", "paypal.com", "paypal", "com", "finance", "global")

    transformations = {name for name, _, _ in _mutations(reference)}

    assert transformations == {
        "deletion",
        "insertion",
        "substitution",
        "transposition",
        "ascii_confusable",
        "unicode_confusable",
        "prefix_suffix",
        "alternate_tld",
        "deceptive_subdomain",
    }


def test_synthetic_validation_rejects_reference_leakage():
    rows = []
    for index, transformation in enumerate(
        [
            "deletion",
            "insertion",
            "substitution",
            "transposition",
            "ascii_confusable",
            "unicode_confusable",
            "prefix_suffix",
            "alternate_tld",
            "deceptive_subdomain",
        ]
    ):
        rows.append(
            {
                "pair_id": f"P{index}",
                "reference_id": "REF1",
                "reference_domain": "paypal.com",
                "generated_hostname": f"example{index}.com",
                "generated_registrable_domain": f"example{index}.com",
                "transformation_type": transformation,
                "is_synthetic_lookalike": 1,
                "generation_seed": 42,
                "split": "tuning" if index != 8 else "test",
            }
        )
    rows.append({**rows[0], "pair_id": "NEG", "is_synthetic_lookalike": 0})

    with pytest.raises(ValueError, match="more than one split"):
        validate_synthetic_frame(pd.DataFrame(rows))
