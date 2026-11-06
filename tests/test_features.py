from phishguard.features import FEATURE_COLUMNS, extract_url_features, feature_schema


def test_feature_contract_is_complete_and_deterministic():
    first = extract_url_features("https://secure-login.example.com/a?verify=1")
    second = extract_url_features("https://secure-login.example.com/a?verify=1")

    assert first == second
    assert list(first) == FEATURE_COLUMNS
    assert first["is_https"] == 1
    assert first["subdomain_count"] == 1
    assert first["suspicious_token_count"] >= 2
    assert feature_schema()["all"] == FEATURE_COLUMNS


def test_repeated_character_feature():
    features = extract_url_features("https://aaabbb.example.com")

    assert features["max_repeated_run"] == 3
