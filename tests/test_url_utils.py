from phishguard.url_utils import normalize_url


def test_bare_domain_is_parsed_without_claiming_supplied_scheme():
    result = normalize_url("Example.COM/path")

    assert result.valid
    assert result.registrable_domain == "example.com"
    assert result.ascii_hostname == "example.com"
    assert result.path == "/path"
    assert not result.supplied_scheme


def test_deep_subdomain_uses_actual_registrable_domain():
    result = normalize_url("https://paypal.com.security-check.example.org/login")

    assert result.valid
    assert result.registrable_domain == "example.org"
    assert result.subdomain == "paypal.com.security-check"


def test_unicode_hostname_retains_unicode_and_ascii_forms():
    result = normalize_url("https://раypal.com")

    assert result.valid
    assert result.has_punycode
    assert result.ascii_hostname.startswith("xn--")
    assert result.unicode_hostname == "раypal.com"


def test_ip_host_is_valid_but_has_no_suffix():
    result = normalize_url("http://192.0.2.10/login")

    assert result.valid
    assert result.is_ip
    assert result.registrable_domain == "192.0.2.10"
    assert result.suffix == ""


def test_invalid_inputs_return_structured_error():
    assert not normalize_url("").valid
    assert not normalize_url("localhost").valid
    assert not normalize_url("javascript:alert(1)").valid
