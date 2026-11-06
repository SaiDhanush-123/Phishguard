from app import app, risk_style, serialize_result
from phishguard.detection import DetectionService
from phishguard.config import CONFIG, project_path


def _service() -> DetectionService:
    paths = CONFIG["paths"]
    return DetectionService(
        project_path(paths["model"]),
        project_path(paths["model_metadata"]),
        project_path(paths["references"]),
        project_path(paths["similarity_metadata"]),
    )


def test_index_page_renders_all_primary_sections():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "PhishGuard AI" in body
    assert "Single scan" in body
    assert "Batch scan" in body
    assert "Methodology" in body
    assert "Team" in body
    assert 'id="scanForm"' in body


def test_api_scan_flags_a_lookalike_domain_as_high_risk():
    client = app.test_client()
    response = client.post("/api/scan", json={"url": "paypa1-login.com"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["valid"] is True
    assert data["risk_band"] == "High risk"
    assert data["risk_class"] == "high"
    assert any("suspicious" in signal["label"] or "paypal" in signal["label"].lower() for signal in data["signals"])


def test_api_scan_exact_reference_is_not_flagged_as_high_risk():
    client = app.test_client()
    response = client.post("/api/scan", json={"url": "paypal.com"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["valid"] is True
    assert data["match_type"] == "exact_reference"
    assert data["risk_band"] == "Lower observed risk"
    assert data["closeness"] == 100.0


def test_api_scan_rejects_empty_url():
    client = app.test_client()
    response = client.post("/api/scan", json={"url": "   "})

    assert response.status_code == 400
    data = response.get_json()
    assert data["valid"] is False
    assert "Enter a URL" in data["error"]


def test_api_scan_reports_validation_errors_for_invalid_input():
    client = app.test_client()
    response = client.post("/api/scan", json={"url": "not a valid url"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["valid"] is False
    assert data["error"]


def test_api_batch_preserves_order_and_flags_invalid_rows():
    client = app.test_client()
    response = client.post(
        "/api/batch",
        json={"urls": ["google.com", "paypa1-login.com", "not a valid URL"]},
    )

    assert response.status_code == 200
    data = response.get_json()
    rows = data["rows"]
    assert len(rows) == 3
    assert rows[0]["input"] == "google.com"
    assert rows[1]["risk_band"] == "High risk"
    assert rows[2]["valid"] is False


def test_api_batch_requires_a_list():
    client = app.test_client()
    response = client.post("/api/batch", json={"urls": "not-a-list"})

    assert response.status_code == 400


def test_risk_style_maps_bands_to_css_classes():
    assert risk_style("High risk") == "high"
    assert risk_style("Caution") == "caution"
    assert risk_style("Lower observed risk") == ""


def test_serialize_result_reports_invalid_input():
    service = _service()
    result = service.scan("not a valid url")
    payload = serialize_result(result)
    assert payload["valid"] is False
    assert payload["error"]


def test_serialize_result_includes_similarity_metrics_for_valid_input():
    service = _service()
    result = service.scan("google.com")
    payload = serialize_result(result)
    assert payload["valid"] is True
    assert set(payload["metrics"]) == {"levenshtein", "jaro_winkler", "ngram", "confusable"}
    assert payload["signals"]
