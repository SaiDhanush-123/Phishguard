import json
import socket
from pathlib import Path

import pandas as pd
import pytest
import requests

from phishguard.config import PROJECT_ROOT
from phishguard.detection import DetectionService, scan_batch
from phishguard.url_utils import _extractor


MODEL = PROJECT_ROOT / "models" / "url_model.joblib"
MODEL_METADATA = PROJECT_ROOT / "models" / "model_metadata.json"
REFERENCES = PROJECT_ROOT / "data" / "reference_domains.csv"
SIMILARITY_METADATA = PROJECT_ROOT / "models" / "similarity_metadata.json"
ARTIFACTS_READY = all(path.exists() for path in (MODEL, MODEL_METADATA, REFERENCES, SIMILARITY_METADATA))


@pytest.mark.skipif(not ARTIFACTS_READY, reason="production artifacts have not been generated")
def test_production_artifacts_scan_without_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("inference attempted a network call")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    _extractor.cache_clear()
    service = DetectionService(MODEL, MODEL_METADATA, REFERENCES, SIMILARITY_METADATA)

    lookalike = service.scan("paypa1-login.com")
    exact = service.scan("paypal.com")

    assert lookalike.valid and lookalike.risk_band == "High risk"
    assert lookalike.closest_reference_domain == "paypal.com"
    assert exact.match_type == "exact_reference"
    assert exact.risk_band == "Lower observed risk"


@pytest.mark.skipif(not ARTIFACTS_READY, reason="production artifacts have not been generated")
def test_batch_preserves_invalid_rows_and_output_schemas(tmp_path: Path):
    input_path = tmp_path / "input.csv"
    csv_output = tmp_path / "output.csv"
    json_output = tmp_path / "output.json"
    pd.DataFrame({"url": ["google.com", "not a valid URL", "paypa1-login.com"]}).to_csv(input_path, index=False)
    service = DetectionService(MODEL, MODEL_METADATA, REFERENCES, SIMILARITY_METADATA)

    summary = scan_batch(service, input_path, "URL", csv_output, json_output)
    csv_rows = pd.read_csv(csv_output)
    json_rows = json.loads(json_output.read_text(encoding="utf-8"))

    assert summary["input_rows"] == 3
    assert summary["invalid_rows"] == 1
    assert len(csv_rows) == len(json_rows) == 3
    assert set(["phishing_score", "closeness_score", "evidence", "risk_band"]).issubset(csv_rows.columns)
