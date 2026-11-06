from pathlib import Path

import pandas as pd
import pytest

from phishguard.reference import REFERENCE_COLUMNS, validate_reference_registry


def _row(identifier: str, brand: str, domain: str) -> dict[str, str]:
    return {
        "reference_id": identifier,
        "brand_name": brand,
        "canonical_domain": domain,
        "category": "technology",
        "region": "global",
        "source_name": "official-site",
        "source_url": f"https://{domain}",
        "source_rank": "",
        "verified_at": "2026-11-04",
        "verification_status": "verified",
        "active": "true",
    }


def test_reference_registry_validation(tmp_path: Path):
    path = tmp_path / "references.csv"
    pd.DataFrame([_row("REF0001", "Example", "example.com"), _row("REF0002", "IANA", "iana.org")]).to_csv(
        path, index=False
    )

    report = validate_reference_registry(path, expected_count=2)

    assert report["valid"]
    assert report["active_verified_count"] == 2


def test_reference_registry_rejects_subdomain_and_duplicate(tmp_path: Path):
    path = tmp_path / "references.csv"
    pd.DataFrame(
        [_row("REF0001", "Example", "www.example.com"), _row("REF0002", "Duplicate", "www.example.com")],
        columns=REFERENCE_COLUMNS,
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid reference registry"):
        validate_reference_registry(path, expected_count=2)
