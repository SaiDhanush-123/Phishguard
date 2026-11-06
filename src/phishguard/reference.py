from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_audit import sha256_file
from .url_utils import normalize_url


REFERENCE_COLUMNS = [
    "reference_id",
    "brand_name",
    "canonical_domain",
    "category",
    "region",
    "source_name",
    "source_url",
    "source_rank",
    "verified_at",
    "verification_status",
    "active",
]


@dataclass(frozen=True)
class ReferenceDomain:
    reference_id: str
    brand_name: str
    canonical_domain: str
    domain_label: str
    suffix: str
    category: str
    region: str


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_reference_registry(
    path: str | Path,
    output_path: str | Path | None = None,
    expected_count: int | None = 200,
) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REFERENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"reference registry is missing columns: {sorted(missing)}")

    errors: list[str] = []
    active_verified: list[str] = []
    seen_ids: set[str] = set()
    seen_domains: set[str] = set()
    categories: dict[str, int] = {}
    regions: dict[str, int] = {}

    for row_number, row in frame.iterrows():
        display_row = row_number + 2
        reference_id = row["reference_id"].strip()
        domain = row["canonical_domain"].strip().lower().rstrip(".")
        if not reference_id or reference_id in seen_ids:
            errors.append(f"row {display_row}: missing or duplicate reference_id")
        seen_ids.add(reference_id)
        if not row["brand_name"].strip():
            errors.append(f"row {display_row}: brand_name is empty")
        normalized = normalize_url(domain)
        if not normalized.valid or normalized.is_ip:
            errors.append(f"row {display_row}: invalid canonical domain {domain!r}")
        elif normalized.registrable_domain != domain or normalized.ascii_hostname != domain:
            errors.append(f"row {display_row}: canonical_domain must be a registrable domain")
        if domain in seen_domains:
            errors.append(f"row {display_row}: duplicate canonical domain {domain!r}")
        seen_domains.add(domain)
        if row["source_url"] and not row["source_url"].lower().startswith("https://"):
            errors.append(f"row {display_row}: source_url must use HTTPS")
        active = _as_bool(row["active"])
        verified = row["verification_status"].strip().lower() == "verified"
        if active and verified and normalized.valid:
            active_verified.append(domain)
            category = row["category"].strip().lower()
            region = row["region"].strip().lower()
            categories[category] = categories.get(category, 0) + 1
            regions[region] = regions.get(region, 0) + 1
        elif active:
            errors.append(f"row {display_row}: active rows must be verified")

    if expected_count is not None and len(active_verified) != expected_count:
        errors.append(f"expected {expected_count} active verified domains, found {len(active_verified)}")

    report: dict[str, Any] = {
        "registry": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "active_verified_count": len(active_verified),
        "unique_domain_count": len(seen_domains),
        "categories": dict(sorted(categories.items())),
        "regions": dict(sorted(regions.items())),
        "errors": errors,
        "valid": not errors,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if errors:
        raise ValueError("invalid reference registry:\n" + "\n".join(errors[:25]))
    return report


def load_references(path: str | Path, validate: bool = True) -> list[ReferenceDomain]:
    if validate:
        validate_reference_registry(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    references: list[ReferenceDomain] = []
    for _, row in frame.iterrows():
        if not _as_bool(row["active"]) or row["verification_status"].strip().lower() != "verified":
            continue
        normalized = normalize_url(row["canonical_domain"])
        references.append(
            ReferenceDomain(
                reference_id=row["reference_id"].strip(),
                brand_name=row["brand_name"].strip(),
                canonical_domain=normalized.registrable_domain,
                domain_label=normalized.domain_label,
                suffix=normalized.suffix,
                category=row["category"].strip(),
                region=row["region"].strip(),
            )
        )
    return references


def registry_version(path: str | Path) -> str:
    return f"sha256:{sha256_file(path)[:16]}"
