from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .url_utils import normalize_url


REQUIRED_DATASET_COLUMNS = {"URL", "label"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    header = pd.read_csv(dataset_path, nrows=0)
    missing_columns = REQUIRED_DATASET_COLUMNS - set(header.columns)
    if missing_columns:
        raise ValueError(f"dataset is missing required columns: {sorted(missing_columns)}")

    frame = pd.read_csv(dataset_path, usecols=["URL", "label"])
    labels = set(frame["label"].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"unexpected PhiUSIIL labels: {sorted(labels)}")

    normalized = frame["URL"].map(normalize_url)
    domains = normalized.map(lambda item: item.registrable_domain if item.valid else "")
    valid_mask = normalized.map(lambda item: item.valid)
    working = pd.DataFrame({"domain": domains[valid_mask], "label": frame.loc[valid_mask, "label"]})
    conflicting_domains = int((working.groupby("domain")["label"].nunique() > 1).sum())

    report: dict[str, Any] = {
        "dataset": str(dataset_path.resolve()),
        "sha256": sha256_file(dataset_path),
        "row_count": int(len(frame)),
        "column_count": int(len(header.columns)),
        "label_mapping": {"raw_0": "phishing", "raw_1": "legitimate", "internal_positive": "is_phishing"},
        "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts(dropna=False).items()},
        "missing_url": int(frame["URL"].isna().sum()),
        "missing_label": int(frame["label"].isna().sum()),
        "duplicate_urls": int(frame["URL"].duplicated().sum()),
        "unique_urls": int(frame["URL"].nunique(dropna=True)),
        "valid_url_rows": int(valid_mask.sum()),
        "invalid_url_rows": int((~valid_mask).sum()),
        "unique_registrable_domains": int(domains[valid_mask].nunique()),
        "domains_with_conflicting_labels": conflicting_domains,
        "deployable_columns": ["URL", "label"],
        "excluded_column_count": int(len(header.columns) - 2),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return report
