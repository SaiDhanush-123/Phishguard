from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .config import CONFIG, FEATURE_SCHEMA_VERSION
from .url_utils import NormalizedURL, normalize_url


NUMERIC_FEATURES = [
    "url_length",
    "hostname_length",
    "registrable_length",
    "path_length",
    "query_length",
    "fragment_length",
    "letter_count",
    "letter_ratio",
    "digit_count",
    "digit_ratio",
    "hyphen_count",
    "dot_count",
    "at_count",
    "percent_count",
    "equals_count",
    "question_count",
    "ampersand_count",
    "underscore_count",
    "special_count",
    "special_ratio",
    "subdomain_count",
    "label_count",
    "max_label_length",
    "is_ip",
    "is_https",
    "has_punycode",
    "has_port",
    "hostname_entropy",
    "suspicious_token_count",
    "max_repeated_run",
    "double_slash_in_path",
]
CATEGORICAL_FEATURES = ["tld"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _ratio(count: int, length: int) -> float:
    return float(count / length) if length else 0.0


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _max_repeated_run(value: str) -> int:
    matches = re.findall(r"(.)\1*", value)
    if not matches:
        return 0
    longest = 1
    current = 1
    for previous, current_char in zip(value, value[1:]):
        if current_char == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def extract_url_features(value: object, normalized: NormalizedURL | None = None) -> dict[str, Any]:
    normalized = normalized or normalize_url(value)
    if not normalized.valid:
        raise ValueError(normalized.error or "invalid URL")

    raw = normalized.original.lower()
    hostname = normalized.ascii_hostname
    raw_length = len(raw)
    letters = sum(char.isalpha() for char in raw)
    digits = sum(char.isdigit() for char in raw)
    special = sum(not char.isalnum() for char in raw)
    labels = hostname.split(".")
    tokens = re.findall(r"[a-z0-9]+", raw)
    suspicious = set(CONFIG["features"]["suspicious_tokens"])

    return {
        "url_length": raw_length,
        "hostname_length": len(hostname),
        "registrable_length": len(normalized.registrable_domain),
        "path_length": len(normalized.path),
        "query_length": len(normalized.query),
        "fragment_length": len(normalized.fragment),
        "letter_count": letters,
        "letter_ratio": _ratio(letters, raw_length),
        "digit_count": digits,
        "digit_ratio": _ratio(digits, raw_length),
        "hyphen_count": raw.count("-"),
        "dot_count": raw.count("."),
        "at_count": raw.count("@"),
        "percent_count": raw.count("%"),
        "equals_count": raw.count("="),
        "question_count": raw.count("?"),
        "ampersand_count": raw.count("&"),
        "underscore_count": raw.count("_"),
        "special_count": special,
        "special_ratio": _ratio(special, raw_length),
        "subdomain_count": len(normalized.subdomain.split(".")) if normalized.subdomain else 0,
        "label_count": len(labels),
        "max_label_length": max(map(len, labels), default=0),
        "is_ip": int(normalized.is_ip),
        "is_https": int(normalized.scheme == "https"),
        "has_punycode": int(normalized.has_punycode),
        "has_port": int(normalized.port is not None),
        "hostname_entropy": _entropy(hostname),
        "suspicious_token_count": sum(token in suspicious for token in tokens),
        "max_repeated_run": _max_repeated_run(hostname),
        "double_slash_in_path": int("//" in normalized.path),
        "tld": normalized.suffix or "__ip__",
    }


def build_feature_frame(values: Iterable[object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for value in values:
        normalized = normalize_url(value)
        metadata_rows.append(
            {
                "valid": normalized.valid,
                "error": normalized.error,
                "normalized_host": normalized.ascii_hostname,
                "registrable_domain": normalized.registrable_domain,
            }
        )
        if normalized.valid:
            feature_rows.append(extract_url_features(value, normalized))
        else:
            feature_rows.append({column: ("__invalid__" if column == "tld" else 0) for column in FEATURE_COLUMNS})
    return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS), pd.DataFrame(metadata_rows)


def feature_schema() -> dict[str, Any]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "all": FEATURE_COLUMNS,
    }
