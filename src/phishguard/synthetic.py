from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .config import SEED
from .reference import ReferenceDomain, load_references
from .url_utils import normalize_url


ASCII_CONFUSABLES = {
    "a": "4",
    "b": "8",
    "e": "3",
    "g": "9",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
}
UNICODE_CONFUSABLES = {
    "a": "а",
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "s": "ѕ",
    "x": "х",
    "y": "у",
}
REQUIRED_POSITIVE_TRANSFORMATIONS = {
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


def _replace_first(value: str, mapping: dict[str, str], fallback: str) -> str:
    for index, character in enumerate(value):
        if character in mapping:
            return f"{value[:index]}{mapping[character]}{value[index + 1:]}"
    return f"{fallback}{value[1:]}" if value else fallback


def _transpose(value: str) -> str:
    for index in range(max(0, len(value) - 1)):
        if value[index] != value[index + 1]:
            return f"{value[:index]}{value[index + 1]}{value[index]}{value[index + 2:]}"
    return f"{value}x"


def _substitute(value: str) -> str:
    if not value:
        return "x"
    index = len(value) // 2
    replacement = "x" if value[index] != "x" else "z"
    return f"{value[:index]}{replacement}{value[index + 1:]}"


def _mutations(reference: ReferenceDomain) -> list[tuple[str, str, str]]:
    stem = reference.domain_label
    suffix = reference.suffix
    middle = max(1, len(stem) // 2)
    deletion_stem = f"{stem[:middle]}{stem[middle + 1:]}" if len(stem) > 3 else f"{stem}x"
    insertion_stem = f"{stem[:middle]}x{stem[middle:]}"
    unicode_stem = _replace_first(stem, UNICODE_CONFUSABLES, "а")
    alternate_tld = "net" if suffix != "net" else "org"
    return [
        ("deletion", f"{deletion_stem}.{suffix}", "one character deleted"),
        ("insertion", f"{insertion_stem}.{suffix}", "one character inserted"),
        ("substitution", f"{_substitute(stem)}.{suffix}", "one character substituted"),
        ("transposition", f"{_transpose(stem)}.{suffix}", "adjacent characters transposed"),
        (
            "ascii_confusable",
            f"{_replace_first(stem, ASCII_CONFUSABLES, '1')}.{suffix}",
            "ASCII digit/letter confusable",
        ),
        ("unicode_confusable", f"{unicode_stem}.{suffix}", "Unicode confusable character"),
        ("prefix_suffix", f"secure-{stem}-login.{suffix}", "security/login tokens added"),
        ("alternate_tld", f"{stem}.{alternate_tld}", "top-level domain changed"),
        (
            "deceptive_subdomain",
            f"{stem}.security-check.example.com",
            "brand appears in a subdomain of another registrable domain",
        ),
    ]


def _split_map(references: list[ReferenceDomain], seed: int) -> dict[str, str]:
    identifiers = [reference.reference_id for reference in references]
    random.Random(seed).shuffle(identifiers)
    tuning_end = int(len(identifiers) * 0.60)
    validation_end = tuning_end + int(len(identifiers) * 0.20)
    result: dict[str, str] = {}
    for index, identifier in enumerate(identifiers):
        result[identifier] = "tuning" if index < tuning_end else "validation" if index < validation_end else "test"
    return result


def validate_synthetic_frame(frame: pd.DataFrame) -> dict[str, object]:
    required_columns = {
        "pair_id",
        "reference_id",
        "reference_domain",
        "generated_hostname",
        "generated_registrable_domain",
        "transformation_type",
        "is_synthetic_lookalike",
        "generation_seed",
        "split",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"synthetic dataset is missing columns: {sorted(missing)}")
    errors: list[str] = []
    if frame["pair_id"].duplicated().any():
        errors.append("pair_id values are not unique")
    labels = set(frame["is_synthetic_lookalike"].astype(int).unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        errors.append("synthetic labels must contain both 0 and 1")
    split_counts = frame.groupby("reference_id")["split"].nunique()
    if (split_counts != 1).any():
        errors.append("a reference_id occurs in more than one split")
    valid_splits = set(frame["split"].astype(str).unique())
    if valid_splits != {"tuning", "validation", "test"}:
        errors.append(f"unexpected splits: {sorted(valid_splits)}")
    positives = frame[frame["is_synthetic_lookalike"].astype(int) == 1]
    coverage = positives.groupby("reference_id")["transformation_type"].agg(set)
    missing_coverage = {
        reference_id: sorted(REQUIRED_POSITIVE_TRANSFORMATIONS - transformations)
        for reference_id, transformations in coverage.items()
        if REQUIRED_POSITIVE_TRANSFORMATIONS - transformations
    }
    if missing_coverage:
        errors.append(f"positive transformation coverage is incomplete for {len(missing_coverage)} references")
    if errors:
        raise ValueError("invalid synthetic dataset: " + "; ".join(errors))
    return {
        "valid": True,
        "reference_group_leakage": False,
        "reference_count": int(frame["reference_id"].nunique()),
        "positive_transformation_count": len(REQUIRED_POSITIVE_TRANSFORMATIONS),
    }


def generate_synthetic_pairs(
    reference_path: str | Path,
    output_path: str | Path,
    seed: int = SEED,
) -> dict[str, object]:
    references = load_references(reference_path)
    if len(references) < 2:
        raise ValueError("at least two reference domains are required")
    split_by_reference = _split_map(references, seed)
    reference_domains = {reference.canonical_domain for reference in references}
    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    def add_row(
        reference: ReferenceDomain,
        hostname: str,
        transformation: str,
        label: int,
        evidence: str,
    ) -> None:
        normalized = normalize_url(hostname)
        if not normalized.valid:
            raise ValueError(f"generated invalid hostname {hostname!r}: {normalized.error}")
        key = (reference.reference_id, normalized.ascii_hostname, transformation)
        if key in seen_pairs:
            return
        if label == 1 and normalized.registrable_domain in reference_domains:
            if normalized.registrable_domain != reference.canonical_domain:
                return
            if normalized.ascii_hostname == reference.canonical_domain:
                return
        seen_pairs.add(key)
        rows.append(
            {
                "pair_id": "",
                "reference_id": reference.reference_id,
                "brand_name": reference.brand_name,
                "reference_domain": reference.canonical_domain,
                "generated_hostname": hostname,
                "generated_ascii_hostname": normalized.ascii_hostname,
                "generated_registrable_domain": normalized.registrable_domain,
                "transformation_type": transformation,
                "is_synthetic_lookalike": label,
                "expected_evidence": evidence,
                "generation_seed": seed,
                "split": split_by_reference[reference.reference_id],
            }
        )

    for index, reference in enumerate(references):
        for transformation, hostname, evidence in _mutations(reference):
            add_row(reference, hostname, transformation, 1, evidence)

        add_row(reference, reference.canonical_domain, "exact_reference", 0, "exact genuine domain")
        add_row(reference, f"www.{reference.canonical_domain}", "legitimate_subdomain", 0, "genuine subdomain")
        add_row(reference, f"neutral-{index}.example.org", "unrelated", 0, "unrelated reserved example domain")
        for offset in range(1, 7):
            other = references[(index + offset) % len(references)]
            add_row(reference, other.canonical_domain, f"other_reference_{offset}", 0, "different genuine reference")

    for index, row in enumerate(rows, start=1):
        row["pair_id"] = f"PAIR{index:06d}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    validation = validate_synthetic_frame(frame)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    summary = {
        "output": str(output_path.resolve()),
        "row_count": int(len(frame)),
        "positive_count": int((frame["is_synthetic_lookalike"] == 1).sum()),
        "negative_count": int((frame["is_synthetic_lookalike"] == 0).sum()),
        "reference_count": len(references),
        "split_counts": {key: int(value) for key, value in frame["split"].value_counts().items()},
        "transformation_counts": {
            key: int(value) for key, value in frame["transformation_type"].value_counts().items()
        },
        "seed": seed,
        "validation": validation,
    }
    summary_path = output_path.with_suffix(".metadata.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
