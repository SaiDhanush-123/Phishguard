from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rapidfuzz.distance import JaroWinkler, Levenshtein

from .config import CONFIG
from .reference import ReferenceDomain, load_references, registry_version
from .url_utils import NormalizedURL, normalize_url


_SKELETON_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "8": "b",
        "9": "g",
        "а": "a",
        "с": "c",
        "е": "e",
        "і": "i",
        "о": "o",
        "р": "p",
        "ѕ": "s",
        "х": "x",
        "у": "y",
    }
)


def confusable_skeleton(value: str) -> str:
    return value.lower().translate(_SKELETON_TRANSLATION).replace("rn", "m")


def _character_ngrams(value: str) -> set[str]:
    result: set[str] = set()
    padded = f"^{value}$"
    for size in (2, 3, 4):
        result.update(padded[index : index + size] for index in range(max(0, len(padded) - size + 1)))
    return result


def _ngram_jaccard(left_grams: set[str], right_grams: set[str]) -> float:
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 1.0


def character_ngram_similarity(left: str, right: str) -> float:
    return _ngram_jaccard(_character_ngrams(left), _character_ngrams(right))


def _adjacent_transposition(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    differences = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
    return (
        len(differences) == 2
        and differences[1] == differences[0] + 1
        and left[differences[0]] == right[differences[1]]
        and left[differences[1]] == right[differences[0]]
    )


@dataclass(frozen=True)
class SimilarityResult:
    match_type: str
    closest_reference_domain: str | None
    closest_brand: str | None
    diagnostic_reference_domain: str | None
    closeness_score: float
    component_scores: dict[str, float]
    evidence: list[str]
    threshold: float
    registry_version: str

    def to_dict(self, include_diagnostic: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_diagnostic:
            result.pop("diagnostic_reference_domain", None)
        return result


class SimilarityEngine:
    def __init__(
        self,
        references: list[ReferenceDomain],
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
        version: str = "in-memory",
    ) -> None:
        if not references:
            raise ValueError("similarity engine requires at least one reference domain")
        self.references = references
        self.reference_by_domain = {reference.canonical_domain: reference for reference in references}
        self.reference_profiles = {
            reference.reference_id: {
                "canonical_grams": _character_ngrams(reference.canonical_domain),
                "label_grams": _character_ngrams(reference.domain_label),
                "skeleton": confusable_skeleton(reference.domain_label),
            }
            for reference in references
        }
        configured = CONFIG["similarity"]
        self.weights = dict(weights or configured["weights"])
        if set(self.weights) != {"levenshtein", "jaro_winkler", "ngram", "confusable"}:
            raise ValueError("similarity weights have an invalid schema")
        if any(value < 0 for value in self.weights.values()) or abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("similarity weights must be non-negative and sum to 1")
        self.threshold = float(threshold if threshold is not None else configured["threshold"])
        self.version = version

    @classmethod
    def from_files(
        cls,
        reference_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> "SimilarityEngine":
        weights = None
        threshold = None
        if metadata_path is not None and Path(metadata_path).exists():
            with Path(metadata_path).open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            weights = metadata.get("weights")
            threshold = metadata.get("threshold")
        return cls(
            load_references(reference_path),
            weights=weights,
            threshold=threshold,
            version=registry_version(reference_path),
        )

    @staticmethod
    def _candidate_labels(normalized: NormalizedURL) -> list[str]:
        labels = [normalized.domain_label]
        suffix_parts = normalized.suffix.split(".") if normalized.suffix else []
        ascii_parts = normalized.ascii_hostname.split(".")
        unicode_parts = normalized.unicode_hostname.split(".")
        keep = max(1, len(ascii_parts) - len(suffix_parts))
        labels.extend(ascii_parts[:keep])
        labels.extend(unicode_parts[:keep])
        return list(dict.fromkeys(label.lower() for label in labels if label))

    def candidate_profile(self, normalized: NormalizedURL) -> dict[str, Any]:
        labels = self._candidate_labels(normalized)
        return {
            "labels": labels,
            "label_grams": [_character_ngrams(label) for label in labels],
            "label_skeletons": [confusable_skeleton(label) for label in labels],
            "registrable_grams": _character_ngrams(normalized.registrable_domain),
        }

    def component_scores(
        self,
        normalized: NormalizedURL,
        reference: ReferenceDomain,
        candidate_profile: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], str]:
        candidate_profile = candidate_profile or self.candidate_profile(normalized)
        candidate_labels = candidate_profile["labels"]
        string_pairs = [(normalized.registrable_domain, reference.canonical_domain)]
        string_pairs.extend((label, reference.domain_label) for label in candidate_labels)
        levenshtein = max(float(Levenshtein.normalized_similarity(left, right)) for left, right in string_pairs)
        jaro = max(float(JaroWinkler.normalized_similarity(left, right)) for left, right in string_pairs)
        reference_profile = self.reference_profiles[reference.reference_id]
        ngram = _ngram_jaccard(candidate_profile["registrable_grams"], reference_profile["canonical_grams"])
        ngram = max(
            ngram,
            *(
                _ngram_jaccard(label_grams, reference_profile["label_grams"])
                for label_grams in candidate_profile["label_grams"]
            ),
        )

        reference_skeleton = reference_profile["skeleton"]
        confusable = 0.0
        best_label = max(candidate_labels, key=lambda label: JaroWinkler.normalized_similarity(label, reference.domain_label))
        for label, skeleton in zip(candidate_labels, candidate_profile["label_skeletons"]):
            if skeleton == reference_skeleton and label != reference.domain_label:
                confusable = max(confusable, 1.0)
            elif reference_skeleton in skeleton and label != reference.domain_label:
                confusable = max(confusable, 0.8)
            elif any(char.isdigit() or ord(char) > 127 for char in label) and levenshtein >= 0.75:
                confusable = max(confusable, 0.7)
        if normalized.registrable_domain != reference.canonical_domain:
            subdomain_labels = normalized.subdomain.split(".") if normalized.subdomain else []
            if reference.domain_label in subdomain_labels or reference.domain_label in candidate_labels[:-1]:
                confusable = 1.0
        return {
            "levenshtein": levenshtein,
            "jaro_winkler": jaro,
            "ngram": ngram,
            "confusable": confusable,
        }, best_label

    def _evidence(
        self,
        normalized: NormalizedURL,
        reference: ReferenceDomain,
        best_label: str,
        components: dict[str, float],
    ) -> list[str]:
        evidence: list[str] = []
        source = reference.domain_label
        target = best_label
        if normalized.has_punycode or any(ord(char) > 127 for char in normalized.unicode_hostname):
            evidence.append("internationalized/Punycode hostname")
        target_skeleton = confusable_skeleton(target)
        source_skeleton = confusable_skeleton(source)
        if source_skeleton in target_skeleton and target != source:
            evidence.append("confusable character substitution")
        distance = int(Levenshtein.distance(source, target))
        if _adjacent_transposition(source, target):
            evidence.append("adjacent character transposition")
        elif distance == 1 and len(target) == len(source):
            evidence.append("single-character substitution")
        elif distance == 1 and len(target) > len(source):
            evidence.append("single-character insertion")
        elif distance == 1 and len(target) < len(source):
            evidence.append("single-character deletion")
        if source_skeleton in target_skeleton and target_skeleton != source_skeleton:
            evidence.append("brand-related prefix or suffix added")
        if normalized.suffix != reference.suffix and components["jaro_winkler"] >= 0.8:
            evidence.append(f"top-level domain changed from .{reference.suffix} to .{normalized.suffix}")
        subdomain_labels = normalized.subdomain.split(".") if normalized.subdomain else []
        if normalized.registrable_domain != reference.canonical_domain and source in subdomain_labels:
            evidence.append("genuine brand appears only in a deceptive subdomain")
        if not evidence and components["jaro_winkler"] >= 0.85:
            evidence.append("high lexical similarity to the reference domain")
        return list(dict.fromkeys(evidence))

    def compare(self, value: object) -> SimilarityResult:
        normalized = normalize_url(value)
        if not normalized.valid or normalized.is_ip:
            return SimilarityResult(
                match_type="not_applicable",
                closest_reference_domain=None,
                closest_brand=None,
                diagnostic_reference_domain=None,
                closeness_score=0.0,
                component_scores={key: 0.0 for key in self.weights},
                evidence=[normalized.error or "IP hosts are not compared with brand domains"],
                threshold=self.threshold,
                registry_version=self.version,
            )

        exact = self.reference_by_domain.get(normalized.registrable_domain)
        if exact is not None:
            return SimilarityResult(
                match_type="exact_reference",
                closest_reference_domain=exact.canonical_domain,
                closest_brand=exact.brand_name,
                diagnostic_reference_domain=exact.canonical_domain,
                closeness_score=100.0,
                component_scores={key: 1.0 for key in self.weights},
                evidence=["exact match to a verified reference domain"],
                threshold=self.threshold,
                registry_version=self.version,
            )

        best: tuple[float, ReferenceDomain, dict[str, float], str] | None = None
        candidate_profile = self.candidate_profile(normalized)
        for reference in self.references:
            components, best_label = self.component_scores(normalized, reference, candidate_profile)
            weighted = sum(self.weights[key] * components[key] for key in self.weights)
            if best is None or weighted > best[0]:
                best = (weighted, reference, components, best_label)
        assert best is not None
        weighted, reference, components, best_label = best
        evidence = self._evidence(normalized, reference, best_label, components)
        is_match = weighted >= self.threshold and bool(evidence)
        return SimilarityResult(
            match_type="lookalike" if is_match else "no_close_match",
            closest_reference_domain=reference.canonical_domain if is_match else None,
            closest_brand=reference.brand_name if is_match else None,
            diagnostic_reference_domain=reference.canonical_domain,
            closeness_score=round(weighted * 100, 2),
            component_scores={key: round(value, 6) for key, value in components.items()},
            evidence=evidence if is_match else [],
            threshold=self.threshold,
            registry_version=self.version,
        )
