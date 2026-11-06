from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG
from .features import FEATURE_COLUMNS, extract_url_features
from .similarity import SimilarityEngine, SimilarityResult
from .training import load_model
from .url_utils import NormalizedURL, normalize_url


@dataclass
class DetectionResult:
    input_url: str
    normalized_host: str
    registrable_domain: str
    valid: bool
    error: str | None
    predicted_label: str | None
    phishing_score: float | None
    model_threshold: float | None
    model_version: str | None
    closest_reference_domain: str | None
    closest_brand: str | None
    match_type: str
    closeness_score: float | None
    similarity_threshold: float | None
    similarity_components: dict[str, float]
    evidence: list[str]
    risk_band: str
    limitation: str
    reference_version: str | None
    scanned_at: str
    processing_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["similarity_components"] = json.dumps(self.similarity_components, sort_keys=True)
        result["evidence"] = "; ".join(self.evidence)
        return result


def _model_probability(model: Any, frame: pd.DataFrame) -> float:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("model artifact has no phishing-positive class")
    return float(probabilities[0, classes.index(1)])


def _total_ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.total_physical / (1024**3), 2)
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * page_count / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        return None


def _url_evidence(normalized: NormalizedURL, features: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    if normalized.is_ip:
        evidence.append("hostname is an IP address")
    if normalized.has_punycode:
        evidence.append("hostname uses Punycode")
    if features["suspicious_token_count"]:
        evidence.append(f"contains {features['suspicious_token_count']} suspicious lexical token(s)")
    if features["subdomain_count"] >= 3:
        evidence.append("unusually deep subdomain structure")
    if features["url_length"] >= 100:
        evidence.append("unusually long URL")
    if features["digit_ratio"] >= 0.25:
        evidence.append("high digit ratio")
    if features["at_count"]:
        evidence.append("URL contains an @ character")
    if features["percent_count"] >= 3:
        evidence.append("URL contains multiple percent-encoded characters")
    return evidence


def determine_risk_band(
    phishing_score: float,
    model_threshold: float,
    model_caution_threshold: float,
    similarity: SimilarityResult,
    similarity_caution_threshold: float,
) -> str:
    if similarity.match_type == "exact_reference":
        return "Lower observed risk"
    if phishing_score >= model_threshold or similarity.match_type == "lookalike":
        return "High risk"
    if phishing_score >= model_caution_threshold or similarity.closeness_score >= similarity_caution_threshold * 100:
        return "Caution"
    return "Lower observed risk"


class DetectionService:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata_path: str | Path,
        reference_path: str | Path,
        similarity_metadata_path: str | Path | None = None,
    ) -> None:
        self.model, self.model_metadata = load_model(model_path, model_metadata_path)
        self.similarity = SimilarityEngine.from_files(reference_path, similarity_metadata_path)
        self.model_threshold = float(self.model_metadata["threshold"])
        self.model_caution_threshold = float(
            self.model_metadata.get(
                "caution_threshold",
                self.model_threshold * float(CONFIG["model"]["caution_ratio"]),
            )
        )
        similarity_metadata: dict[str, Any] = {}
        if similarity_metadata_path is not None and Path(similarity_metadata_path).exists():
            with Path(similarity_metadata_path).open("r", encoding="utf-8") as handle:
                similarity_metadata = json.load(handle)
        self.similarity_caution_threshold = float(
            similarity_metadata.get("caution_threshold", CONFIG["similarity"]["caution_threshold"])
        )

    def scan(self, value: object) -> DetectionResult:
        start = time.perf_counter()
        normalized = normalize_url(value)
        scanned_at = datetime.now(timezone.utc).isoformat()
        if not normalized.valid:
            return DetectionResult(
                input_url=normalized.original,
                normalized_host="",
                registrable_domain="",
                valid=False,
                error=normalized.error,
                predicted_label=None,
                phishing_score=None,
                model_threshold=self.model_threshold,
                model_version=self.model_metadata.get("model_version"),
                closest_reference_domain=None,
                closest_brand=None,
                match_type="not_applicable",
                closeness_score=None,
                similarity_threshold=self.similarity.threshold,
                similarity_components={},
                evidence=[],
                risk_band="Validation error",
                limitation="The input could not be scored.",
                reference_version=self.similarity.version,
                scanned_at=scanned_at,
                processing_ms=round((time.perf_counter() - start) * 1000, 3),
            )

        features = extract_url_features(value, normalized)
        feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        phishing_score = _model_probability(self.model, feature_frame)
        predicted_label = "phishing-like" if phishing_score >= self.model_threshold else "legitimate-like"
        similarity = self.similarity.compare(value)
        evidence = _url_evidence(normalized, features) + similarity.evidence

        risk_band = determine_risk_band(
            phishing_score,
            self.model_threshold,
            self.model_caution_threshold,
            similarity,
            self.similarity_caution_threshold,
        )
        if similarity.match_type == "exact_reference" and phishing_score >= self.model_threshold:
            evidence.append("exact verified domain match takes precedence over URL-only model risk")

        return DetectionResult(
            input_url=normalized.original,
            normalized_host=normalized.ascii_hostname,
            registrable_domain=normalized.registrable_domain,
            valid=True,
            error=None,
            predicted_label=predicted_label,
            phishing_score=round(phishing_score, 6),
            model_threshold=self.model_threshold,
            model_version=self.model_metadata.get("model_version"),
            closest_reference_domain=similarity.closest_reference_domain,
            closest_brand=similarity.closest_brand,
            match_type=similarity.match_type,
            closeness_score=similarity.closeness_score,
            similarity_threshold=similarity.threshold,
            similarity_components=similarity.component_scores,
            evidence=list(dict.fromkeys(evidence)),
            risk_band=risk_band,
            limitation="Scores are screening evidence, not proof that a site is safe or malicious.",
            reference_version=similarity.registry_version,
            scanned_at=scanned_at,
            processing_ms=round((time.perf_counter() - start) * 1000, 3),
        )


def scan_batch(
    service: DetectionService,
    input_path: str | Path,
    url_column: str,
    csv_output: str | Path,
    json_output: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    frame = pd.read_csv(input_path)
    if url_column not in frame.columns:
        case_insensitive = {str(column).lower(): str(column) for column in frame.columns}
        matched = case_insensitive.get(url_column.lower())
        if matched is None:
            raise ValueError(f"batch CSV is missing URL column {url_column!r}")
        url_column = matched

    if limit is not None:
        if limit <= 0:
            raise ValueError("batch limit must be greater than zero")
        frame = frame.head(limit).copy()

    started = time.perf_counter()
    results = [service.scan(value) for value in frame[url_column].tolist()]
    flat = pd.DataFrame([result.to_flat_dict() for result in results])
    nested = [result.to_dict() for result in results]
    csv_output = Path(csv_output)
    json_output = Path(json_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(csv_output, index=False, encoding="utf-8")
    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(nested, handle, indent=2, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    latencies = np.array([result.processing_ms for result in results], dtype=float)
    summary = {
        "input_rows": int(len(frame)),
        "valid_rows": sum(result.valid for result in results),
        "invalid_rows": sum(not result.valid for result in results),
        "high_risk_rows": sum(result.risk_band == "High risk" for result in results),
        "csv_output": str(csv_output.resolve()),
        "json_output": str(json_output.resolve()),
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": len(results) / elapsed if elapsed else 0.0,
        "median_processing_ms": float(np.median(latencies)) if len(latencies) else 0.0,
        "p95_processing_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "model_version": service.model_metadata.get("model_version"),
        "model_type": service.model_metadata.get("selected_model"),
        "reference_count": len(service.similarity.references),
        "timing_scope": "elapsed time includes scoring and output serialization; model and input loading are excluded",
        "warmup": "none; percentile statistics include the first inference",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "total_ram_gb": _total_ram_gb(),
        },
    }
    summary_output = json_output.with_name(f"{json_output.stem}.summary.json")
    summary["summary_output"] = str(summary_output.resolve())
    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
