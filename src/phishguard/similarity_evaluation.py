from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score

from .config import CONFIG, SEED
from .data_audit import sha256_file
from .reference import load_references, registry_version
from .similarity import SimilarityEngine
from .url_utils import normalize_url


COMPONENT_NAMES = ["levenshtein", "jaro_winkler", "ngram", "confusable"]


def _weight_candidates() -> list[np.ndarray]:
    candidates: list[np.ndarray] = [np.array([0.35, 0.25, 0.25, 0.15], dtype=float)]
    for first in range(1, 8):
        for second in range(1, 8):
            for third in range(1, 8):
                fourth = 10 - first - second - third
                if 0 <= fourth <= 7:
                    candidates.append(np.array([first, second, third, fourth], dtype=float) / 10.0)
    unique: dict[tuple[float, ...], np.ndarray] = {tuple(candidate): candidate for candidate in candidates}
    return list(unique.values())


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.8, 0.0
    numerator = 5 * precision[:-1] * recall[:-1]
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    index = int(np.nanargmax(f2))
    return float(thresholds[index]), float(f2[index])


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_similarity(
    reference_path: str | Path,
    synthetic_path: str | Path,
    metadata_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    references = load_references(reference_path)
    pairs = pd.read_csv(synthetic_path)
    required = {
        "reference_id",
        "generated_hostname",
        "transformation_type",
        "is_synthetic_lookalike",
        "split",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"synthetic data is missing columns: {sorted(missing)}")
    reference_index = {reference.reference_id: index for index, reference in enumerate(references)}
    unknown = set(pairs["reference_id"]) - set(reference_index)
    if unknown:
        raise ValueError(f"synthetic data contains unknown reference IDs: {sorted(unknown)[:5]}")
    split_per_reference = pairs.groupby("reference_id")["split"].nunique()
    if (split_per_reference != 1).any():
        raise ValueError("synthetic reference groups leak across splits")

    initial = SimilarityEngine(references)
    components = np.zeros((len(pairs), len(references), len(COMPONENT_NAMES)), dtype=np.float32)
    exact_any = np.zeros(len(pairs), dtype=bool)
    normalized_values = []
    print(f"Precomputing similarity components for {len(pairs):,} pairs...", flush=True)
    for row_index, hostname in enumerate(pairs["generated_hostname"].astype(str)):
        normalized = normalize_url(hostname)
        if not normalized.valid:
            raise ValueError(f"invalid synthetic hostname at row {row_index + 2}: {hostname}")
        normalized_values.append(normalized)
        exact_any[row_index] = normalized.registrable_domain in initial.reference_by_domain
        candidate_profile = initial.candidate_profile(normalized)
        for ref_index, reference in enumerate(references):
            values, _ = initial.component_scores(normalized, reference, candidate_profile)
            components[row_index, ref_index] = [values[name] for name in COMPONENT_NAMES]

    labels = pairs["is_synthetic_lookalike"].to_numpy(dtype=int)
    tuning_mask = pairs["split"].eq("tuning").to_numpy()
    best_weights = None
    best_tuning_f2 = -1.0
    for weights in _weight_candidates():
        all_scores = np.einsum("nrf,f->nr", components, weights, optimize=True)
        best_scores = all_scores.max(axis=1)
        effective_scores = best_scores.copy()
        effective_scores[exact_any] = 0.0
        _, tuning_f2 = _best_threshold(labels[tuning_mask], effective_scores[tuning_mask])
        if tuning_f2 > best_tuning_f2:
            best_tuning_f2 = tuning_f2
            best_weights = weights
    assert best_weights is not None

    final_scores = np.einsum("nrf,f->nr", components, best_weights, optimize=True)
    best_scores = final_scores.max(axis=1)
    best_reference_indices = final_scores.argmax(axis=1)
    effective_scores = best_scores.copy()
    effective_scores[exact_any] = 0.0
    validation_mask = pairs["split"].eq("validation").to_numpy()
    threshold, validation_f2 = _best_threshold(labels[validation_mask], effective_scores[validation_mask])

    weights_dict = {name: float(value) for name, value in zip(COMPONENT_NAMES, best_weights)}
    engine = SimilarityEngine(references, weights=weights_dict, threshold=threshold, version=registry_version(reference_path))
    test_mask = pairs["split"].eq("test").to_numpy()
    test_indices = np.flatnonzero(test_mask)
    predictions: list[int] = []
    retrieval_correct: list[bool] = []
    predicted_results = []
    for row_index in test_indices:
        result = engine.compare(pairs.iloc[row_index]["generated_hostname"])
        predictions.append(int(result.match_type == "lookalike"))
        predicted_results.append(result)
        if labels[row_index] == 1:
            expected_index = reference_index[str(pairs.iloc[row_index]["reference_id"])]
            retrieval_correct.append(int(best_reference_indices[row_index]) == expected_index)

    test_labels = labels[test_mask]
    test_predictions = np.asarray(predictions, dtype=int)
    metrics = _binary_metrics(test_labels, test_predictions)
    metrics["nearest_reference_top1_accuracy"] = float(np.mean(retrieval_correct)) if retrieval_correct else 0.0
    by_transformation: dict[str, Any] = {}
    test_transformations = pairs.loc[test_mask, "transformation_type"].astype(str).to_numpy()
    for transformation in sorted(set(test_transformations)):
        mask = test_transformations == transformation
        subset_labels = test_labels[mask]
        subset_predictions = test_predictions[mask]
        by_transformation[transformation] = {
            "count": int(mask.sum()),
            "accuracy": float((subset_labels == subset_predictions).mean()),
            "detected_positive_count": int(subset_predictions.sum()),
        }

    metadata = {
        "schema_version": "1.0",
        "weights": weights_dict,
        "threshold": threshold,
        "caution_threshold": max(0.3, threshold * 0.9),
        "validation_f2": validation_f2,
        "tuning_f2": best_tuning_f2,
        "registry_version": registry_version(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "synthetic_sha256": sha256_file(synthetic_path),
        "seed": SEED,
        "test_metrics": metrics,
    }
    report = {
        **metadata,
        "row_count": int(len(pairs)),
        "split_counts": {key: int(value) for key, value in pairs["split"].value_counts().items()},
        "test_by_transformation": by_transformation,
        "reference_group_leakage": False,
    }
    metadata_path = Path(metadata_path)
    report_path = Path(report_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
