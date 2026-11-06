from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, FEATURE_SCHEMA_VERSION, SEED
from .data_audit import sha256_file
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_feature_frame, feature_schema


def _group_holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    relative_train, relative_holdout = next(splitter.split(indices, labels[indices], groups[indices]))
    return indices[relative_train], indices[relative_holdout]


def grouped_partitions(labels: np.ndarray, groups: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    all_indices = np.arange(len(labels))
    remaining_90, test = _group_holdout(all_indices, labels, groups, 10, seed)
    remaining_80, calibration = _group_holdout(remaining_90, labels, groups, 9, seed + 1)
    train, validation = _group_holdout(remaining_80, labels, groups, 8, seed + 2)
    return {"train": train, "calibration": calibration, "validation": validation, "test": test}


def assert_no_group_leakage(partitions: dict[str, np.ndarray], groups: np.ndarray) -> dict[str, int]:
    group_sets = {name: set(groups[indices]) for name, indices in partitions.items()}
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise AssertionError(f"group leakage between {left} and {right}: {len(overlap)} domains")
    return {name: len(values) for name, values in group_sets.items()}


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("classifier", estimator)])


def _candidate_models(labels: np.ndarray, seed: int) -> dict[str, Pipeline]:
    negative = max(1, int((labels == 0).sum()))
    positive = max(1, int((labels == 1).sum()))
    models: dict[str, Pipeline] = {
        "dummy": _pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=700, class_weight="balanced", solver="liblinear", random_state=seed)
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=180,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = _pipeline(
            XGBClassifier(
                n_estimators=350,
                max_depth=7,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_lambda=1.0,
                scale_pos_weight=negative / positive,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=seed,
            )
        )
    except ImportError:
        pass
    return models


def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("trained model does not expose phishing as class 1")
    return probabilities[:, classes.index(1)]


def _threshold_for_f2(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5, 0.0
    numerator = 5 * precision[:-1] * recall[:-1]
    denominator = 4 * precision[:-1] + recall[:-1]
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    best_index = int(np.nanargmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fraction_positive, mean_predicted = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
    return {
        "threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": [
            {"mean_predicted": float(predicted), "fraction_positive": float(observed)}
            for predicted, observed in zip(mean_predicted, fraction_positive)
        ],
    }


def _calibrate_prefit(model: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> Any:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(features, labels)
    return calibrated


def train_model(
    dataset_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    report_path: str | Path,
    seed: int = SEED,
    max_rows: int | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    frame = pd.read_csv(dataset_path, usecols=["URL", "label"])
    frame = frame.dropna(subset=["URL", "label"]).copy()
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("PhiUSIIL label must contain only 0 and 1")
    conflicting_urls = frame.groupby("URL")["label"].nunique()
    conflicting_values = set(conflicting_urls[conflicting_urls > 1].index)
    if conflicting_values:
        frame = frame[~frame["URL"].isin(conflicting_values)]
    frame = frame.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)
    if max_rows is not None and max_rows < len(frame):
        frame = frame.groupby("label", group_keys=False).sample(
            n=max(2, max_rows // 2), random_state=seed, replace=False
        ).head(max_rows).reset_index(drop=True)

    features, normalized = build_feature_frame(frame["URL"].tolist())
    valid_mask = normalized["valid"].astype(bool).to_numpy()
    frame = frame.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask, FEATURE_COLUMNS].reset_index(drop=True)
    normalized = normalized.loc[valid_mask].reset_index(drop=True)
    labels = (1 - frame["label"].to_numpy(dtype=int)).astype(int)
    groups = normalized["registrable_domain"].astype(str).to_numpy()

    partitions = grouped_partitions(labels, groups, seed)
    group_counts = assert_no_group_leakage(partitions, groups)
    train_index = partitions["train"]
    validation_index = partitions["validation"]
    candidate_models = _candidate_models(labels[train_index], seed)
    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in candidate_models.items():
        print(f"Training {name} on {len(train_index):,} rows...", flush=True)
        model.fit(features.iloc[train_index], labels[train_index])
        probabilities = _positive_probability(model, features.iloc[validation_index])
        candidate_metrics[name] = {
            "pr_auc": float(average_precision_score(labels[validation_index], probabilities)),
            "roc_auc": float(roc_auc_score(labels[validation_index], probabilities)),
        }
        fitted_models[name] = model
        print(f"{name}: PR-AUC={candidate_metrics[name]['pr_auc']:.6f}", flush=True)

    eligible = [name for name in fitted_models if name != "dummy"]
    selected_name = max(eligible, key=lambda name: candidate_metrics[name]["pr_auc"])
    selected = fitted_models[selected_name]
    calibration_index = partitions["calibration"]
    calibrated = _calibrate_prefit(selected, features.iloc[calibration_index], labels[calibration_index])
    validation_probabilities = _positive_probability(calibrated, features.iloc[validation_index])
    threshold, validation_f2 = _threshold_for_f2(labels[validation_index], validation_probabilities)
    test_index = partitions["test"]
    test_probabilities = _positive_probability(calibrated, features.iloc[test_index])
    test_metrics = _metrics(labels[test_index], test_probabilities, threshold)

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    report_path = Path(report_path)
    for path in (model_path, metadata_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, model_path, compress=3)
    artifact_hash = sha256_file(model_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    model_version = f"url-model-{artifact_hash[:12]}"
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "model_version": model_version,
        "generated_at": generated_at,
        "selected_model": selected_name,
        "artifact": str(model_path.resolve()),
        "artifact_sha256": artifact_hash,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "label_mapping": {"raw_0": "phishing", "raw_1": "legitimate", "model_positive_1": "phishing"},
        "feature_schema": feature_schema(),
        "threshold": threshold,
        "caution_threshold": threshold * float(CONFIG["model"]["caution_ratio"]),
        "calibration": "sigmoid on domain-grouped calibration partition",
        "seed": seed,
        "split_row_counts": {name: int(len(indices)) for name, indices in partitions.items()},
        "split_group_counts": group_counts,
        "candidate_metrics": candidate_metrics,
        "validation_f2": validation_f2,
        "test_metrics": test_metrics,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    report = {
        "dataset_rows_after_cleaning": int(len(frame)),
        "removed_conflicting_url_count": len(conflicting_values),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selected_model": selected_name,
        "candidate_metrics": candidate_metrics,
        "threshold": threshold,
        "test_metrics": test_metrics,
        "split_row_counts": metadata["split_row_counts"],
        "split_group_counts": group_counts,
        "no_group_leakage": True,
        "model_version": model_version,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return metadata


def load_model(model_path: str | Path, metadata_path: str | Path) -> tuple[Any, dict[str, Any]]:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_hash = metadata.get("artifact_sha256")
    actual_hash = sha256_file(model_path)
    if expected_hash != actual_hash:
        raise ValueError("model artifact hash does not match metadata")
    schema = metadata.get("feature_schema", {})
    if schema.get("version") != FEATURE_SCHEMA_VERSION or schema.get("all") != FEATURE_COLUMNS:
        raise ValueError("model feature schema is incompatible with this application")
    return joblib.load(model_path), metadata
