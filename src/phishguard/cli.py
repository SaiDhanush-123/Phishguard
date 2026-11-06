from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config, project_path
from .data_audit import audit_dataset
from .detection import DetectionService, scan_batch
from .reference import validate_reference_registry
from .similarity_evaluation import evaluate_similarity
from .synthetic import generate_synthetic_pairs
from .training import train_model


def _path(value: str | Path) -> Path:
    return project_path(value)


def _service(config: dict[str, Any]) -> DetectionService:
    paths = config["paths"]
    return DetectionService(
        _path(paths["model"]),
        _path(paths["model_metadata"]),
        _path(paths["references"]),
        _path(paths["similarity_metadata"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishguard", description="Offline phishing URL and look-alike detector")
    parser.add_argument("--config", help="Optional JSON configuration override")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-data", help="Audit the PhiUSIIL dataset")
    audit.add_argument("--dataset")
    audit.add_argument("--output", default="reports/dataset_audit.json")

    references = commands.add_parser("validate-reference", help="Validate the genuine-domain registry")
    references.add_argument("--references")
    references.add_argument("--output", default="reports/reference_registry_audit.json")
    references.add_argument("--expected-count", type=int, default=200)

    synthetic = commands.add_parser("generate-synthetic", help="Generate controlled look-alike pairs")
    synthetic.add_argument("--references")
    synthetic.add_argument("--output")
    synthetic.add_argument("--seed", type=int)

    training = commands.add_parser("train", help="Train and evaluate the URL-only classifier")
    training.add_argument("--dataset")
    training.add_argument("--model")
    training.add_argument("--metadata")
    training.add_argument("--report", default="reports/model_evaluation.json")
    training.add_argument("--max-rows", type=int, help="Optional development-only stratified row limit")

    similarity = commands.add_parser("evaluate-similarity", help="Tune and evaluate look-alike similarity")
    similarity.add_argument("--references")
    similarity.add_argument("--synthetic")
    similarity.add_argument("--metadata")
    similarity.add_argument("--report", default="reports/similarity_evaluation.json")

    scan = commands.add_parser("scan", help="Scan one URL or domain")
    scan.add_argument("url")
    scan.add_argument("--json-output")

    batch = commands.add_parser("scan-batch", help="Scan a CSV of URLs")
    batch.add_argument("input")
    batch.add_argument("--column", default="URL")
    batch.add_argument("--csv-output", default="reports/batch_scan.csv")
    batch.add_argument("--json-output", default="reports/batch_scan.json")
    batch.add_argument("--limit", type=int, help="Optional maximum rows to process")
    return parser


def _print_scan(result: Any) -> None:
    print(f"Input: {result.input_url}")
    if not result.valid:
        print(f"Status: validation error ({result.error})")
        return
    print(f"Domain: {result.registrable_domain}")
    print(f"Risk: {result.risk_band}")
    print(f"ML result: {result.predicted_label}")
    print(f"Phishing probability: {result.phishing_score * 100:.2f}%")
    print(f"Domain closeness: {result.closeness_score:.2f}/100")
    print(f"Reference match: {result.closest_reference_domain or 'no close reference match'}")
    if result.evidence:
        print("Evidence:")
        for item in result.evidence:
            print(f"- {item}")
    print(f"Notice: {result.limitation}")


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    paths = config["paths"]
    if args.command == "audit-data":
        report = audit_dataset(_path(args.dataset or paths["dataset"]), _path(args.output))
        print(json.dumps(report, indent=2))
    elif args.command == "validate-reference":
        report = validate_reference_registry(
            _path(args.references or paths["references"]),
            _path(args.output),
            expected_count=args.expected_count,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "generate-synthetic":
        report = generate_synthetic_pairs(
            _path(args.references or paths["references"]),
            _path(args.output or paths["synthetic_pairs"]),
            seed=args.seed if args.seed is not None else int(config["seed"]),
        )
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        report = train_model(
            _path(args.dataset or paths["dataset"]),
            _path(args.model or paths["model"]),
            _path(args.metadata or paths["model_metadata"]),
            _path(args.report),
            seed=int(config["seed"]),
            max_rows=args.max_rows,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate-similarity":
        report = evaluate_similarity(
            _path(args.references or paths["references"]),
            _path(args.synthetic or paths["synthetic_pairs"]),
            _path(args.metadata or paths["similarity_metadata"]),
            _path(args.report),
        )
        print(json.dumps(report, indent=2))
    elif args.command == "scan":
        result = _service(config).scan(args.url)
        _print_scan(result)
        if args.json_output:
            output = _path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
    elif args.command == "scan-batch":
        summary = scan_batch(
            _service(config),
            _path(args.input),
            args.column,
            _path(args.csv_output),
            _path(args.json_output),
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
