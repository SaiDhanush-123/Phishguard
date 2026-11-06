from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from phishguard.config import CONFIG, project_path
from phishguard.detection import DetectionResult, DetectionService


app = Flask(__name__)

BATCH_ROW_LIMIT = 5_000

_service: DetectionService | None = None


def get_service() -> DetectionService:
    global _service
    if _service is None:
        paths = CONFIG["paths"]
        _service = DetectionService(
            project_path(paths["model"]),
            project_path(paths["model_metadata"]),
            project_path(paths["references"]),
            project_path(paths["similarity_metadata"]),
        )
    return _service


def try_get_service() -> tuple[DetectionService | None, str | None]:
    try:
        return get_service(), None
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def risk_style(risk_band: str) -> str:
    """CSS modifier class applied to the verdict pill in the UI."""
    if risk_band == "High risk":
        return "high"
    if risk_band == "Caution":
        return "caution"
    return "low"


def serialize_result(result: DetectionResult) -> dict[str, Any]:
    if not result.valid:
        return {
            "valid": False,
            "input_url": result.input_url,
            "error": result.error,
        }

    evidence = result.evidence
    signals = [{"label": item, "points": 0} for item in evidence]
    if not signals:
        signals = [{"label": "No specific URL or imitation indicators crossed the explanation rules.", "points": 0}]
    evidence_text = (
        "Indicators found: " + "; ".join(evidence) + "."
        if evidence
        else "No specific URL or imitation indicators crossed the explanation rules."
    )
    metrics = {name: round(value * 100, 1) for name, value in result.similarity_components.items()}

    return {
        "valid": True,
        "input_url": result.input_url,
        "host": result.registrable_domain or result.normalized_host,
        "score": round(float(result.phishing_score or 0) * 100, 1),
        "closeness": round(float(result.closeness_score or 0), 1),
        "closest": result.closest_reference_domain or "No close reference match",
        "closest_brand": result.closest_brand,
        "match_type": result.match_type,
        "risk_band": result.risk_band,
        "risk_class": risk_style(result.risk_band),
        "signals": signals,
        "evidence_text": evidence_text,
        "metrics": metrics,
        "model": result.model_version,
        "references": result.reference_version,
        "time_ms": result.processing_ms,
        "limitation": result.limitation,
    }


@app.route("/")
def index():
    service, error = try_get_service()
    if error:
        return render_template("error.html", error=error), 500
    return render_template("index.html", service=service)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    service, error = try_get_service()
    if error:
        return jsonify({"valid": False, "error": f"The detection artifacts could not be loaded: {error}"}), 503

    payload = request.get_json(silent=True) or {}
    value = str(payload.get("url", "")).strip()
    if not value:
        return jsonify({"valid": False, "error": "Enter a URL or domain before starting the analysis."}), 400

    result = service.scan(value)
    return jsonify(serialize_result(result))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    service, error = try_get_service()
    if error:
        return jsonify({"error": f"The detection artifacts could not be loaded: {error}"}), 503

    payload = request.get_json(silent=True) or {}
    raw_urls = payload.get("urls")
    if not isinstance(raw_urls, list):
        return jsonify({"error": "urls must be a list of strings"}), 400

    cleaned = [str(value).strip() for value in raw_urls if str(value).strip()]
    truncated = len(cleaned) > BATCH_ROW_LIMIT
    cleaned = cleaned[:BATCH_ROW_LIMIT]

    rows = []
    for value in cleaned:
        result = service.scan(value)
        row = serialize_result(result)
        row["input"] = value
        rows.append(row)

    return jsonify({"rows": rows, "truncated": truncated, "row_limit": BATCH_ROW_LIMIT})


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
