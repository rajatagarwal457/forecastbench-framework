"""Format and write ForecastBench submission files."""

import json
from pathlib import Path

import config
from questions import QuestionSet


def build_submission(
    question_set: QuestionSet,
    forecasts: list[dict],
    submission_number: int = 1,
) -> dict:
    """Build the submission JSON structure."""
    return {
        "organization": config.ORGANIZATION,
        "model": config.VLLM_MODEL,
        "model_organization": config.MODEL_ORGANIZATION,
        "question_set": question_set.question_set_name,
        "forecast_due_date": question_set.forecast_due_date,
        "forecasts": forecasts,
    }


def write_submission(
    question_set: QuestionSet,
    forecasts: list[dict],
    output_dir: str = "submissions",
    submission_number: int = 1,
) -> Path:
    """Write the submission JSON file."""
    sub = build_submission(question_set, forecasts, submission_number)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{question_set.forecast_due_date}.{config.ORGANIZATION}.{submission_number}.json"
    path = out_dir / filename
    path.write_text(json.dumps(sub, indent=2), encoding="utf-8")
    print(f"Submission written to {path}")
    return path


def validate_submission(path: Path, question_set: QuestionSet) -> list[str]:
    """Basic validation of a submission file. Returns list of issues."""
    issues = []
    data = json.loads(path.read_text(encoding="utf-8"))

    for key in ["organization", "model", "question_set", "forecast_due_date", "forecasts"]:
        if key not in data:
            issues.append(f"Missing required key: {key}")

    if data.get("question_set") != question_set.question_set_name:
        issues.append(
            f"question_set mismatch: {data.get('question_set')} != {question_set.question_set_name}"
        )

    forecasts = data.get("forecasts", [])
    # Check coverage
    market_ids = {q.id for q in question_set.market_questions}
    dataset_ids = {q.id for q in question_set.dataset_questions}
    forecasted_market = {f["id"] for f in forecasts if f["source"] in config.MARKET_SOURCES}
    forecasted_dataset = {f["id"] for f in forecasts if f["source"] in config.DATASET_SOURCES}

    market_coverage = len(forecasted_market & market_ids) / max(len(market_ids), 1)
    dataset_coverage = len(forecasted_dataset & dataset_ids) / max(len(dataset_ids), 1)

    if market_coverage < 0.95:
        issues.append(f"Market coverage {market_coverage:.1%} < 95%")
    if dataset_coverage < 0.95:
        issues.append(f"Dataset coverage {dataset_coverage:.1%} < 95%")

    # Check forecast values
    for f in forecasts:
        p = f.get("forecast")
        if p is None or not (0 <= p <= 1):
            issues.append(f"Invalid forecast value {p} for {f.get('source')}/{f.get('id')}")

    return issues
