"""Download and parse ForecastBench question sets and resolution sets."""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import httpx

import config

CACHE_DIR = Path("data")


@dataclass
class Question:
    id: str
    source: str
    question: str
    background: str
    resolution_criteria: str
    freeze_datetime: str
    freeze_datetime_value: str
    freeze_datetime_value_explanation: str
    url: str
    resolution_dates: list[str] | None  # None for market questions
    source_intro: str = ""
    market_info_open_datetime: str = ""
    market_info_close_datetime: str = ""
    market_info_resolution_criteria: str = ""

    @property
    def is_market(self) -> bool:
        return self.source in config.MARKET_SOURCES

    @property
    def is_dataset(self) -> bool:
        return self.source in config.DATASET_SOURCES


@dataclass
class QuestionSet:
    forecast_due_date: str
    question_set_name: str
    questions: list[Question]

    @property
    def market_questions(self) -> list[Question]:
        return [q for q in self.questions if q.is_market]

    @property
    def dataset_questions(self) -> list[Question]:
        return [q for q in self.questions if q.is_dataset]


@dataclass
class Resolution:
    id: str
    source: str
    resolution_date: str | None
    resolved_to: float | None  # 0 or 1, or None if unresolved


def _download(url: str, cache_path: Path) -> dict:
    """Download JSON, caching locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Downloading {url} ...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def download_latest_question_set() -> QuestionSet:
    """Download the latest question set by resolving the symlink."""
    symlink_url = f"{config.QUESTION_SET_RAW_BASE}/latest-llm.json"
    print(f"Resolving {symlink_url} ...")
    resp = httpx.get(symlink_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    target = resp.text.strip()

    # target is like "2026-03-01-llm.json"
    if target.endswith("-llm.json"):
        date = target.replace("-llm.json", "")
        print(f"  Latest question set: {date}")
        return download_question_set(date)
    else:
        # In case the symlink resolves to actual JSON content
        data = json.loads(resp.text)
        date = data["forecast_due_date"]
        filename = f"{date}-llm.json"
        cache_path = CACHE_DIR / "question_sets" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return _parse_question_set(data, filename)


def download_question_set(date: str) -> QuestionSet:
    """Download the question set for a given forecast_due_date (YYYY-MM-DD)."""
    filename = f"{date}-llm.json"
    url = f"{config.QUESTION_SET_RAW_BASE}/{filename}"
    cache_path = CACHE_DIR / "question_sets" / filename
    data = _download(url, cache_path)
    return _parse_question_set(data, filename)


def _parse_question_set(data: dict, filename: str) -> QuestionSet:
    questions = []
    for q in data["questions"]:
        res_dates = q.get("resolution_dates")
        if res_dates == "N/A" or res_dates is None:
            res_dates = None
        elif isinstance(res_dates, str):
            res_dates = [res_dates]

        questions.append(Question(
            id=str(q["id"]),
            source=q["source"],
            question=q["question"],
            background=q.get("background", ""),
            resolution_criteria=q.get("resolution_criteria", ""),
            freeze_datetime=q.get("freeze_datetime", ""),
            freeze_datetime_value=str(q.get("freeze_datetime_value", "")),
            freeze_datetime_value_explanation=q.get("freeze_datetime_value_explanation", ""),
            url=q.get("url", ""),
            resolution_dates=res_dates,
            source_intro=q.get("source_intro", ""),
            market_info_open_datetime=q.get("market_info_open_datetime", ""),
            market_info_close_datetime=q.get("market_info_close_datetime", ""),
            market_info_resolution_criteria=q.get("market_info_resolution_criteria", ""),
        ))

    return QuestionSet(
        forecast_due_date=data["forecast_due_date"],
        question_set_name=data.get("question_set", filename),
        questions=questions,
    )


def download_resolutions(date: str) -> list[Resolution]:
    """Download the resolution set for a given forecast_due_date."""
    filename = f"{date}_resolution_set.json"
    url = f"{config.RESOLUTION_SET_RAW_BASE}/{filename}"
    cache_path = CACHE_DIR / "resolution_sets" / filename
    data = _download(url, cache_path)

    resolutions = []
    for r in data.get("resolutions", []):
        # Skip unresolved entries
        if not r.get("resolved", False):
            continue

        resolved = r.get("resolved_to")
        if resolved is not None:
            try:
                resolved = float(resolved)
            except (ValueError, TypeError):
                continue
        else:
            continue

        res_date = r.get("resolution_date")
        if res_date == "N/A" or res_date is None:
            res_date = None

        resolutions.append(Resolution(
            id=str(r["id"]),
            source=r["source"],
            resolution_date=res_date,
            resolved_to=resolved,
        ))
    return resolutions


def list_available_question_sets() -> list[str]:
    """List cached question set dates."""
    d = CACHE_DIR / "question_sets"
    if not d.exists():
        return []
    return sorted(
        f.stem.replace("-llm", "")
        for f in d.glob("*-llm.json")
    )
