"""Agentic forecasting pipeline.

Per question:
  1. Initial web search for context
  2. LLM analyzes question + results, can request more searches
  3. Loop until LLM gives ANSWER: *probability* or max rounds hit

Features:
  - Incremental saves: forecasts written after each question
  - Resume: skips questions already forecasted (rerun same command after crash)
  - File logging: all output goes to logs/{date}.log
  - Auto-reconnect: Exa MCP reconnects on connection drops

Usage:
    python run.py --date 2025-12-21 --evaluate           # full run + eval
    python run.py --date 2025-12-21 --limit 5 -v         # test 5 questions, verbose
    python run.py --date 2025-12-21 --no-search --evaluate  # LLM only, no web
    python run.py --date 2025-12-21 --baseline --evaluate   # always-0.5 baseline
    python run.py --date 2025-12-21 --download-only         # just get the data
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from questions import download_question_set, download_resolutions, Question
from exa_search import ExaSearcher
from forecaster import Forecaster
from submission import write_submission, validate_submission
from evaluate import evaluate_submission, print_evaluation

log = logging.getLogger(__name__)


def setup_logging(date: str, verbose: bool):
    """Log to both console and file. Only our loggers, not library noise."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{date}.log"

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("anyio").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Our loggers
    our_logger = logging.getLogger()
    our_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler — always verbose for our code
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    our_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    our_logger.addHandler(ch)

    log.info(f"Logging to {log_file}")


# ---------------------------------------------------------------------------
# Progress / incremental save
# ---------------------------------------------------------------------------

PROGRESS_DIR = Path("progress")


def _progress_path(date: str) -> Path:
    return PROGRESS_DIR / f"{date}_forecasts.json"


def load_progress(date: str) -> list[dict]:
    """Load previously saved forecasts for this date."""
    path = _progress_path(date)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_progress(date: str, forecasts: list[dict]):
    """Save forecasts incrementally."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    path = _progress_path(date)
    path.write_text(json.dumps(forecasts, indent=2), encoding="utf-8")


def get_completed_keys(forecasts: list[dict]) -> set[str]:
    """Get set of (source:id) keys already forecasted."""
    keys = set()
    for f in forecasts:
        keys.add(f"{f['source']}:{f['id']}")
    return keys


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def process_questions(
    questions: list[Question],
    today: str,
    searcher: ExaSearcher | None,
    forecaster: Forecaster,
    existing_forecasts: list[dict],
    date: str,
) -> list[dict]:
    """Process questions sequentially with incremental saves and resume."""
    all_forecasts = list(existing_forecasts)
    completed = get_completed_keys(all_forecasts)
    total = len(questions)
    skipped = 0

    for i, q in enumerate(questions, 1):
        key = f"{q.source}:{q.id}"
        label = f"[{i}/{total}] {q.source}/{q.id}"

        # Skip already completed
        if key in completed:
            skipped += 1
            continue

        if skipped and i == skipped + 1:
            log.info(f"  Skipped {skipped} already-completed questions, resuming...")

        log.info(f"  {label}: {q.question[:80]}...")

        # Initial search
        initial_search = ""
        if searcher:
            try:
                initial_search = await searcher.search_for_question(
                    q.question, q.background, q.source,
                )
            except Exception as e:
                log.warning(f"    Initial search failed: {e}")

        # Agentic loop
        t0 = time.time()
        try:
            forecasts = await forecaster.forecast(q, initial_search, today)
        except Exception as e:
            log.error(f"    FORECAST FAILED for {label}: {e}")
            # Fall back to 0.5
            if q.is_market:
                forecasts = [{"id": q.id, "source": q.source,
                              "forecast": 0.5, "resolution_date": None,
                              "reasoning": f"Error: {e}"}]
            else:
                forecasts = [{"id": q.id, "source": q.source,
                              "forecast": 0.5, "resolution_date": d,
                              "reasoning": f"Error: {e}"}
                             for d in (q.resolution_dates or [None])]

        elapsed = time.time() - t0
        all_forecasts.extend(forecasts)

        prob_str = ", ".join(f"{f['forecast']:.2f}" for f in forecasts)
        log.info(f"    -> [{prob_str}] ({elapsed:.1f}s)")

        # Save after every question
        save_progress(date, all_forecasts)

    if skipped:
        log.info(f"  Total skipped (already done): {skipped}")

    return all_forecasts


def make_baseline_forecasts(questions: list[Question]) -> list[dict]:
    forecasts = []
    for q in questions:
        if q.is_market:
            forecasts.append({
                "id": q.id, "source": q.source,
                "forecast": 0.5, "resolution_date": None, "reasoning": None,
            })
        else:
            for date in (q.resolution_dates or []):
                forecasts.append({
                    "id": q.id, "source": q.source,
                    "forecast": 0.5, "resolution_date": date, "reasoning": None,
                })
    return forecasts


async def run_pipeline(args):
    date = args.date

    setup_logging(date, args.verbose)

    log.info(f"Downloading question set for {date}...")
    qs = download_question_set(date)
    log.info(f"  {len(qs.questions)} questions "
             f"({len(qs.market_questions)} market, {len(qs.dataset_questions)} dataset)")

    if args.download_only:
        return

    questions = qs.questions
    if args.limit:
        questions = questions[:args.limit]
        log.info(f"  Limited to {len(questions)} questions")

    if args.baseline:
        log.info("Baseline mode: predicting 0.5 for everything...")
        forecasts = make_baseline_forecasts(questions)
        log.info(f"  Generated {len(forecasts)} baseline forecasts")

    else:
        # Load progress for resume
        existing = load_progress(date)
        if existing:
            completed = get_completed_keys(existing)
            remaining = sum(1 for q in questions if f"{q.source}:{q.id}" not in completed)
            log.info(f"  Resuming: {len(completed)} questions done, {remaining} remaining")

        # Determine date cutoff for fair backtesting
        from datetime import date as dt
        today = dt.today().isoformat()
        date_cutoff = date if date < today else None

        # Set up searcher
        searcher = None
        if not args.no_search:
            if date_cutoff:
                log.info(f"Connecting to Exa MCP (date cutoff: {date_cutoff})...")
            else:
                log.info("Connecting to Exa MCP (live, no date cutoff)...")
            searcher = ExaSearcher(date_cutoff=date_cutoff)
            log.info("  Exa searcher ready (each search creates its own connection).")
        else:
            log.info("Skipping web search (--no-search)")

        forecaster = Forecaster(searcher=searcher, verbose=args.verbose, date_cutoff=date_cutoff)

        log.info(f"Processing {len(questions)} questions (agentic loop, sequential)...")
        t0 = time.time()
        try:
            forecasts = await process_questions(
                questions, qs.forecast_due_date, searcher, forecaster,
                existing, date,
            )
        finally:
            if searcher:
                await searcher.close()

        elapsed = time.time() - t0
        n_new = len(forecasts) - len(existing)
        log.info(f"  Done: {len(forecasts)} total forecasts "
                 f"({n_new} new) in {elapsed:.1f}s")

    sub_path = write_submission(qs, forecasts)

    issues = validate_submission(sub_path, qs)
    if issues:
        log.warning("Validation issues:")
        for issue in issues:
            log.warning(f"  - {issue}")
    else:
        log.info("Submission passes validation.")

    if args.evaluate:
        log.info(f"\nDownloading resolutions for {date}...")
        try:
            resolutions = download_resolutions(date)
            result = evaluate_submission(sub_path, resolutions)
            print_evaluation(result)
        except Exception as e:
            log.error(f"  Could not evaluate: {e}")


def main():
    parser = argparse.ArgumentParser(description="ForecastBench agentic pipeline")
    parser.add_argument("--date", required=True, help="Forecast due date (YYYY-MM-DD)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against resolutions")
    parser.add_argument("--no-search", action="store_true", help="Skip Exa web search")
    parser.add_argument("--baseline", action="store_true", help="Always predict 0.5 (no LLM)")
    parser.add_argument("--limit", type=int, help="Limit to N questions")
    parser.add_argument("--download-only", action="store_true", help="Just download data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show LLM reasoning")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
