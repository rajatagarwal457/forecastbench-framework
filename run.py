"""Agentic forecasting pipeline.

Per question:
  1. Initial web search for context
  2. LLM analyzes question + results, can request more searches
  3. Loop until LLM gives ANSWER: *probability* or max rounds hit

Usage:
    python run.py --date 2024-07-21 --evaluate           # full run + eval
    python run.py --date 2026-03-01 --limit 5 -v         # test 5 questions, verbose
    python run.py --date 2024-07-21 --no-search --evaluate  # LLM only, no web
    python run.py --date 2024-07-21 --baseline --evaluate   # always-0.5 baseline
    python run.py --date 2024-07-21 --download-only         # just get the data
"""

import argparse
import asyncio
import time

from questions import download_question_set, download_resolutions, Question
from exa_search import ExaSearcher
from forecaster import Forecaster
from submission import write_submission, validate_submission
from evaluate import evaluate_submission, print_evaluation


async def process_questions(
    questions: list[Question],
    today: str,
    searcher: ExaSearcher | None,
    forecaster: Forecaster,
) -> list[dict]:
    """Process questions sequentially. Each gets its own agentic loop."""
    all_forecasts = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        label = f"[{i}/{total}] {q.source}/{q.id}"
        print(f"  {label}: {q.question[:80]}...")

        # Initial search to seed the agent with context
        initial_search = ""
        if searcher:
            try:
                initial_search = await searcher.search_for_question(
                    q.question, q.background, q.source,
                )
            except Exception as e:
                print(f"    Initial search failed: {e}")

        # Agentic loop: LLM reasons + requests searches until it answers
        t0 = time.time()
        forecasts = await forecaster.forecast(q, initial_search, today)
        elapsed = time.time() - t0
        all_forecasts.extend(forecasts)

        prob_str = ", ".join(f"{f['forecast']:.2f}" for f in forecasts)
        print(f"    -> [{prob_str}] ({elapsed:.1f}s)")

    return all_forecasts


def make_baseline_forecasts(questions: list[Question]) -> list[dict]:
    """Baseline: always predict 0.5."""
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

    print(f"Downloading question set for {date}...")
    qs = download_question_set(date)
    print(f"  {len(qs.questions)} questions "
          f"({len(qs.market_questions)} market, {len(qs.dataset_questions)} dataset)")

    if args.download_only:
        return

    questions = qs.questions
    if args.limit:
        questions = questions[:args.limit]
        print(f"  Limited to {len(questions)} questions")

    if args.baseline:
        print("Baseline mode: predicting 0.5 for everything...")
        forecasts = make_baseline_forecasts(questions)
        print(f"  Generated {len(forecasts)} baseline forecasts")

    else:
        # Determine date cutoff for fair backtesting
        from datetime import date as dt
        today = dt.today().isoformat()
        date_cutoff = date if date < today else None

        # Set up searcher
        searcher = None
        if not args.no_search:
            if date_cutoff:
                print(f"Connecting to Exa MCP (date cutoff: {date_cutoff})...")
            else:
                print("Connecting to Exa MCP (live, no date cutoff)...")
            searcher = ExaSearcher(date_cutoff=date_cutoff)
            try:
                await searcher.connect()
                print("  Connected.")
            except Exception as e:
                print(f"  Exa connection failed: {e}")
                searcher = None
        else:
            print("Skipping web search (--no-search)")

        # Forecaster gets the searcher so it can do tool-call searches
        forecaster = Forecaster(searcher=searcher, verbose=args.verbose, date_cutoff=date_cutoff)

        print(f"Processing {len(questions)} questions (agentic loop, sequential)...")
        t0 = time.time()
        try:
            forecasts = await process_questions(
                questions, qs.forecast_due_date, searcher, forecaster,
            )
        finally:
            if searcher:
                await searcher.close()

        elapsed = time.time() - t0
        print(f"  Done: {len(forecasts)} forecasts in {elapsed:.1f}s "
              f"({elapsed/len(questions):.1f}s/question)")

    sub_path = write_submission(qs, forecasts)

    issues = validate_submission(sub_path, qs)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Submission passes validation.")

    if args.evaluate:
        print(f"\nDownloading resolutions for {date}...")
        try:
            resolutions = download_resolutions(date)
            result = evaluate_submission(sub_path, resolutions)
            print_evaluation(result)
        except Exception as e:
            print(f"  Could not evaluate: {e}")


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
