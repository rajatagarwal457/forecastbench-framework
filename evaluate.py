"""Evaluate forecasts against historical resolutions (Brier score).

Also compares against the tournament leaderboard to show where you'd rank.
"""

import csv
import json
import math
from pathlib import Path

import httpx

from questions import Resolution
from config import MARKET_SOURCES, DATASET_SOURCES


LEADERBOARD_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/"
    "main/leaderboards/csv/leaderboard_tournament.csv"
)

DATA_DIR = Path("data")


def brier_score(forecast: float, outcome: float) -> float:
    return (forecast - outcome) ** 2


def evaluate_submission(
    submission_path: Path,
    resolutions: list[Resolution],
) -> dict:
    """Score a submission against resolutions."""
    data = json.loads(submission_path.read_text(encoding="utf-8"))
    forecasts = data["forecasts"]

    # Build forecast lookups
    market_forecast_map = {}
    dataset_forecast_map = {}
    for f in forecasts:
        if f["source"] in MARKET_SOURCES:
            market_forecast_map[(str(f["id"]), f["source"])] = f["forecast"]
        else:
            dataset_forecast_map[(str(f["id"]), f["source"], f.get("resolution_date"))] = f["forecast"]

    scores = {"all": [], "market": [], "dataset": []}

    for r in resolutions:
        if r.resolved_to is None:
            continue

        if r.source in MARKET_SOURCES:
            p = market_forecast_map.get((r.id, r.source), 0.5)
        elif r.source in DATASET_SOURCES:
            p = dataset_forecast_map.get((r.id, r.source, r.resolution_date), 0.5)
        else:
            continue

        bs = brier_score(p, r.resolved_to)
        scores["all"].append(bs)
        if r.source in MARKET_SOURCES:
            scores["market"].append(bs)
        else:
            scores["dataset"].append(bs)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    def _brier_index(avg_bs):
        if math.isnan(avg_bs):
            return float("nan")
        return (1 - math.sqrt(avg_bs)) * 100

    result = {
        "total_forecasts": len(forecasts),
        "total_resolutions": len(resolutions),
    }
    for cat in ["all", "market", "dataset"]:
        avg = _avg(scores[cat])
        result[f"{cat}_brier_score"] = round(avg, 4)
        result[f"{cat}_brier_index"] = round(_brier_index(avg), 2)
        result[f"{cat}_count"] = len(scores[cat])

    return result


def download_leaderboard() -> list[dict]:
    """Download the tournament leaderboard CSV."""
    cache_path = DATA_DIR / "leaderboard_tournament.csv"
    if not cache_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(LEADERBOARD_URL, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")

    rows = []
    with open(cache_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def print_evaluation(result: dict):
    """Pretty-print evaluation results + leaderboard comparison."""
    print("\n" + "=" * 70)
    print("YOUR RESULTS")
    print("=" * 70)
    print(f"Forecasts submitted: {result['total_forecasts']}")
    print(f"Resolutions scored:  {result['total_resolutions']}")
    print()

    for cat in ["all", "market", "dataset"]:
        n = result[f"{cat}_count"]
        bs = result[f"{cat}_brier_score"]
        bi = result[f"{cat}_brier_index"]
        label = cat.upper()
        print(f"  {label:>10}:  Brier={bs:.4f}  Index={bi:.1f}  (n={n})")

    print()
    print("  Brier Score: 0=perfect, 0.25=always-0.5, 1=worst")
    print("  Brier Index: 100=perfect, 50=always-0.5, 0=worst")

    # Compare against leaderboard
    try:
        leaderboard = download_leaderboard()
    except Exception:
        leaderboard = []

    if leaderboard:
        my_overall = result["all_brier_index"]
        my_brier = result["all_brier_score"]
        my_dataset_bi = result["dataset_brier_index"]
        my_market_bi = result["market_brier_index"]

        print()
        print("=" * 70)
        print("TOURNAMENT LEADERBOARD COMPARISON")
        print("=" * 70)
        print(f"{'Rank':<6}{'Team':<25}{'Model':<30}{'Overall':>8}{'Dataset':>9}{'Market':>8}")
        print("-" * 70)

        # Find where we'd rank
        your_rank = len(leaderboard) + 1
        printed_you = False

        for row in leaderboard:
            rank = row.get("Rank", "?")
            team = row.get("Team", "?")[:24]
            model = row.get("Model", "?")[:29]
            overall = row.get("Overall", "?")
            dataset = row.get("Dataset", "?")
            market = row.get("Market", "?")

            try:
                their_overall = float(overall)
            except (ValueError, TypeError):
                their_overall = 0

            # Insert our row when we'd rank here
            if not printed_you and my_overall >= their_overall:
                your_rank = int(rank) if rank.isdigit() else "?"
                print(f"{'>YOU':<6}{'YOU':<25}{'Qwen3.5-27B + Exa':<30}"
                      f"{my_overall:>7.1f}  {my_dataset_bi:>7.1f}  {my_market_bi:>7.1f}")
                printed_you = True

            print(f"{rank:<6}{team:<25}{model:<30}{overall:>8}{dataset:>9}{market:>8}")

        if not printed_you:
            print(f"{'>YOU':<6}{'YOU':<25}{'Qwen3.5-27B + Exa':<30}"
                  f"{my_overall:>7.1f}  {my_dataset_bi:>7.1f}  {my_market_bi:>7.1f}")

        print("-" * 70)
        print(f"\nYou would rank approximately #{your_rank} out of {len(leaderboard)} entries")
        print("(Note: leaderboard uses difficulty-adjusted scores; this is a rough comparison)")

    print("=" * 70)
