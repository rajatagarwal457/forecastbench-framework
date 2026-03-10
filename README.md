<div align="center">

# forecastbench-agent

**An agentic forecasting system for [ForecastBench](https://www.forecastbench.org/)**

Any open-source LLM + free web search &rarr; calibrated probability forecasts on real-world questions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ForecastBench](https://img.shields.io/badge/benchmark-ForecastBench-orange.svg)](https://www.forecastbench.org/)
[![Exa Search](https://img.shields.io/badge/search-Exa%20MCP-purple.svg)](https://exa.ai/mcp)
[![vLLM](https://img.shields.io/badge/inference-vLLM-red.svg)](https://github.com/vllm-project/vllm)

[ForecastBench Paper (ICLR 2025)](https://arxiv.org/abs/2409.19839) &bull; [Leaderboard](https://www.forecastbench.org/) &bull; [Question Sets](https://github.com/forecastingresearch/forecastbench-datasets)

</div>

---

## What is this?

An agent that participates in [ForecastBench](https://www.forecastbench.org/) — the dynamic, contamination-free benchmark for evaluating LLM forecasting accuracy. It uses any open-source LLM (served via [vLLM](https://github.com/vllm-project/vllm)) combined with free [Exa](https://exa.ai) web search to produce well-calibrated probability forecasts on prediction market and time-series questions.

The LLM doesn't just get search results dumped on it — **it drives its own research**. It reads initial context, decides what additional information it needs, requests targeted searches, analyzes results, and iterates until it's confident enough to commit to a probability.

## How It Works

```
  Question + metadata
         │
         ▼
  ┌─────────────────────────┐
  │  Initial web search     │  2-3 Exa queries for baseline context
  │  (date-filtered for     │
  │   fair backtesting)     │
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  LLM analyzes context,  │
  │  reasons about question, │◄──┐
  │  requests more searches  │   │
  │  via SEARCH("query")     │   │
  └──────────┬───────────────┘   │
             │                    │
             ▼                    │
  ┌─────────────────────────┐    │
  │  Execute Exa searches,  │────┘
  │  feed results back      │  (up to 5 rounds)
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  LLM commits:           │
  │  ANSWER: *0.73*          │
  └─────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+**
- **A vLLM server** running any model with an OpenAI-compatible API
- **No API keys needed** for web search — [Exa MCP](https://exa.ai/mcp) hosted endpoint is free

### Install

```bash
git clone https://github.com/yourusername/forecastbench-agent.git
cd forecastbench-agent
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your vLLM endpoint
```

```bash
# .env
VLLM_BASE_URL=http://your-server:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen3.5-27B-FP8
ORGANIZATION=MyOrg
MODEL_ORGANIZATION=Alibaba
```

Any model served by vLLM works — Qwen, Llama, Mistral, DeepSeek, etc. Thinking models (Qwen3.5, QwQ, DeepSeek-R1) are handled automatically.

### Run

```bash
# Test on 3 questions with verbose output
python run.py --date 2025-12-21 --limit 3 -v

# Full tournament round
python run.py --date 2025-12-21

# Full run + score against actual outcomes
python run.py --date 2025-12-21 --evaluate
```

## Usage

| Command | What it does |
|---------|-------------|
| `python run.py --date 2025-12-21` | Full run: search + agentic LLM on all 500 questions |
| `python run.py --date 2025-12-21 --evaluate` | Run + score against resolved outcomes + leaderboard comparison |
| `python run.py --date 2025-12-21 --limit 10 -v` | Test 10 questions with verbose LLM reasoning |
| `python run.py --date 2025-12-21 --no-search` | Pure LLM, no web search |
| `python run.py --date 2025-12-21 --baseline --evaluate` | Always-0.5 baseline (test pipeline) |
| `python run.py --date 2025-12-21 --download-only` | Just download the question set |

## Fair Backtesting

When running on a historical date, the system automatically:

1. **Restricts web searches** to results published **before** the forecast due date (via Exa's `endPublishedDate` filter on `web_search_advanced_exa`)
2. **Instructs the LLM** to forecast as if today is that date

For live/future dates, no restrictions are applied. This ensures backtests measure actual forecasting ability, not hindsight.

## Output

```
submissions/
  2025-12-21.MyOrg.1.json    ← submission file (upload to ForecastBench)

data/                          ← cached downloads
  question_sets/
    2025-12-21-llm.json
  resolution_sets/
    2025-12-21_resolution_set.json
  leaderboard_tournament.csv
```

The submission file follows the [ForecastBench format](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench) and includes the model's reasoning for each forecast.

### Sample output

```
[1/500] manifold/6K4VOSrdR44W8AFB2taR: Will Tom Brady play in an NFL game again...
  Round 1: LLM requests 3 searches (ownership rules, selling stake, comeback news)
  Round 2: LLM analyzes results, gives answer
  -> [0.02] (117.5s)
```

## Evaluation

With `--evaluate`, the system:

1. Downloads resolution data (what actually happened)
2. Scores your forecasts using **Brier score** (0 = perfect, 0.25 = always-0.5, 1 = worst)
3. Computes **Brier Index** (100 = perfect, 50 = always-0.5)
4. Shows where you'd rank on the **full tournament leaderboard** (260+ entries)

```
======================================================================
YOUR RESULTS
======================================================================
       ALL:  Brier=0.1823  Index=57.3  (n=517)
    MARKET:  Brier=0.0912  Index=69.8  (n=82)
   DATASET:  Brier=0.1995  Index=55.3  (n=435)

TOURNAMENT LEADERBOARD COMPARISON
======================================================================
Rank  Team                     Model                          Overall
----------------------------------------------------------------------
1     ForecastBench            Superforecaster median            70.6
2     Cassi-AI                 ensemble_2_crowdadj               68.0
...
>YOU  YOU                      Qwen3.5-27B + Exa                 57.3
...
```

## ForecastBench Overview

[ForecastBench](https://www.forecastbench.org/) publishes 500 questions every 2 weeks:

| Category | Count | Sources | Format |
|----------|-------|---------|--------|
| **Market** | 250 | [Manifold](https://manifold.markets), [Metaculus](https://metaculus.com), [Polymarket](https://polymarket.com), [INFER](https://www.infer-pub.com) | 1 probability per question |
| **Dataset** | 250 | [FRED](https://fred.stlouisfed.org), [Yahoo Finance](https://finance.yahoo.com), [DBnomics](https://db.nomics.world), [Wikipedia](https://wikipedia.org), [ACLED](https://acleddata.com) | Up to 8 probabilities (7d to 10y horizons) |

All questions are binary: probability 0-1 that a statement resolves true. Scored by [Brier score](https://en.wikipedia.org/wiki/Brier_score).

**To participate officially**: email `forecastbench@forecastingresearch.org` to register. You get a GCP bucket and 24 hours per round to upload your submission.

### Historical dates for backtesting

| Date | Questions | Resolved | % | Notes |
|------|-----------|----------|---|-------|
| 2024-07-21 | 1000 | 7259/7579 | 96% | Oldest, most complete |
| 2025-10-26 | 500 | 777/924 | 84% | Best post-format-change |
| 2025-11-09 | 500 | 759/908 | 84% | |
| 2025-11-23 | 500 | 766/922 | 83% | |
| 2025-12-07 | 500 | 711/869 | 82% | |
| 2025-12-21 | 500 | 517/678 | 76% | |

Dataset questions always resolve fully. Market questions depend on when the underlying prediction market closes (some run until 2028+).

## Project Structure

```
forecastbench-agent/
├── run.py              # Main pipeline orchestration
├── forecaster.py       # Agentic LLM loop (reason → search → answer)
├── exa_search.py       # Exa MCP client with date-filtered search
├── questions.py        # Download/parse question sets and resolutions
├── submission.py       # Format and validate submission files
├── evaluate.py         # Brier scoring + leaderboard comparison
├── config.py           # Configuration
├── requirements.txt
├── .env.example
└── LICENSE
```

## Performance Notes

- ~2 min per question with a 27B thinking model + 2-4 search rounds
- 500 questions &asymp; 17 hours — run overnight
- [Exa MCP](https://exa.ai/mcp) hosted endpoint is free, no API key required
- Each question gets its own isolated context window (up to 128K tokens)
- Sequential processing (one question at a time) — designed for single-GPU inference

## Acknowledgments

- [ForecastBench](https://www.forecastbench.org/) by [Forecasting Research Institute](https://forecastingresearch.org/) — the benchmark itself
- [Exa](https://exa.ai) — free hosted MCP search endpoint
- [vLLM](https://github.com/vllm-project/vllm) — fast LLM inference

## Contributing

PRs welcome. Some ideas:

- Prompt engineering for better calibration
- Multi-model ensembling
- Smarter search query generation
- Caching to resume interrupted runs
- Parallel search while GPU is busy with inference

## License

[MIT](LICENSE)
