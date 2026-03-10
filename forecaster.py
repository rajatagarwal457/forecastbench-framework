"""Agentic forecaster: the LLM iteratively searches and reasons.

Flow per question:
  1. System prompt tells LLM it can search the web via SEARCH("query")
  2. Initial user message: question metadata + first batch of search results
  3. Loop:
     a. LLM responds — may contain SEARCH("...") requests and/or reasoning
     b. We execute any searches, feed results back as a new user message
     c. Repeat until LLM outputs ANSWER: *0.XX* or we hit max rounds
  4. Extract probabilities from the final answer

Each question gets a completely fresh conversation (isolated context).
"""

import re
from openai import AsyncOpenAI

import config
from questions import Question

MAX_ROUNDS = 5  # max search-reason cycles before forcing an answer

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
You combine base rates, reference classes, and up-to-date information to produce \
well-calibrated probability estimates.

You have access to a web search tool. To search, write on its own line:
SEARCH("your query here")

You can make multiple searches in one response. After each round, you will receive \
the search results and can search again or give your final answer.

Strategy:
- Start by analyzing the question and the initial search results provided.
- Think about what additional information would help you make a better forecast.
- Search for base rates, recent news, expert opinions, historical precedents, etc.
- Consider multiple perspectives and reference classes.
- When you have enough information, give your final probability.

When you are ready to give your final answer, write:
ANSWER: *0.XX*

For dataset questions with multiple resolution dates, give one answer per date:
ANSWER: *0.XX* *0.YY* *0.ZZ* ...

You MUST eventually give a probability between 0 and 1. If unsure, estimate a base rate. \
Do NOT refuse to answer. Do NOT output 0.0 or 1.0 — use 0.01 to 0.99.

{date_note}"""


def get_system_prompt(date_cutoff: str | None = None) -> str:
    if date_cutoff:
        note = (
            f"IMPORTANT: Your web searches are restricted to results published before "
            f"{date_cutoff}. You are forecasting as if today is {date_cutoff}. "
            f"Do NOT use knowledge of events after this date."
        )
    else:
        note = ""
    return SYSTEM_PROMPT_TEMPLATE.format(date_note=note)


def _format_question_message(q: Question, initial_search: str, today: str) -> str:
    """Build the initial user message with question + first search results."""
    parts = []

    parts.append(f"TODAY'S DATE: {today}")
    parts.append("")

    # Question details
    parts.append(f"QUESTION: {q.question}")
    if q.background and q.background != "N/A":
        parts.append(f"BACKGROUND: {q.background}")
    if q.resolution_criteria and q.resolution_criteria != "N/A":
        parts.append(f"RESOLUTION CRITERIA: {q.resolution_criteria}")
    if q.freeze_datetime_value and q.freeze_datetime_value != "N/A":
        parts.append(f"CURRENT VALUE ({q.freeze_datetime}): {q.freeze_datetime_value}")
    if q.freeze_datetime_value_explanation and q.freeze_datetime_value_explanation != "N/A":
        parts.append(f"VALUE EXPLANATION: {q.freeze_datetime_value_explanation}")
    if q.url and q.url != "N/A":
        parts.append(f"SOURCE URL: {q.url}")

    if q.is_market:
        parts.append(f"TYPE: Market question (single probability)")
    else:
        parts.append(f"TYPE: Dataset question")
        parts.append(f"RESOLUTION DATES: {q.resolution_dates}")
        n = len(q.resolution_dates) if q.resolution_dates else 1
        parts.append(f"(You must provide {n} probabilities, one per resolution date)")

    parts.append("")

    if initial_search:
        parts.append("=== INITIAL WEB SEARCH RESULTS ===")
        parts.append(initial_search)
        parts.append("=== END INITIAL RESULTS ===")
        parts.append("")

    parts.append(
        "Analyze the question and search results above. Search for any additional "
        "information you need, then provide your forecast. Use SEARCH(\"query\") to "
        "search, and ANSWER: *probability* when ready."
    )

    return "\n".join(parts)


def extract_searches(text: str) -> list[str]:
    """Extract SEARCH("...") queries from LLM output."""
    return re.findall(r'SEARCH\("([^"]+)"\)', text)


def extract_probabilities(text: str, expected_count: int = 1) -> list[float] | None:
    """Extract probabilities from ANSWER: *0.XX* format.

    Returns None if no ANSWER line found (LLM wants to keep searching).
    Returns list of floats if answer found.
    """
    # Look for ANSWER: line first
    answer_match = re.search(r'ANSWER:\s*(.*)', text, re.IGNORECASE)
    if answer_match:
        answer_line = answer_match.group(1)
        starred = re.findall(r"\*\s*(0?\.\d+|1\.0*|0|1)\s*\*", answer_line)
        if starred:
            probs = [_clamp(float(x)) for x in starred]
            return _pad_or_trim(probs, expected_count)

    # Also check for standalone *0.XX* patterns (model might skip the ANSWER: prefix)
    starred = re.findall(r"\*\s*(0?\.\d+|1\.0*|0|1)\s*\*", text)
    if starred:
        probs = [_clamp(float(x)) for x in starred]
        return _pad_or_trim(probs, expected_count)

    return None  # no answer yet


def force_extract_probabilities(text: str, expected_count: int = 1) -> list[float]:
    """Last-resort extraction: find any decimal numbers that look like probabilities."""
    # Try *0.XX* first
    starred = re.findall(r"\*\s*(0?\.\d+|1\.0*|0|1)\s*\*", text)
    if starred:
        probs = [_clamp(float(x)) for x in starred]
        return _pad_or_trim(probs, expected_count)

    # Any decimal in [0, 1]
    decimals = re.findall(r"(?<!\d)(0\.\d+|1\.0+|0\.0+)(?!\d)", text)
    if decimals:
        probs = [_clamp(float(x)) for x in decimals]
        return _pad_or_trim(probs, expected_count)

    return [0.5] * expected_count


def _clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


def _pad_or_trim(probs: list[float], n: int) -> list[float]:
    if len(probs) >= n:
        return probs[:n]
    fill = probs[-1] if probs else 0.5
    return probs + [fill] * (n - len(probs))


class Forecaster:
    """Agentic forecaster: LLM reasons and searches iteratively.

    Each question gets a fresh conversation. The LLM can request web searches
    via SEARCH("query") and gives its final answer via ANSWER: *0.XX*.
    """

    def __init__(self, searcher=None, verbose: bool = False, date_cutoff: str | None = None):
        self.client = AsyncOpenAI(
            base_url=config.VLLM_BASE_URL,
            api_key=config.VLLM_API_KEY,
        )
        self.model = config.VLLM_MODEL
        self.searcher = searcher  # ExaSearcher instance (or None)
        self.verbose = verbose
        self.system_prompt = get_system_prompt(date_cutoff)

    async def _call_llm(self, messages: list[dict]) -> tuple[str, str]:
        """Call the LLM and return (content, reasoning)."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning", None) or ""
        return content, reasoning

    async def forecast(self, q: Question, initial_search: str, today: str) -> list[dict]:
        """Run the agentic loop for a single question.

        Returns a list of forecast dicts ready for submission.
        """
        expected_count = 1 if q.is_market else len(q.resolution_dates or [1])

        # Build initial conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": _format_question_message(q, initial_search, today)},
        ]

        all_reasoning = []  # collect reasoning across rounds

        for round_num in range(1, MAX_ROUNDS + 1):
            if self.verbose:
                print(f"    Round {round_num}/{MAX_ROUNDS}...")

            try:
                content, reasoning = await self._call_llm(messages)
            except Exception as e:
                print(f"  LLM error for {q.source}/{q.id} round {round_num}: {e}")
                break

            if self.verbose:
                print(f"    Content: {content[:200]!r}")
                if reasoning:
                    print(f"    Reasoning: {reasoning[:200]!r}")

            # Combine content and reasoning for extraction
            full_text = content
            if reasoning:
                all_reasoning.append(reasoning)
                if not content:
                    full_text = reasoning

            # Check if LLM gave a final answer
            probs = extract_probabilities(full_text, expected_count)
            if probs is not None:
                if self.verbose:
                    print(f"    Got answer in round {round_num}: {probs}")
                return self._build_forecasts(q, probs, full_text, all_reasoning)

            # Check for search requests
            searches = extract_searches(full_text)
            if not searches and reasoning:
                searches = extract_searches(reasoning)

            if searches and self.searcher:
                # Execute searches and feed results back
                search_results = []
                for query in searches[:3]:  # cap at 3 searches per round
                    if self.verbose:
                        print(f"    Searching: {query[:80]}")
                    try:
                        result = await self.searcher.search(query)
                        search_results.append(f"Results for \"{query}\":\n{result}")
                    except Exception as e:
                        search_results.append(f"Search failed for \"{query}\": {e}")

                # Add assistant response and search results to conversation
                messages.append({"role": "assistant", "content": content or reasoning})
                messages.append({
                    "role": "user",
                    "content": (
                        "Here are the search results:\n\n"
                        + "\n\n---\n\n".join(search_results)
                        + "\n\nContinue your analysis. Search more if needed, "
                        "or give your ANSWER: *probability* when ready."
                    ),
                })
            else:
                # No searches and no answer — ask LLM to commit
                messages.append({"role": "assistant", "content": content or reasoning})
                messages.append({
                    "role": "user",
                    "content": (
                        "Please give your final forecast now. "
                        "Write ANSWER: *probability* (between 0 and 1)."
                    ),
                })

        # Exhausted all rounds — force extract from everything we have
        if self.verbose:
            print(f"    Max rounds reached, force-extracting...")

        all_text = " ".join(m.get("content", "") for m in messages if m["role"] == "assistant")
        all_text += " " + " ".join(all_reasoning)
        probs = force_extract_probabilities(all_text, expected_count)
        return self._build_forecasts(q, probs, all_text, all_reasoning)

    def _build_forecasts(
        self, q: Question, probs: list[float], content: str, reasoning_parts: list[str],
    ) -> list[dict]:
        """Build forecast dicts from extracted probabilities."""
        reasoning_summary = content[:500] if content else None
        if not reasoning_summary and reasoning_parts:
            reasoning_summary = reasoning_parts[-1][:500]

        if q.is_market:
            return [{
                "id": q.id,
                "source": q.source,
                "forecast": probs[0],
                "resolution_date": None,
                "reasoning": reasoning_summary,
            }]
        else:
            dates = q.resolution_dates or [None]
            return [
                {
                    "id": q.id,
                    "source": q.source,
                    "forecast": prob,
                    "resolution_date": date,
                    "reasoning": reasoning_summary,
                }
                for date, prob in zip(dates, probs)
            ]
