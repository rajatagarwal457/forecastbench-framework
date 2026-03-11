"""Exa web search via the free hosted MCP endpoint.

Uses web_search_advanced_exa for date-filtered searches (fair backtesting).
Multiple searches per question to build richer context.

Each search runs in its own thread with its own event loop and its own
MCP connection. This is the only reliable way to prevent anyio cancel
scopes from poisoning the main asyncio event loop on 429s/errors.
"""

import asyncio
import logging
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import config

log = logging.getLogger(__name__)

EXA_MCP_URL_WITH_ADVANCED = config.EXA_MCP_BASE + "?tools=web_search_advanced_exa"

MAX_SEARCH_RETRIES = 5
RATE_LIMIT_BACKOFF = [30, 60, 120, 300, 600]


def _build_url() -> str:
    url = EXA_MCP_URL_WITH_ADVANCED
    if config.EXA_API_KEY:
        url += f"&exaApiKey={config.EXA_API_KEY}"
    return url


async def _isolated_search(url: str, query: str, end_date: str | None) -> str:
    """Run a single search on a fresh MCP connection. Runs in its own event loop."""
    args: dict = {"query": query, "numResults": 8}
    if end_date:
        args["endPublishedDate"] = end_date

    async with streamable_http_client(url=url) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("web_search_advanced_exa", arguments=args)

    parts = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            parts.append(content.text)
    return "\n".join(parts)


def _sync_search(url: str, query: str, end_date: str | None) -> str:
    """Synchronous wrapper — creates its own event loop in the calling thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_isolated_search(url, query, end_date))
    finally:
        loop.close()


def _build_search_queries(question_text: str, background: str, source: str) -> list[str]:
    queries = []
    queries.append(question_text[:400])
    core = question_text.replace("Will ", "").replace("?", "").strip()
    if len(core) > 30:
        queries.append(f"{core[:200]} latest news")
    if source in config.DATASET_SOURCES and background and background != "N/A":
        queries.append(f"{background[:300]} forecast outlook")
    return queries[:3]


def _is_rate_limit(error: BaseException) -> bool:
    return "429" in str(error) or "Too Many Requests" in str(error)


class ExaSearcher:
    """Exa web search with complete isolation per search call.

    Each search runs in its own thread with its own event loop and MCP
    connection. This prevents anyio cancel scope pollution from killing
    the main event loop on 429s or connection errors.
    """

    def __init__(self, date_cutoff: str | None = None):
        self.date_cutoff = date_cutoff
        self._url = _build_url()

    async def connect(self):
        """Verify the endpoint works with a test connection."""
        # Just test that we can connect — each search creates its own connection
        result = await asyncio.to_thread(_sync_search, self._url, "test", self.date_cutoff)
        if result:
            log.info("  Exa MCP endpoint verified.")

    async def close(self):
        """Nothing to close — each search manages its own connection."""
        pass

    async def search(self, query: str) -> str:
        """Search with retry and rate limit backoff.

        Each attempt runs in a separate thread with a fresh event loop,
        so MCP/anyio errors are completely isolated.
        """
        last_error = None
        for attempt in range(MAX_SEARCH_RETRIES):
            try:
                return await asyncio.to_thread(
                    _sync_search, self._url, query, self.date_cutoff,
                )
            except Exception as e:
                last_error = e
                if _is_rate_limit(e):
                    wait = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                    log.warning(f"Rate limited (429). Waiting {wait}s "
                                f"({attempt+1}/{MAX_SEARCH_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    log.warning(f"Search attempt {attempt+1}/{MAX_SEARCH_RETRIES} "
                                f"failed: {type(e).__name__}: {e}")
                    if attempt < MAX_SEARCH_RETRIES - 1:
                        await asyncio.sleep(2)

        log.error(f"Search failed after {MAX_SEARCH_RETRIES} attempts: {last_error}")
        return ""

    async def search_for_question(self, question_text: str, background: str = "",
                                  source: str = "") -> str:
        queries = _build_search_queries(question_text, background, source)
        all_results = []
        seen_urls = set()

        for i, query in enumerate(queries):
            result = await self.search(query)
            if result:
                lines = result.split("\n")
                filtered = []
                for line in lines:
                    if line.startswith("URL: "):
                        url = line[5:].strip()
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                    filtered.append(line)
                block = "\n".join(filtered)
                all_results.append(f"--- Search {i+1}: {query[:80]} ---\n{block}")

        return "\n\n".join(all_results)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()
