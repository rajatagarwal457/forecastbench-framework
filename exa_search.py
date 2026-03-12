"""Exa web search via the free hosted MCP endpoint.

Uses web_search_advanced_exa for date-filtered searches (fair backtesting).
Multiple searches per question to build richer context.

Each search runs in its own thread with its own event loop and fresh
MCP connection. Optionally routes through Tor SOCKS5 proxy for IP
rotation to avoid rate limits.
"""

import asyncio
import logging
import os

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import config

log = logging.getLogger(__name__)

EXA_MCP_URL_WITH_ADVANCED = config.EXA_MCP_BASE + "?tools=web_search_advanced_exa"

MAX_SEARCH_RETRIES = 5
RATE_LIMIT_BACKOFF = [30, 60, 120, 300, 600]

# Tor SOCKS5 proxy — set TOR_PROXY env var or defaults to direct connection
TOR_PROXY = os.getenv("TOR_PROXY", "")  # e.g. "socks5://127.0.0.1:9050"
TOR_CONTROL_PORT = int(os.getenv("TOR_CONTROL_PORT", "9051"))
TOR_CONTROL_PASSWORD = os.getenv("TOR_CONTROL_PASSWORD", "")


def _rotate_tor_circuit():
    """Request a new Tor circuit (new exit IP). Requires stem package."""
    if not TOR_PROXY or not TOR_CONTROL_PASSWORD:
        return
    try:
        from stem import Signal
        from stem.control import Controller
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            controller.authenticate(password=TOR_CONTROL_PASSWORD)
            controller.signal(Signal.NEWNYM)
            log.info("Rotated Tor circuit (new IP).")
    except Exception as e:
        log.warning(f"Failed to rotate Tor circuit: {e}")


def _build_url() -> str:
    url = EXA_MCP_URL_WITH_ADVANCED
    if config.EXA_API_KEY:
        url += f"&exaApiKey={config.EXA_API_KEY}"
    return url


def _make_http_client() -> httpx.AsyncClient:
    """Create an httpx client, optionally routed through Tor."""
    kwargs = {
        "timeout": httpx.Timeout(30.0, read=300.0),
        "follow_redirects": True,
    }
    if TOR_PROXY:
        kwargs["proxy"] = TOR_PROXY
    return httpx.AsyncClient(**kwargs)


async def _isolated_search(url: str, query: str, end_date: str | None) -> str:
    """Run a single search on a fresh MCP connection in its own event loop."""
    args: dict = {"query": query, "numResults": 8}
    if end_date:
        args["endPublishedDate"] = end_date

    async with _make_http_client() as http_client:
        async with streamable_http_client(url=url, http_client=http_client) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool("web_search_advanced_exa", arguments=args)

    parts = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            parts.append(content.text)
    return "\n".join(parts)


SEARCH_TIMEOUT = 60  # seconds — kill hung connections


def _sync_search(url: str, query: str, end_date: str | None) -> str:
    """Synchronous wrapper — creates its own event loop in the calling thread."""
    import warnings
    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
    warnings.filterwarnings("ignore", message=".*Enable tracemalloc.*")

    # Suppress EpollSelector and other asyncio debug logs in this thread
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    loop = asyncio.new_event_loop()
    try:
        loop.set_exception_handler(lambda l, c: None)
        # Timeout so hung connections don't block forever
        return loop.run_until_complete(
            asyncio.wait_for(_isolated_search(url, query, end_date), timeout=SEARCH_TIMEOUT)
        )
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


def _unwrap_exception(error: BaseException) -> BaseException:
    """Unwrap single-exception ExceptionGroups (MCP SDK bug #2114)."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _is_rate_limit(error: BaseException) -> bool:
    """Check for 429 rate limit, including inside ExceptionGroups."""
    error = _unwrap_exception(error)
    full = repr(error)
    if "429" in full or "Too Many Requests" in full:
        return True
    if isinstance(error, BaseExceptionGroup):
        for sub in error.exceptions:
            if _is_rate_limit(sub):
                return True
    if error.__cause__:
        return _is_rate_limit(error.__cause__)
    return False


class ExaSearcher:
    """Exa web search with complete isolation per search call.

    Each search runs in its own thread with its own event loop, httpx client,
    and MCP connection. Optionally routes through Tor for IP rotation.
    """

    def __init__(self, date_cutoff: str | None = None):
        self.date_cutoff = date_cutoff
        self._url = _build_url()
        if TOR_PROXY:
            log.info(f"  Exa searches will route through Tor proxy: {TOR_PROXY}")

    async def connect(self):
        pass

    async def close(self):
        pass

    async def search(self, query: str) -> str:
        """Search with retry and rate limit backoff."""
        last_error = None
        for attempt in range(MAX_SEARCH_RETRIES):
            try:
                return await asyncio.to_thread(
                    _sync_search, self._url, query, self.date_cutoff,
                )
            except Exception as e:
                last_error = e
                if _is_rate_limit(e):
                    # Rotate Tor circuit for a fresh IP before retrying
                    _rotate_tor_circuit()
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
