"""Exa web search via the free hosted MCP endpoint.

Uses web_search_advanced_exa for date-filtered searches (fair backtesting).
Multiple searches per question to build richer context.
Auto-reconnects on connection drops and backs off on rate limits (429).
"""

import asyncio
import logging
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import config

log = logging.getLogger(__name__)

# Enable the advanced search tool via URL param
EXA_MCP_URL_WITH_ADVANCED = config.EXA_MCP_BASE + "?tools=web_search_advanced_exa"

MAX_SEARCH_RETRIES = 5
RATE_LIMIT_BACKOFF = [30, 60, 120, 300, 600]  # seconds to wait on 429


async def _search(query: str, session: ClientSession,
                  end_date: str | None = None,
                  max_chars: int = 0) -> str:
    """Call web_search_advanced_exa with optional date cutoff."""
    args: dict = {
        "query": query,
        "numResults": 8,
    }
    if max_chars > 0:
        args["contextMaxCharacters"] = max_chars
    if end_date:
        args["endPublishedDate"] = end_date

    result = await session.call_tool("web_search_advanced_exa", arguments=args)
    parts = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            parts.append(content.text)
    return "\n".join(parts)


def _build_search_queries(question_text: str, background: str, source: str) -> list[str]:
    """Build multiple search queries for diverse context."""
    queries = []

    queries.append(question_text[:400])

    core = question_text.replace("Will ", "").replace("?", "").strip()
    if len(core) > 30:
        queries.append(f"{core[:200]} latest news")

    if source in config.DATASET_SOURCES and background and background != "N/A":
        queries.append(f"{background[:300]} forecast outlook")

    return queries[:3]


def _is_rate_limit(error: Exception) -> bool:
    """Check if the error is a 429 rate limit."""
    return "429" in str(error) or "Too Many Requests" in str(error)


class ExaSearcher:
    """Manages an MCP connection to Exa with date-filtered search.

    Auto-reconnects on connection drops. Backs off on 429 rate limits.
    """

    def __init__(self, date_cutoff: str | None = None):
        self._session: ClientSession | None = None
        self._cm_transport = None
        self._cm_session = None
        self._lock = asyncio.Lock()
        self.date_cutoff = date_cutoff
        self._connected = False

    def _build_url(self) -> str:
        url = EXA_MCP_URL_WITH_ADVANCED
        if config.EXA_API_KEY:
            url += f"&exaApiKey={config.EXA_API_KEY}"
        return url

    async def connect(self):
        url = self._build_url()
        self._cm_transport = streamable_http_client(url=url)
        read_stream, write_stream, _ = await self._cm_transport.__aenter__()
        self._cm_session = ClientSession(read_stream, write_stream)
        self._session = await self._cm_session.__aenter__()
        await self._session.initialize()

        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        if "web_search_advanced_exa" not in tool_names:
            raise RuntimeError(
                f"web_search_advanced_exa not available. Got: {tool_names}"
            )
        self._connected = True

    async def close(self):
        self._connected = False
        try:
            if self._cm_session:
                await self._cm_session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._cm_transport:
                await self._cm_transport.__aexit__(None, None, None)
        except Exception:
            pass
        self._session = None
        self._cm_session = None
        self._cm_transport = None

    async def _reconnect(self):
        log.warning("Reconnecting to Exa MCP...")
        await self.close()
        await self.connect()
        log.info("Reconnected to Exa MCP.")

    async def search(self, query: str) -> str:
        """Search with auto-reconnect and rate limit backoff."""
        async with self._lock:
            for attempt in range(MAX_SEARCH_RETRIES):
                try:
                    if not self._connected or not self._session:
                        await self._reconnect()
                    return await _search(query, self._session, self.date_cutoff)

                except Exception as e:
                    if _is_rate_limit(e):
                        wait = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                        log.warning(f"Rate limited (429). Waiting {wait}s before retry "
                                    f"({attempt+1}/{MAX_SEARCH_RETRIES})...")
                        await asyncio.sleep(wait)
                        # Reconnect after rate limit since the connection may be dead
                        await self._reconnect()
                    else:
                        log.warning(f"Search attempt {attempt+1}/{MAX_SEARCH_RETRIES} "
                                    f"failed: {e}")
                        if attempt < MAX_SEARCH_RETRIES - 1:
                            await self._reconnect()
                        else:
                            log.error(f"Search failed after {MAX_SEARCH_RETRIES} attempts")
                            raise

            # Should not reach here, but return empty if somehow we do
            return ""

    async def search_for_question(self, question_text: str, background: str = "",
                                  source: str = "") -> str:
        """Run multiple searches for a question, combine all results."""
        queries = _build_search_queries(question_text, background, source)
        all_results = []
        seen_urls = set()

        for i, query in enumerate(queries):
            try:
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
            except Exception as e:
                log.warning(f"Search for question failed on query {i+1}: {e}")
                continue

        return "\n\n".join(all_results)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()
