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

EXA_MCP_URL_WITH_ADVANCED = config.EXA_MCP_BASE + "?tools=web_search_advanced_exa"

MAX_SEARCH_RETRIES = 5
RATE_LIMIT_BACKOFF = [30, 60, 120, 300, 600]


async def _search(query: str, session: ClientSession,
                  end_date: str | None = None,
                  max_chars: int = 0) -> str:
    args: dict = {"query": query, "numResults": 8}
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
    """Manages an MCP connection to Exa with date-filtered search.

    On connection death (429, CancelledError, etc.), the old connection
    is ABANDONED (references dropped, no __aexit__) and a completely
    fresh one is created. This prevents CancelledError from dead MCP
    transports from poisoning the asyncio event loop.
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
        """Create a fresh MCP connection."""
        url = self._build_url()
        self._cm_transport = streamable_http_client(url=url)
        read_stream, write_stream, _ = await self._cm_transport.__aenter__()
        self._cm_session = ClientSession(read_stream, write_stream)
        self._session = await self._cm_session.__aenter__()
        await self._session.initialize()

        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        if "web_search_advanced_exa" not in tool_names:
            raise RuntimeError(f"web_search_advanced_exa not available. Got: {tool_names}")
        self._connected = True

    def _abandon(self):
        """Drop all references to a dead connection WITHOUT calling __aexit__.

        This is the key fix: calling __aexit__ on a dead MCP session propagates
        CancelledError through anyio cancel scopes, poisoning the entire event loop.
        We just let the garbage collector clean up the dead objects.
        """
        self._connected = False
        self._session = None
        self._cm_session = None
        self._cm_transport = None

    async def close(self):
        """Graceful close for healthy connections (e.g. end of run)."""
        self._connected = False
        self._session = None
        for cm in [self._cm_session, self._cm_transport]:
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except BaseException:
                    pass
        self._cm_session = None
        self._cm_transport = None

    async def _fresh_connect(self):
        """Abandon dead connection and create a completely fresh one."""
        self._abandon()
        log.info("Creating fresh Exa MCP connection...")
        await self.connect()
        log.info("Fresh connection established.")

    async def search(self, query: str) -> str:
        """Search with auto-reconnect and rate limit backoff.

        On any failure, the dead connection is abandoned (not closed)
        and a fresh one is created. This prevents cancel scope pollution.
        """
        async with self._lock:
            last_error = None
            for attempt in range(MAX_SEARCH_RETRIES):
                try:
                    if not self._connected or not self._session:
                        await self._fresh_connect()
                    return await _search(query, self._session, self.date_cutoff)

                except BaseException as e:
                    last_error = e
                    self._abandon()

                    # The dead MCP cancel scope marks our task as cancelled.
                    # Uncancel it so we can actually continue (sleep, reconnect, etc).
                    task = asyncio.current_task()
                    if task and task.cancelling():
                        task.uncancel()

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
