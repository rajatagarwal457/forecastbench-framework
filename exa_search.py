"""Exa web search via the free hosted MCP endpoint.

Uses web_search_advanced_exa for date-filtered searches (fair backtesting).
Multiple searches per question to build richer context.
"""

import asyncio
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import config

# Enable the advanced search tool via URL param
EXA_MCP_URL_WITH_ADVANCED = config.EXA_MCP_BASE + "?tools=web_search_advanced_exa"


async def _search(query: str, session: ClientSession,
                  end_date: str | None = None) -> str:
    """Call web_search_advanced_exa with optional date cutoff."""
    args: dict = {
        "query": query,
        "numResults": 8,
    }
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

    # Query 1: the question itself
    queries.append(question_text[:400])

    # Query 2: core topic as news search
    core = question_text.replace("Will ", "").replace("?", "").strip()
    if len(core) > 30:
        queries.append(f"{core[:200]} latest news")

    # Query 3: source-specific for dataset questions
    if source in config.DATASET_SOURCES and background and background != "N/A":
        queries.append(f"{background[:300]} forecast outlook")

    return queries[:3]


class ExaSearcher:
    """Manages an MCP connection to Exa with date-filtered search.

    For backtests, pass date_cutoff (YYYY-MM-DD) to only get results
    published before the forecast due date. For live runs, leave it None.
    """

    def __init__(self, date_cutoff: str | None = None):
        self._session: ClientSession | None = None
        self._cm_transport = None
        self._cm_session = None
        self._lock = asyncio.Lock()
        self.date_cutoff = date_cutoff

    async def connect(self):
        url = EXA_MCP_URL_WITH_ADVANCED
        if config.EXA_API_KEY:
            url += f"&exaApiKey={config.EXA_API_KEY}"
        self._cm_transport = streamable_http_client(url=url)
        read_stream, write_stream, _ = await self._cm_transport.__aenter__()
        self._cm_session = ClientSession(read_stream, write_stream)
        self._session = await self._cm_session.__aenter__()
        await self._session.initialize()

        # Verify advanced tool is available
        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        if "web_search_advanced_exa" not in tool_names:
            raise RuntimeError(
                f"web_search_advanced_exa not available. Got: {tool_names}"
            )

    async def close(self):
        if self._cm_session:
            await self._cm_session.__aexit__(None, None, None)
        if self._cm_transport:
            await self._cm_transport.__aexit__(None, None, None)
        self._session = None

    async def search(self, query: str) -> str:
        """Search with date cutoff applied."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")
        async with self._lock:
            return await _search(query, self._session, self.date_cutoff)

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
                    # Basic dedup by URL
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
            except Exception:
                continue

        return "\n\n".join(all_results)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()


async def test_search():
    """Quick test with date cutoff."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # Simulate backtest: only results before 2025-12-21
    async with ExaSearcher(date_cutoff="2025-12-21") as exa:
        tools = await exa._session.list_tools()
        print(f"Tools: {[t.name for t in tools.tools]}")
        result = await exa.search("Will Tesla have more autonomous rides than Waymo?")
        print(f"Total: {len(result):,} chars")
        print(result[:1000])


if __name__ == "__main__":
    asyncio.run(test_search())
