from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


def web_search(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the web with Tavily and return its structured results."""

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")

    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured. Add it to the environment "
            "or the project .env file."
        )

    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise RuntimeError(
            "Tavily search requires the langchain-tavily package. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    search_tool = TavilySearch(
        api_key=api_key,
        max_results=max_results,
    )
    result = search_tool.invoke(query)

    if isinstance(result, dict):
        return result

    return {"results": result}