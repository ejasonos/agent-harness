import sys
from types import SimpleNamespace

import pytest

from tools.web.search import web_search


def test_web_search_passes_query_and_result_count_to_tavily(
    monkeypatch,
):
    received = {}
    expected_result = {
        "query": "python packaging",
        "results": [{"title": "Packaging guide"}],
    }

    class FakeTavilySearch:
        def __init__(self, *, api_key, max_results):
            received["api_key"] = api_key
            received["max_results"] = max_results

        def invoke(self, query):
            received["query"] = query
            return expected_result

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_tavily",
        SimpleNamespace(TavilySearch=FakeTavilySearch),
    )

    result = web_search("  python packaging  ", max_results=3)

    assert result == expected_result
    assert received == {
        "api_key": "test-key",
        "max_results": 3,
        "query": "python packaging",
    }


@pytest.mark.parametrize(
    ("query", "max_results", "message"),
    [
        ("  ", 5, "query cannot be empty"),
        ("python", 0, "between 1 and 20"),
        ("python", 21, "between 1 and 20"),
    ],
)
def test_web_search_validates_arguments(query, max_results, message):
    with pytest.raises(ValueError, match=message):
        web_search(query, max_results=max_results)


def test_web_search_requires_tavily_api_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY is not configured"):
        web_search("python packaging")