"""Talk to the MCP server through a real MCP client over an in-memory transport."""

import asyncio
import json

import pytest
from mcp import Client

from coffee_agent import mcp_server
from coffee_agent.tools import Toolbox


@pytest.fixture(autouse=True)
def use_test_toolbox(monkeypatch, toolbox: Toolbox):
    monkeypatch.setattr(mcp_server, "toolbox", lambda: toolbox)


def with_client(coro_fn):
    async def run():
        async with Client(mcp_server.server) as client:
            return await coro_fn(client)

    return asyncio.run(run())


def call(name, arguments):
    return with_client(lambda client: client.call_tool(name, arguments))


def text_of(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def test_tools_are_registered_as_read_only():
    listed = with_client(lambda client: client.list_tools())
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == {"describe_schema", "run_sql", "search_knowledge"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False


def test_run_sql_over_mcp():
    result = call("run_sql", {"query": "SELECT COUNT(*) FROM coffees WHERE process = 'natural'"})
    assert not result.is_error
    assert json.loads(text_of(result))["rows"][0][0] > 0


def test_rejected_query_returns_a_readable_error():
    result = call("run_sql", {"query": "DELETE FROM orders"})
    assert result.is_error
    assert "only SELECT" in text_of(result)  # the model sees why, not a generic crash


def test_search_over_mcp():
    result = call("search_knowledge", {"query": "cold brew ratio", "k": 1})
    assert "brewing.md › Cold brew" in text_of(result)


def test_describe_schema_over_mcp():
    assert "CREATE TABLE reviews" in text_of(call("describe_schema", {}))
