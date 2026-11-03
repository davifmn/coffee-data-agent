"""MCP server exposing the same read-only tools to any MCP client.

Register the command ``coffee-agent mcp`` in an MCP client (see README). The client's
own model then does the reasoning; this process only runs the tools.
"""

from __future__ import annotations

from functools import cache

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .config import open_toolbox
from .tools import MAX_ROWS_LIMIT, Toolbox

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

server = MCPServer(
    "coffee-data",
    instructions=(
        "Read-only access to the Grão Fino specialty-coffee marketplace: call describe_schema "
        "before writing SQL, run_sql for figures, search_knowledge for coffee knowledge and "
        "marketplace policies (cite the returned sections)."
    ),
)


@cache
def toolbox() -> Toolbox:
    return open_toolbox()


@server.tool(annotations=READ_ONLY)
def describe_schema() -> str:
    """Return the database schema, row counts and the date range of the orders."""
    return toolbox().describe_schema()


@server.tool(annotations=READ_ONLY)
def run_sql(query: str, max_rows: int = 100) -> str:
    """Run one read-only SQLite SELECT query and return the rows as JSON.

    Writes, PRAGMA and ATTACH are rejected; queries time out after a few seconds.
    """
    max_rows = max(1, min(max_rows, MAX_ROWS_LIMIT))
    outcome = toolbox().run_sql(query, max_rows)
    if outcome.is_error:
        # ToolError (unlike other exceptions) sends the message to the model, which can
        # then fix its query.
        raise ToolError(outcome.content)
    return outcome.content


@server.tool(annotations=READ_ONLY)
def search_knowledge(query: str, k: int = 4) -> str:
    """Search coffee knowledge and marketplace policies; each hit has a citation."""
    return toolbox().search_knowledge(query, max(1, min(k, 8))).content


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
