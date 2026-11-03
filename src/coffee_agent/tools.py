"""Tools shared by the agent and the MCP server.

Both front ends call the same ``Toolbox`` methods, so the SQL guard rails and the
output format are identical whether the model runs in our own agent loop or in any MCP
client (a desktop assistant, an IDE agent...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .db import QueryError, ReadOnlyDatabase
from .retrieval import KnowledgeBase

MAX_ROWS_LIMIT = 500


class RunSqlInput(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=1, max_length=8000)
    max_rows: int = Field(default=100, ge=1, le=MAX_ROWS_LIMIT)


class SearchKnowledgeInput(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=4, ge=1, le=8)


# Function tools in the chat-completions format. Arguments come back as JSON text and are
# validated with pydantic before anything runs.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run one read-only SQLite SELECT query against the Grão Fino marketplace "
                "database and get the rows back as JSON. Use it for every number you report: "
                "revenue, counts, averages, rankings, trends. Only SELECT/WITH is allowed; "
                "writes, PRAGMA and ATTACH are rejected. Dates are ISO strings, so use "
                "strftime('%Y', ordered_on) or BETWEEN for date filters. Results are capped at "
                "max_rows; aggregate in SQL instead of fetching raw rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SQLite SELECT query."},
                    "max_rows": {
                        "type": "integer",
                        "description": f"Maximum rows to return (1-{MAX_ROWS_LIMIT}). Default 100.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the coffee knowledge base and return the most relevant sections with a "
                "citation for each. It covers processing methods, Arabica varieties, Brazilian "
                "regions, SCA cupping scores, roast levels, brewing recipes and the marketplace's "
                "internal policies (partner requirements, commission, shipping, returns). Use it "
                "for definitions, recommendations and business rules, then cite the sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up, in English."},
                    "k": {"type": "integer", "description": "Number of sections (1-8). Default 4."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


class Toolbox:
    def __init__(self, db: ReadOnlyDatabase, kb: KnowledgeBase) -> None:
        self.db = db
        self.kb = kb

    # --- tool implementations -------------------------------------------------
    def run_sql(self, query: str, max_rows: int = 100) -> ToolOutcome:
        try:
            result = self.db.run(query, max_rows=max_rows)
        except QueryError as exc:
            return ToolOutcome(f"SQL error: {exc}", is_error=True)
        payload = {
            "columns": result.columns,
            "rows": [[_jsonable(v) for v in row] for row in result.rows],
            "row_count": len(result.rows),
            "truncated": result.truncated,
        }
        if result.truncated:
            payload["note"] = f"Only the first {max_rows} rows are shown; aggregate in SQL."
        return ToolOutcome(json.dumps(payload, ensure_ascii=False))

    def search_knowledge(self, query: str, k: int = 4) -> ToolOutcome:
        hits = self.kb.search(query, k=k)
        if not hits:
            return ToolOutcome("No matching sections. Try different keywords.")
        sections = [
            f"[{i}] {hit.chunk.citation} (score {hit.score})\n{hit.chunk.text}"
            for i, hit in enumerate(hits, start=1)
        ]
        return ToolOutcome("\n\n".join(sections))

    def describe_schema(self) -> str:
        return self.db.describe_schema()

    # --- dispatcher used by the agent loop ------------------------------------
    def execute(self, name: str, raw_input: Any) -> ToolOutcome:
        try:
            if name == "run_sql":
                args = RunSqlInput.model_validate(raw_input)
                return self.run_sql(args.query, args.max_rows)
            if name == "search_knowledge":
                args = SearchKnowledgeInput.model_validate(raw_input)
                return self.search_knowledge(args.query, args.k)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            return ToolOutcome(f"Invalid input for {name}: {problems}", is_error=True)
        return ToolOutcome(f"Unknown tool: {name}", is_error=True)
