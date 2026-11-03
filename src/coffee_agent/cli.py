"""coffee-agent command line: build the demo DB, query it, ask the agent, run the MCP server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import default_db_path, open_toolbox
from .db import build_database


def _print_trace(result) -> None:
    for i, call in enumerate(result.tool_calls, start=1):
        marker = "error" if call.is_error else "ok"
        print(f"\n--- tool call {i}: {call.name} ({marker})", file=sys.stderr)
        print(json.dumps(call.input, ensure_ascii=False), file=sys.stderr)
        preview = call.output if len(call.output) < 600 else call.output[:600] + " …"
        print(preview, file=sys.stderr)
    u = result.usage
    print(
        f"\n[{result.model} · {result.turns} turns · in {u.input_tokens} · out {u.output_tokens}"
        f" · cached {u.cached_input_tokens}]",
        file=sys.stderr,
    )


def _make_agent(args):
    from .agent import CoffeeAgent, make_client

    return CoffeeAgent(
        make_client(), open_toolbox(args.db), model=args.model, reasoning_effort=args.effort
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coffee-agent", description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help=f"default: {default_db_path()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="(re)build the synthetic marketplace database")
    p_init.add_argument("--seed", type=int, default=42)

    p_sql = sub.add_parser("sql", help="run a read-only query (same guard rails as the agent)")
    p_sql.add_argument("query")

    p_search = sub.add_parser("search", help="search the knowledge base")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=3)

    sub.add_parser("schema", help="print the schema the agent sees")

    for name, help_text in [("ask", "ask one question"), ("chat", "interactive session")]:
        p = sub.add_parser(name, help=help_text)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--model", default=None, help="model id (default: $LLM_MODEL)")
        p.add_argument(
            "--effort", default=None, choices=["low", "medium", "high"], help="reasoning effort"
        )
        p.add_argument("--trace", action="store_true", help="print tool calls and token usage")

    sub.add_parser("mcp", help="run the MCP server on stdio")

    args = parser.parse_args(argv)

    if args.command == "init-db":
        path = build_database(args.db or default_db_path(), seed=args.seed)
        print(f"database written to {path}")
        return 0
    if args.command == "mcp":
        from .mcp_server import main as run_server

        run_server()
        return 0

    toolbox = open_toolbox(args.db)
    if args.command == "sql":
        outcome = toolbox.run_sql(args.query)
        print(outcome.content)
        return 1 if outcome.is_error else 0
    if args.command == "search":
        print(toolbox.search_knowledge(args.query, args.k).content)
        return 0
    if args.command == "schema":
        print(toolbox.describe_schema())
        return 0

    if args.model is None:
        from .agent import DEFAULT_MODEL

        args.model = DEFAULT_MODEL
    if not args.model:
        print("coffee-agent: no model configured; set LLM_MODEL or pass --model", file=sys.stderr)
        return 2
    agent = _make_agent(args)
    if args.command == "ask":
        result = agent.ask(args.question)
        print(result.answer)
        if args.trace:
            _print_trace(result)
        return 0

    history: list = []
    print("Ask about the marketplace (Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            continue
        result = agent.ask(question, history)
        history = result.messages
        print(result.answer)
        if args.trace:
            _print_trace(result)


if __name__ == "__main__":
    sys.exit(main())
