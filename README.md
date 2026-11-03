# coffee-data-agent

[![CI](https://github.com/davifmn/coffee-data-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/davifmn/coffee-data-agent/actions/workflows/ci.yml)

An AI data analyst for a specialty-coffee marketplace: an LLM tool-use agent with text-to-SQL,
RAG and an MCP server. It works with any model behind an OpenAI-compatible chat API.

Ask a business question in plain English or Portuguese. The agent writes **read-only SQL**
against the marketplace database, looks up **coffee knowledge and company policies** in a
small RAG index, and answers with the numbers, the SQL it ran and citations. The same tools
are also exposed as an **MCP server**, so any MCP client (a desktop assistant, an IDE agent)
can use them directly.

```text
"Which partner farms are below the minimum average rating required by our partner policy?"

 1. search_knowledge  → marketplace_policies.md › Partner farm requirements  (minimum 4.0 stars)
 2. run_sql           → SELECT s.name, AVG(r.rating) ... GROUP BY s.id HAVING AVG(r.rating) < 4.0
 3. answer            → the farms and their averages, the SQL, and the policy citation
```

The data is synthetic: "Grão Fino" is a fictional Brazilian marketplace (12 partner farms,
52 coffees, 450 customers, 3,600 orders from 2024 to 2026, about 2,600 reviews) generated
from a fixed seed.

## Architecture

```
            ┌────────────── coffee-agent ask / chat ──────────────┐
question ─▶ │ CoffeeAgent (manual tool-use loop, any chat model)   │ ─▶ answer + SQL + citations
            └──────────────┬───────────────────────┬──────────────┘
                 run_sql   │                       │  search_knowledge
                           ▼                       ▼
            ┌──────────────────────┐   ┌──────────────────────────┐
            │ ReadOnlyDatabase     │   │ KnowledgeBase (BM25)      │
            │ SQLite, 5 guard rails│   │ 7 markdown docs           │
            └──────────────────────┘   └──────────────────────────┘
                           ▲                       ▲
                           └────── Toolbox ────────┘
                                      ▲
                          coffee-agent mcp (MCP server, stdio)
                                      ▲
                              any MCP client
```

| Module | What it does |
|---|---|
| `agent.py` | The agent loop: parallel tool calls, malformed arguments, content filtering, truncation and a turn limit. |
| `tools.py` | Function-tool definitions (JSON schemas) plus pydantic validation of every tool input before it runs. |
| `db.py` | Seeded data generator and `ReadOnlyDatabase`. |
| `retrieval.py` | BM25 over the knowledge base, split on headings so every hit can be cited as `file.md › Heading`. |
| `mcp_server.py` | The same tools as an MCP server, marked read-only. |
| `evaluation.py`, `evals/` | 15-question eval set with automatic grading. |

## Design decisions

**Model-written SQL is untrusted input.** `ReadOnlyDatabase` stacks independent defences, and
`tests/test_db.py` attacks each one:

1. the SQLite file is opened with `mode=ro`;
2. `PRAGMA query_only = ON`;
3. an authorizer callback allows only reads. ATTACH, PRAGMA, writes and `load_extension` are denied, so `WITH x AS (...) DELETE ...` and `SELECT load_extension(...)` fail even though they start like a read;
4. one statement per call (`SELECT 1; DROP TABLE orders` is rejected);
5. a timeout through a progress handler (a runaway recursive CTE is interrupted), plus a row cap.

**The agent loop is hand-written.** I didn't use an agent framework. Every tool call is
recorded with its input and output, which the CLI's `--trace` and the eval grader use. Inputs are
validated with pydantic, and bad input or a rejected query goes back to the model as an
error tool result, so the model can fix its query instead of the program crashing.

**Provider-neutral model access** (`agent.py`):

- It uses the chat-completions API with function tools, the de facto standard, so the same code
  runs against a hosted model or a local server (vLLM, Ollama, LM Studio). `LLM_BASE_URL`,
  `LLM_API_KEY` and `LLM_MODEL` choose the endpoint and the model; nothing is hard-coded.
- **Prompt-cache friendly:** the system prompt (instructions + schema) never changes and the
  history is append-only, so providers that cache prompt prefixes reuse them on every turn of
  the tool loop. Cached tokens are counted separately in the usage report and in the eval cost.
- Optional `reasoning_effort` for reasoning models (`--effort`).
- No forced `tool_choice`: the prompt says when to use each tool.

**BM25 instead of embeddings for RAG.** The corpus is small and full of exact terms (process
names, varieties, regions, policy numbers). Lexical search finds them reliably, runs offline in
milliseconds and needs no vector database or embedding API. Accents are folded so `Caparaó`
matches `caparao`.

**Evals without hand-typed answers.** Each case has a reference SQL query, and the expected answer
is whatever that query returns, so the eval stays valid if the data is regenerated. Checks
include execution accuracy (one of the agent's queries returns the same rows), numbers in the
answer (both `31,202.99` and `31.202,99`), names mentioned and knowledge sources consulted. Some
cases need both tools: "which farms break the rating policy" only passes if the agent read the
policy *and* queried the data.

## Quick start

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

coffee-agent init-db                     # builds ~/.local/share/coffee-data-agent/marketplace.db
coffee-agent schema                      # what the model sees
coffee-agent sql "SELECT channel, COUNT(*) FROM orders GROUP BY channel"
coffee-agent search "pulped natural"
```

Asking the agent needs a model behind an OpenAI-compatible API:

```sh
export LLM_MODEL=<model id>
export LLM_API_KEY=<key>
export LLM_BASE_URL=<endpoint>          # omit for the default OpenAI endpoint

coffee-agent ask "What was the revenue from delivered orders in 2025?" --trace
coffee-agent ask "Quanto a plataforma ganhou de comissão em 2025?"
coffee-agent chat                        # multi-turn session
```

### Use the tools from an MCP client

Register the stdio server in the client's configuration:

```json
{
  "mcpServers": {
    "coffee-data": { "command": "/path/to/.venv/bin/coffee-agent", "args": ["mcp"] }
  }
}
```

Tools: `describe_schema`, `run_sql`, `search_knowledge`.

## Evaluation

```sh
python evals/run_eval.py                 # all 15 cases: pass/fail and tokens
python evals/run_eval.py --only weak_partners commission_2025 --effort high
python evals/run_eval.py --price-input 3 --price-output 15 --price-cached 0.3   # cost estimate
```

Results go to `evals/results/`. The eval calls the configured model and may cost money. In CI it
runs only when you trigger it by hand (`workflow_dispatch` with `run_eval`); it needs the
`LLM_API_KEY` secret and the `LLM_MODEL` (and optionally `LLM_BASE_URL`) repository variables.

## Tests

```sh
pytest            # no network or API key needed
ruff check . && ruff format --check .
```

The agent tests use a scripted fake of the chat-completions client. They check the request
(model, system prompt, tools, reasoning effort), the loop (parallel tool calls, tool errors,
malformed arguments, content filtering, truncation, the turn limit) and that history is only
ever appended to. The MCP tests talk to the server
through a real MCP client over an in-memory transport.

## License

MIT. The data is synthetic, and every farm and customer name is made up.
