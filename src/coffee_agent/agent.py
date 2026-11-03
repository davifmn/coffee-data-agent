"""Tool-use agent that answers questions with SQL and the knowledge base.

It talks to any chat-completions endpoint that follows the OpenAI API: a hosted model or
a local server (vLLM, Ollama, LM Studio...). Three environment variables configure it:
``LLM_BASE_URL`` (empty means the default OpenAI endpoint), ``LLM_API_KEY`` and
``LLM_MODEL``.

The loop is written by hand so that every step is visible and testable: each tool call is
recorded in ``AgentResult.tool_calls``, tool arguments are parsed and validated before they
run, and every ``finish_reason`` is handled explicitly.

The system prompt (instructions + schema) is built once and never changes, and the history
is only ever appended to, so providers that cache prompt prefixes re-read the conversation
from cache on every turn of the tool loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .tools import TOOL_SPECS, Toolbox

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "")

SYSTEM_PROMPT = """\
You are the data analyst of Grão Fino, a Brazilian specialty-coffee marketplace. People \
from the team ask you about the business and about coffee.

You have two tools:
- run_sql queries the marketplace database (schema below). Use it for every figure you \
report about sales, customers, suppliers, coffees or reviews; do not estimate a number \
you can query.
- search_knowledge looks up coffee knowledge and the marketplace's internal policies. \
Cite the sections you rely on as [file.md › Heading].

When a question depends on a business rule (a policy threshold, a commission rate) and on \
data, look the rule up first and then query with it. Money is in Brazilian reais (R$). \
Unless the user says otherwise, revenue means quantity_kg * unit_price_brl over orders \
that were not cancelled; say which definition you used.

Lead with the answer, then the supporting numbers, then the SQL you ran in a ```sql \
block. Reply in the language the question was asked in.

Database schema:
"""


def make_client() -> Any:
    """An OpenAI-compatible client configured from LLM_BASE_URL and LLM_API_KEY."""
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )


@dataclass
class ToolCall:
    name: str
    input: Any
    output: str
    is_error: bool


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached_input_tokens += getattr(details, "cached_tokens", 0) or 0


@dataclass
class AgentResult:
    answer: str
    stop_reason: str
    model: str
    turns: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sql_queries(self) -> list[str]:
        return [c.input.get("query", "") for c in self.tool_calls if c.name == "run_sql"]


class CoffeeAgent:
    def __init__(
        self,
        client: Any,
        toolbox: Toolbox,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = None,
        max_turns: int = 12,
        max_tokens: int = 16000,
    ) -> None:
        if not model:
            raise ValueError("no model configured: set LLM_MODEL or pass model=")
        self.client = client
        self.toolbox = toolbox
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        # Built once and never changed, so the cached prefix stays byte-identical.
        self._system = {"role": "system", "content": SYSTEM_PROMPT + toolbox.describe_schema()}

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._system, *messages],
            "tools": TOOL_SPECS,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self.client.chat.completions.create(**kwargs)

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        """Answer one question. Pass ``result.messages`` back as ``history`` to keep chatting."""
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": question})
        usage = Usage()
        calls: list[ToolCall] = []

        for turn in range(1, self.max_turns + 1):
            response = self._request(messages)
            usage.add(response.usage)
            choice = response.choices[0]
            message = choice.message
            messages.append(_assistant_message(message))
            reason = choice.finish_reason or "stop"

            if message.tool_calls:
                messages.extend(self._run_tools(message.tool_calls, calls))
                continue

            answer = (message.content or "").strip()
            if reason == "content_filter":
                answer = answer or "The request was declined by the provider's content filter."
            elif reason == "length":
                answer += "\n\n[answer truncated: max_tokens reached]"
            return AgentResult(answer, reason, response.model, turn, calls, usage, messages)

        return AgentResult(
            "Stopped after reaching the maximum number of tool-use turns.",
            "max_turns",
            self.model,
            self.max_turns,
            calls,
            usage,
            messages,
        )

    def _run_tools(self, tool_calls: list[Any], calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Run every tool call of a turn; each result goes back as its own ``tool`` message."""
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                content, is_error = f"Invalid JSON arguments for {name}: {exc}", True
            else:
                outcome = self.toolbox.execute(name, args)
                content, is_error = outcome.content, outcome.is_error
            calls.append(ToolCall(name, args, content, is_error))
            if is_error:
                content = f"Error: {content}"
            results.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        return results


def _assistant_message(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return out
