"""Agent loop tests with a scripted fake chat-completions client (no network, no API key)."""

import json
from types import SimpleNamespace as NS

import pytest

from coffee_agent.agent import CoffeeAgent


def tool_call(id_, name, arguments):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return NS(id=id_, type="function", function=NS(name=name, arguments=raw))


def response(finish_reason, content=None, tool_calls=None, model="test-model", **usage):
    defaults = dict(prompt_tokens=100, completion_tokens=20)
    cached = usage.pop("cached_tokens", 0)
    return NS(
        model=model,
        choices=[
            NS(
                finish_reason=finish_reason,
                message=NS(role="assistant", content=content, tool_calls=tool_calls),
            )
        ],
        usage=NS(**{**defaults, **usage}, prompt_tokens_details=NS(cached_tokens=cached)),
    )


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.script:
            raise AssertionError("agent made more API calls than scripted")
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.chat = NS(completions=FakeCompletions(script))

    @property
    def calls(self):
        return self.chat.completions.calls


def make_agent(toolbox, script, **kwargs):
    client = FakeClient(script)
    kwargs.setdefault("model", "test-model")
    return CoffeeAgent(client, toolbox, **kwargs), client


def test_answers_with_sql_and_records_the_trace(toolbox):
    agent, client = make_agent(
        toolbox,
        [
            response(
                "tool_calls",
                "Let me count.",
                [tool_call("t1", "run_sql", {"query": "SELECT COUNT(*) FROM suppliers"})],
            ),
            response("stop", "There are 12 partner farms.", cached_tokens=900),
        ],
    )
    result = agent.ask("How many suppliers?")

    assert result.answer == "There are 12 partner farms."
    assert result.stop_reason == "stop"
    assert result.turns == 2
    assert result.sql_queries == ["SELECT COUNT(*) FROM suppliers"]
    assert result.usage.input_tokens == 200
    assert result.usage.cached_input_tokens == 900

    # The second request carries the assistant tool call and the matching tool result.
    sent = client.calls[1]["messages"]
    assert sent[-2]["role"] == "assistant"
    assert sent[-2]["tool_calls"][0]["id"] == "t1"
    assert sent[-1]["role"] == "tool"
    assert sent[-1]["tool_call_id"] == "t1"
    assert '"rows": [[12]]' in sent[-1]["content"]


def test_request_shape(toolbox):
    agent, client = make_agent(toolbox, [response("stop", "hi")])
    agent.ask("hello")
    call = client.calls[0]

    assert call["model"] == "test-model"
    assert call["messages"][0]["role"] == "system"
    assert "CREATE TABLE orders" in call["messages"][0]["content"]
    assert call["messages"][1] == {"role": "user", "content": "hello"}
    assert [t["function"]["name"] for t in call["tools"]] == ["run_sql", "search_knowledge"]
    assert all(t["type"] == "function" for t in call["tools"])
    assert "reasoning_effort" not in call
    assert "tool_choice" not in call  # the prompt says when to use each tool


def test_a_model_is_required(toolbox):
    with pytest.raises(ValueError, match="LLM_MODEL"):
        CoffeeAgent(FakeClient([]), toolbox, model="")


def test_system_prompt_is_identical_across_questions(toolbox):
    agent, client = make_agent(toolbox, [response("stop", "a"), response("stop", "b")])
    agent.ask("one")
    agent.ask("two")
    assert client.calls[0]["messages"][0] == client.calls[1]["messages"][0]  # cacheable prefix


def test_parallel_tool_calls_each_get_a_result(toolbox):
    agent, client = make_agent(
        toolbox,
        [
            response(
                "tool_calls",
                None,
                [
                    tool_call("a", "search_knowledge", {"query": "partner average rating"}),
                    tool_call("b", "run_sql", {"query": "SELECT 1"}),
                ],
            ),
            response("stop", "done"),
        ],
    )
    agent.ask("Which suppliers break the rating policy?")
    results = client.calls[1]["messages"][-2:]
    assert [r["tool_call_id"] for r in results] == ["a", "b"]
    assert "marketplace_policies.md" in results[0]["content"]


def test_tool_errors_are_sent_back_flagged(toolbox):
    agent, client = make_agent(
        toolbox,
        [
            response("tool_calls", None, [tool_call("t1", "run_sql", {"query": "DROP TABLE x"})]),
            response("stop", "I can only read data."),
        ],
    )
    result = agent.ask("Delete all orders")
    assert client.calls[1]["messages"][-1]["content"].startswith("Error: ")
    assert result.tool_calls[0].is_error


def test_malformed_arguments_are_reported_to_the_model(toolbox):
    agent, client = make_agent(
        toolbox,
        [
            response("tool_calls", None, [tool_call("t1", "run_sql", '{"query": ')]),
            response("stop", "retrying"),
        ],
    )
    result = agent.ask("q")
    assert "Invalid JSON arguments" in client.calls[1]["messages"][-1]["content"]
    assert result.tool_calls[0].is_error


def test_content_filter_is_reported(toolbox):
    agent, _ = make_agent(toolbox, [response("content_filter")])
    result = agent.ask("q")
    assert result.stop_reason == "content_filter"
    assert "declined" in result.answer


def test_length_is_flagged(toolbox):
    agent, _ = make_agent(toolbox, [response("length", "partial")])
    assert agent.ask("q").answer.endswith("[answer truncated: max_tokens reached]")


def test_max_turns_stops_a_runaway_loop(toolbox):
    loop = [
        response("tool_calls", None, [tool_call(f"t{i}", "run_sql", {"query": "SELECT 1"})])
        for i in range(3)
    ]
    agent, client = make_agent(toolbox, loop, max_turns=3)
    result = agent.ask("q")
    assert result.stop_reason == "max_turns"
    assert len(client.calls) == 3


def test_history_is_append_only(toolbox):
    agent, client = make_agent(toolbox, [response("stop", "first"), response("stop", "second")])
    first = agent.ask("q1")
    snapshot = list(first.messages)
    agent.ask("q2", history=first.messages)
    sent = client.calls[1]["messages"][1:]  # after the system message
    assert sent[: len(snapshot)] == snapshot
    assert sent[-1] == {"role": "user", "content": "q2"}


@pytest.mark.parametrize("effort", ["low", "high"])
def test_reasoning_effort_is_configurable(toolbox, effort):
    agent, client = make_agent(toolbox, [response("stop", "x")], reasoning_effort=effort)
    agent.ask("q")
    assert client.calls[0]["reasoning_effort"] == effort
