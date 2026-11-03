import json

import pytest

from coffee_agent.retrieval import Chunk, KnowledgeBase, chunk_markdown, tokenize
from coffee_agent.tools import TOOL_SPECS


def test_tokenize_folds_accents_and_plurals():
    assert tokenize("Caparaó cafés naturals") == ["caparao", "cafe", "natural"]
    assert tokenize("the and of") == []


def test_chunking_by_heading():
    chunks = chunk_markdown("x.md", "# Title\nintro\n## One\nfirst\n## Two\nsecond\n")
    assert [(c.heading, c.text) for c in chunks] == [
        ("Title", "intro"),
        ("One", "first"),
        ("Two", "second"),
    ]
    assert chunks[1].citation == "x.md › One"


@pytest.mark.parametrize(
    ("query", "source", "heading"),
    [
        ("what is pulped natural", "processing.md", "Pulped natural (cereja descascado)"),
        ("minimum score to be specialty coffee", "sca_cupping.md", None),
        ("V60 pour over ratio", "brewing.md", "Pour-over (V60, Kalita, Melitta)"),
        ("free shipping threshold", "marketplace_policies.md", "Shipping"),
        ("partner average rating requirement", "marketplace_policies.md", None),
        ("Geisha floral jasmine", "varieties.md", "Geisha (Gesha)"),
        ("Cerrado denomination of origin", "brazilian_regions.md", "Cerrado Mineiro (MG)"),
    ],
)
def test_search_finds_the_right_section(kb, query, source, heading):
    top = kb.search(query, k=1)[0].chunk
    assert top.source == source
    if heading:
        assert top.heading == heading


def test_search_with_no_known_terms(kb):
    assert kb.search("zzzz qqqq") == []


def test_empty_kb_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([])


def test_custom_corpus(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## Espresso\nnine bar pressure\n")
    kb = KnowledgeBase.from_directory(tmp_path)
    assert kb.search("pressure")[0].chunk == Chunk("a.md", "Espresso", "nine bar pressure")


def test_tool_schemas_are_function_tools():
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["additionalProperties"] is False
        # Numeric/length constraints are not portable across providers; they are
        # enforced client-side with pydantic instead.
        for prop in params["properties"].values():
            assert not {"minimum", "maximum", "minLength", "maxLength"} & prop.keys()


def test_run_sql_tool_returns_json(toolbox):
    outcome = toolbox.execute("run_sql", {"query": "SELECT COUNT(*) AS n FROM suppliers"})
    assert not outcome.is_error
    assert json.loads(outcome.content) == {
        "columns": ["n"],
        "rows": [[12]],
        "row_count": 1,
        "truncated": False,
    }


def test_run_sql_tool_reports_truncation(toolbox):
    outcome = toolbox.execute("run_sql", {"query": "SELECT id FROM orders", "max_rows": 3})
    payload = json.loads(outcome.content)
    assert payload["truncated"] is True and "aggregate" in payload["note"]


@pytest.mark.parametrize(
    ("name", "tool_input", "message"),
    [
        ("run_sql", {"query": "DELETE FROM orders"}, "SQL error"),
        ("run_sql", {"query": ""}, "Invalid input"),
        ("run_sql", {"query": "SELECT 1", "max_rows": 10_000}, "Invalid input"),
        ("run_sql", {"sql": "SELECT 1"}, "Invalid input"),
        ("search_knowledge", {"query": "x", "k": 50}, "Invalid input"),
        ("drop_everything", {}, "Unknown tool"),
    ],
)
def test_bad_tool_calls_become_error_results(toolbox, name, tool_input, message):
    outcome = toolbox.execute(name, tool_input)
    assert outcome.is_error
    assert message in outcome.content


def test_search_tool_includes_citations(toolbox):
    outcome = toolbox.execute("search_knowledge", {"query": "anaerobic fermentation", "k": 2})
    assert "[1] processing.md › Anaerobic and experimental fermentation" in outcome.content
