import json
from pathlib import Path

import pytest

from coffee_agent.agent import AgentResult, ToolCall
from coffee_agent.evaluation import Case, grade, load_cases, numbers_in, rows_match

CASES = load_cases(Path(__file__).parents[1] / "evals" / "cases.jsonl")


@pytest.mark.parametrize("case", [c for c in CASES if c.reference_sql], ids=lambda c: c.id)
def test_reference_queries_run_and_return_data(db, case):
    rows = db.run(case.reference_sql).rows
    assert rows and rows[0][0] is not None


def test_case_ids_are_unique():
    assert len({c.id for c in CASES}) == len(CASES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("R$ 31,202.99", 31202.99),  # en-US
        ("R$ 31.202,99", 31202.99),  # pt-BR
        ("about 697 orders.", 697),
    ],
)
def test_numbers_in_understands_both_locales(text, expected):
    assert any(abs(n - expected) < 1e-6 for n in numbers_in(text))


def test_rows_match_is_order_and_column_insensitive():
    reference = [["natural", 84.58], ["washed", 84.15]]
    assert rows_match(reference, [[84.15, "washed", 8], [84.58, "Natural", 21]])
    assert not rows_match(reference, [["natural", 84.58]])
    assert not rows_match(reference, [["natural", 80.0], ["washed", 84.15]])


def sql_call(rows):
    output = json.dumps({"columns": [], "rows": rows, "row_count": len(rows), "truncated": False})
    return ToolCall("run_sql", {"query": "..."}, output, False)


def result(answer, calls=()):
    return AgentResult(answer, "end_turn", "test-model", 2, list(calls))


def test_grade_answer_number(db):
    case = Case("c", "q", ["answer_number"], reference_sql="SELECT COUNT(*) FROM suppliers")
    assert grade(case, result("We have 12 partner farms."), db).passed
    failed = grade(case, result("We have 11 partner farms."), db)
    assert not failed.passed and "12" in failed.failures[0]


def test_grade_sql_match_and_names(db):
    sql = "SELECT process, ROUND(AVG(sca_score), 2) FROM coffees GROUP BY process"
    rows = [list(r) for r in db.run(sql).rows]
    case = Case("c", "q", ["sql_match", "answer_names"], reference_sql=sql)
    names = ", ".join(r[0] for r in rows)
    assert grade(case, result(f"Processes: {names}", [sql_call(rows)]), db).passed
    assert not grade(case, result(names, [sql_call(rows[:-1])]), db).passed


def test_grade_sources_and_answer_any(db):
    case = Case("c", "q", ["answer_any", "sources"], answer_any=["80"], sources=["sca_cupping.md"])
    search = ToolCall("search_knowledge", {"query": "x"}, "[1] sca_cupping.md › ...", False)
    assert grade(case, result("80 points", [search]), db).passed
    assert not grade(case, result("80 points"), db).passed  # answered without looking it up
