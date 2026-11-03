"""Grading for the eval set in ``evals/cases.jsonl``.

Every case is a question plus one or more checks. Expected values are never typed by
hand: each case carries a reference SQL query, and the expected answer is whatever that
query returns on the current database. Rebuilding the data with another seed therefore
keeps the eval valid.

Checks:

``sql_match``      one of the agent's ``run_sql`` results contains the reference rows
                   (same row count; each reference row's values found in an agent row,
                   numbers compared with a 0.5% tolerance). This is the "execution
                   accuracy" metric used for text-to-SQL benchmarks.
``answer_number``  the reference scalar appears in the final answer (accepts 1,234.5,
                   1.234,5 and 1234.50 styles).
``answer_names``   every value in the first column of the reference result is mentioned
                   in the answer (case- and accent-insensitive).
``answer_any``     at least one of the given strings appears in the answer.
``sources``        each listed knowledge file was returned by ``search_knowledge``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import AgentResult
from .db import ReadOnlyDatabase

REL_TOL = 0.005


@dataclass
class Case:
    id: str
    question: str
    checks: list[str]
    reference_sql: str | None = None
    answer_any: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class Grade:
    case_id: str
    passed: bool
    failures: list[str]


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    return cases


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= max(0.01, REL_TOL * max(abs(a), abs(b)))
    return _fold(a).strip() == _fold(b).strip()


def _row_contains(candidate: list[Any], reference: list[Any]) -> bool:
    remaining = list(candidate)
    for value in reference:
        match = next((i for i, v in enumerate(remaining) if _close(v, value)), None)
        if match is None:
            return False
        remaining.pop(match)
    return True


def rows_match(reference: list[list[Any]], candidate: list[list[Any]]) -> bool:
    """Order-insensitive: every reference row must pair with a distinct candidate row."""
    if len(reference) != len(candidate):
        return False
    unused = list(candidate)
    for ref_row in reference:
        hit = next((i for i, row in enumerate(unused) if _row_contains(row, ref_row)), None)
        if hit is None:
            return False
        unused.pop(hit)
    return True


_NUMBER = re.compile(r"-?\d[\d.,]*")


def numbers_in(text: str) -> list[float]:
    """Every number in ``text`` under both 1,234.5 (en) and 1.234,5 (pt-BR) conventions."""
    found = []
    for token in _NUMBER.findall(text):
        token = token.rstrip(".,")
        for thousands, decimal in ((",", "."), (".", ",")):
            candidate = token.replace(thousands, "").replace(decimal, ".")
            try:
                found.append(float(candidate))
            except ValueError:
                continue
    return found


def grade(case: Case, result: AgentResult, db: ReadOnlyDatabase) -> Grade:
    failures: list[str] = []
    reference = db.run(case.reference_sql, max_rows=500).rows if case.reference_sql else []
    reference = [list(row) for row in reference]
    answer = result.answer

    for check in case.checks:
        if check == "sql_match":
            outputs = [
                json.loads(c.output)["rows"]
                for c in result.tool_calls
                if c.name == "run_sql" and not c.is_error
            ]
            if not any(rows_match(reference, rows) for rows in outputs):
                failures.append("no run_sql result matched the reference query")
        elif check == "answer_number":
            expected = reference[0][0]
            if not any(_close(n, expected) for n in numbers_in(answer)):
                failures.append(f"answer does not contain {expected}")
        elif check == "answer_names":
            missing = [row[0] for row in reference if _fold(row[0]) not in _fold(answer)]
            if missing:
                failures.append(f"answer does not mention {missing}")
        elif check == "answer_any":
            if not any(_fold(s) in _fold(answer) for s in case.answer_any):
                failures.append(f"answer contains none of {case.answer_any}")
        elif check == "sources":
            searched = " ".join(c.output for c in result.tool_calls if c.name == "search_knowledge")
            missing = [s for s in case.sources if s not in searched]
            if missing:
                failures.append(f"knowledge base not consulted for {missing}")
        else:
            raise ValueError(f"unknown check {check!r} in case {case.id}")
    return Grade(case.id, not failures, failures)
