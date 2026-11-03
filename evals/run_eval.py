"""Run the agent on every case in cases.jsonl and grade it.

This calls the configured model (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL) and may cost
money (roughly 15 questions x a few turns). Pass prices to get a cost estimate.

    python evals/run_eval.py                    # all cases
    python evals/run_eval.py --only revenue_2025 weak_partners --effort high
    python evals/run_eval.py --price-input 3 --price-output 15 --price-cached 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from coffee_agent.agent import DEFAULT_MODEL, CoffeeAgent, make_client
from coffee_agent.config import open_toolbox
from coffee_agent.evaluation import grade, load_cases

HERE = Path(__file__).parent


def cost_usd(usage, prices: dict[str, float]) -> float:
    """Cost in USD from prices per million tokens; cached input is billed at its own rate."""
    uncached = usage.input_tokens - usage.cached_input_tokens
    return (
        uncached * prices["input"]
        + usage.cached_input_tokens * prices["cached"]
        + usage.output_tokens * prices["output"]
    ) / 1_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--price-input", type=float, default=0.0, help="USD per 1M input tokens")
    parser.add_argument("--price-output", type=float, default=0.0, help="USD per 1M output tokens")
    parser.add_argument("--price-cached", type=float, default=0.0, help="USD per 1M cached tokens")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    args = parser.parse_args()

    toolbox = open_toolbox()
    prices = {"input": args.price_input, "output": args.price_output, "cached": args.price_cached}
    agent = CoffeeAgent(make_client(), toolbox, model=args.model, reasoning_effort=args.effort)
    cases = [c for c in load_cases(HERE / "cases.jsonl") if not args.only or c.id in args.only]

    records, passed, total_cost = [], 0, 0.0
    for case in cases:
        started = time.perf_counter()
        result = agent.ask(case.question)
        seconds = time.perf_counter() - started
        g = grade(case, result, toolbox.db)
        cost = cost_usd(result.usage, prices)
        total_cost += cost
        passed += g.passed
        status = "PASS" if g.passed else "FAIL"
        print(f"{status}  {case.id:<20} {result.turns} turns  {seconds:5.1f}s  ${cost:.4f}")
        for failure in g.failures:
            print(f"      - {failure}")
        records.append(
            {
                "id": case.id,
                "passed": g.passed,
                "failures": g.failures,
                "answer": result.answer,
                "sql": result.sql_queries,
                "turns": result.turns,
                "seconds": round(seconds, 2),
                "usage": vars(result.usage),
                "cost_usd": round(cost, 5),
                "model": result.model,
            }
        )

    print(f"\n{passed}/{len(cases)} passed · total ${total_cost:.3f}")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_model = args.model.replace("/", "_")
    out = out_dir / f"{stamp}-{safe_model}-{args.effort or 'default'}.json"
    out.write_text(json.dumps({"passed": passed, "total": len(cases), "cases": records}, indent=2))
    print(f"details: {out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
