"""Run Nero support-answer evaluations.

Each input file is a JSON list of cases (or an object with a ``cases`` list).
A case has this schema::

    {
      "id": str,
      "question": str,
      "history": [{"role": "user" | "nero", "text": str}],
      "expect": {
        "must_mention_any": [[str, ...], ...],
        "must_not_mention": [str],
        "gui_first": bool,
        "ideal": str
      },
      "source": "help-thread" | "general" | "nero-exchange" | "synthetic",
      "verdict_then": str | null,
      "lang": "en" | "pt" | "es"
    }

Every inner ``must_mention_any`` list is a group of alternatives. Every group
must have at least one case-insensitive substring match in the answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from model2vec import StaticModel

from eval.judge import deterministic_checks, llm_judge
from src.agent.cloud import CloudClient
from src.agent.pipeline import SupportPipeline, Turn
from src.knowledge.index import KnowledgeIndex
from support import SupportRetriever, load_support_cards

_MODEL_NAME = "minishlab/potion-base-32M"
_NO_MATCH = (
    "I couldn't find a reviewed answer for that question. Try naming the Ryoku "
    "component, command, or error you are seeing."
)
_SOURCES = frozenset({"help-thread", "general", "nero-exchange", "synthetic"})
_LANGUAGES = frozenset({"en", "pt", "es"})
_ROLES = frozenset({"user", "nero"})
_CASE_KEYS = frozenset(
    {"id", "question", "history", "expect", "source", "verdict_then", "lang"}
)
_EXPECT_KEYS = frozenset(
    {"must_mention_any", "must_not_mention", "gui_first", "ideal"}
)


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    history: tuple[Turn, ...]
    expect: dict[str, Any]
    source: str
    verdict_then: str | None
    lang: str


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _strict_keys(value: dict[str, Any], allowed: frozenset[str], context: str) -> None:
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _string_list(value: Any, context: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{context} must be a list of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"{context} must not be empty")
    return [item.strip() for item in value]


def _parse_case(row: Any, context: str) -> EvalCase:
    if not isinstance(row, dict):
        raise ValueError(f"{context} must be an object")
    _strict_keys(row, _CASE_KEYS, context)
    case_id = _nonempty_text(row["id"], f"{context}.id")
    question = _nonempty_text(row["question"], f"case {case_id}.question")

    history_raw = row["history"]
    if not isinstance(history_raw, list):
        raise ValueError(f"case {case_id}.history must be a list")
    history: list[Turn] = []
    for index, item in enumerate(history_raw):
        if not isinstance(item, dict) or set(item) != {"role", "text"}:
            raise ValueError(
                f"case {case_id}.history[{index}] must contain role and text"
            )
        if item["role"] not in _ROLES:
            raise ValueError(f"case {case_id}.history[{index}] has invalid role")
        history.append(
            Turn(
                role=item["role"],
                text=_nonempty_text(
                    item["text"], f"case {case_id}.history[{index}].text"
                ),
            )
        )

    expect = row["expect"]
    if not isinstance(expect, dict):
        raise ValueError(f"case {case_id}.expect must be an object")
    _strict_keys(expect, _EXPECT_KEYS, f"case {case_id}.expect")
    groups_raw = expect["must_mention_any"]
    if not isinstance(groups_raw, list):
        raise ValueError(f"case {case_id}.expect.must_mention_any must be a list")
    groups = [
        _string_list(group, f"case {case_id}.expect.must_mention_any[{index}]", nonempty=True)
        for index, group in enumerate(groups_raw)
    ]
    forbidden = _string_list(
        expect["must_not_mention"], f"case {case_id}.expect.must_not_mention"
    )
    if not isinstance(expect["gui_first"], bool):
        raise ValueError(f"case {case_id}.expect.gui_first must be a boolean")
    ideal = _nonempty_text(expect["ideal"], f"case {case_id}.expect.ideal")

    source = row["source"]
    if source not in _SOURCES:
        raise ValueError(f"case {case_id}.source has invalid value: {source}")
    verdict_then = row["verdict_then"]
    if verdict_then is not None and not isinstance(verdict_then, str):
        raise ValueError(f"case {case_id}.verdict_then must be a string or null")
    lang = row["lang"]
    if lang not in _LANGUAGES:
        raise ValueError(f"case {case_id}.lang has invalid value: {lang}")

    return EvalCase(
        id=case_id,
        question=question,
        history=tuple(history),
        expect={
            "must_mention_any": groups,
            "must_not_mention": forbidden,
            "gui_first": expect["gui_first"],
            "ideal": ideal,
        },
        source=source,
        verdict_then=verdict_then,
        lang=lang,
    )


def load_cases(paths: Sequence[Path]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("cases") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list or a cases list")
        for index, row in enumerate(rows):
            case = _parse_case(row, f"{path}: case #{index}")
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("no evaluation cases were loaded")
    return cases


def _baseline_answer(
    retriever: SupportRetriever, question: str
) -> tuple[str, str, str | None]:
    decision = retriever.retrieve(question)
    if decision.kind == "answer" and decision.card is not None:
        return decision.card.answer, "card", decision.card.id
    if decision.kind == "clarify" and decision.alternatives:
        card = decision.alternatives[0]
        return card.answer, "clarify", card.id
    return _NO_MATCH, "no_match", None


def _case_json(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "history": [
            {"role": turn.role, "text": turn.text} for turn in case.history
        ],
        "expect": case.expect,
        "source": case.source,
        "verdict_then": case.verdict_then,
        "lang": case.lang,
    }


async def _evaluate_case(
    case: EvalCase,
    semaphore: asyncio.Semaphore,
    *,
    mode: str,
    pipeline: SupportPipeline | None,
    retriever: SupportRetriever | None,
    judge_client: CloudClient | None,
) -> dict[str, Any]:
    async with semaphore:
        if mode == "new":
            if pipeline is None:
                raise RuntimeError("new mode requires a pipeline")
            answer = await pipeline.answer(case.question, case.history)
            text = answer.text
            route = answer.route
            latency_ms = answer.latency_ms
            evidence_ids = list(answer.evidence_ids)
            card_id = answer.card_id
            note = answer.note
        else:
            if retriever is None:
                raise RuntimeError("baseline mode requires a retriever")
            started = time.perf_counter()
            text, route, card_id = _baseline_answer(retriever, case.question)
            latency_ms = round((time.perf_counter() - started) * 1000)
            evidence_ids = []
            note = ""

        deterministic = deterministic_checks(text, case.expect)
        if judge_client is None:
            judge = {
                "verdict": "not_run",
                "reason": "disabled",
                "gui_first": deterministic["gui_first"],
            }
        else:
            judge = await llm_judge(
                judge_client,
                case.question,
                case.expect["ideal"],
                text,
            )
        return {
            **_case_json(case),
            "answer": {
                "text": text,
                "route": route,
                "latency_ms": latency_ms,
                "evidence_ids": evidence_ids,
                "card_id": card_id,
                "note": note,
            },
            "deterministic": deterministic,
            "judge": judge,
        }


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _rate(passed: int, total: int) -> float:
    return passed / total if total else 1.0


def _summarize(mode: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    deterministic_passed = sum(row["deterministic"]["pass"] for row in results)
    gui_rows = [row for row in results if row["expect"]["gui_first"]]
    gui_passed = sum(row["deterministic"]["gui_first"] for row in gui_rows)
    verdicts = Counter(row["judge"]["verdict"] for row in results)
    routes = Counter(row["answer"]["route"] for row in results)
    latencies = [int(row["answer"]["latency_ms"]) for row in results]
    notes = Counter(
        row["answer"]["note"] for row in results if row["answer"]["note"]
    )
    judged = any(row["judge"]["verdict"] != "not_run" for row in results)
    failing = [
        row["id"]
        for row in results
        if not row["deterministic"]["pass"]
        or (judged and row["judge"]["verdict"] != "correct")
    ]
    return {
        "mode": mode,
        "case_count": len(results),
        "deterministic": {
            "passed": deterministic_passed,
            "rate": _rate(deterministic_passed, len(results)),
        },
        "judge_verdict_counts": dict(sorted(verdicts.items())),
        "gui_first": {
            "passed": gui_passed,
            "required_cases": len(gui_rows),
            "rate": _rate(gui_passed, len(gui_rows)),
        },
        "route_counts": dict(sorted(routes.items())),
        "note_counts": dict(sorted(notes.items())),
        "latency_ms": {
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
            "max": max(latencies),
        },
        "failing_case_ids": failing,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    det = summary["deterministic"]
    gui = summary["gui_first"]
    verdicts = summary["judge_verdict_counts"]
    latency = summary["latency_ms"]
    rows = [
        ("Cases", str(summary["case_count"])),
        ("Deterministic pass rate", f"{det['passed']}/{summary['case_count']} ({det['rate']:.1%})"),
        (
            "Judge verdicts",
            "correct={correct} partially_correct={partially} incorrect={incorrect}".format(
                correct=verdicts.get("correct", 0),
                partially=verdicts.get("partially_correct", 0),
                incorrect=verdicts.get("incorrect", 0),
            ),
        ),
        ("GUI-first pass rate", f"{gui['passed']}/{gui['required_cases']} ({gui['rate']:.1%})"),
        (
            "Routes",
            ", ".join(f"{key}={value}" for key, value in summary["route_counts"].items()),
        ),
        (
            "Notes",
            ", ".join(
                f"{key}={value}" for key, value in summary["note_counts"].items()
            )
            or "none",
        ),
        (
            "Latency (ms)",
            f"p50={latency['p50']} p95={latency['p95']} max={latency['max']}",
        ),
        (
            "Failing case ids",
            ", ".join(summary["failing_case_ids"]) or "none",
        ),
    ]
    width = max(len(label) for label, _ in rows)
    print(f"{'Metric':<{width}}  Value")
    print(f"{'-' * width}  {'-' * 5}")
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def _parser() -> argparse.ArgumentParser:
    forwarded_socket = (
        "Reach the bridge on ally with: ssh -o ControlPath=/tmp/ally-%C -N "
        "-L /tmp/nero-llm.sock:/run/nero-llm/llm.sock ally\n"
        "Then pass --socket /tmp/nero-llm.sock."
    )
    parser = argparse.ArgumentParser(
        description="Evaluate Nero's new answer pipeline or today's card baseline.",
        epilog=forwarded_socket,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("new", "baseline"), required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--socket",
        default=os.getenv("NERO_LLM_SOCKET", "/run/nero-llm/llm.sock"),
        help="LLM bridge Unix socket (default: %(default)s). " + forwarded_socket.replace("%", "%%"),
    )
    parser.add_argument("--judge-model", default="gpt-6.1-sol")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("eval/out"))
    parser.add_argument("--support", type=Path, default=Path("data/support.json"))
    parser.add_argument("--gui-map", type=Path, default=Path("data/gui_map.json"))
    parser.add_argument("--docs", type=Path, default=Path("data/docs_chunks.json"))
    parser.add_argument("--concurrency", type=int, default=3)
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    cases = load_cases(args.cases)
    encoder = StaticModel.from_pretrained(_MODEL_NAME)

    pipeline: SupportPipeline | None = None
    retriever: SupportRetriever | None = None
    if args.mode == "new":
        index = KnowledgeIndex.from_paths(
            args.support,
            args.gui_map,
            args.docs,
            encoder,
        )
        pipeline = SupportPipeline(
            index,
            CloudClient(args.socket, args.model),
            prowl=None,
        )
    else:
        retriever = SupportRetriever(load_support_cards(args.support), encoder)

    judge_client = None
    if not args.no_judge:
        judge_client = CloudClient(args.socket, args.judge_model)
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(
            _evaluate_case(
                case,
                semaphore,
                mode=args.mode,
                pipeline=pipeline,
                retriever=retriever,
                judge_client=judge_client,
            )
            for case in cases
        )
    )

    summary = _summarize(args.mode, results)
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary)
    return 0


def main() -> int:
    args = _parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
