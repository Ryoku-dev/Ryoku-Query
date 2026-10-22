from __future__ import annotations

import json
import re
from typing import Any

from src.agent.cloud import CloudClient

_VERDICTS = frozenset({"correct", "partially_correct", "incorrect"})
_GUI_CUES = (
    re.compile(r"ryoku hub", re.IGNORECASE),
    re.compile(r"super\s*\+", re.IGNORECASE),
    re.compile(r"\bpicker\b", re.IGNORECASE),
    re.compile(r"\s>\s"),
)
_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_CODE_BLOCK = re.compile(r"```", re.MULTILINE)
_KEYBIND_PREFIX = re.compile(
    r"^\s*(?:super|ctrl|alt|shift|print|mod)(?:\b|\s*\+)",
    re.IGNORECASE,
)
_KEYBIND_CHORD = re.compile(
    r"^\s*[^+\s`]+\s*\+\s*[^+\s`]+(?:\s*\+\s*[^+\s`]+)*\s*$"
)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    positions = [match.start() for pattern in patterns if (match := pattern.search(text))]
    return min(positions) if positions else None


def _is_keybind(value: str) -> bool:
    return bool(_KEYBIND_PREFIX.match(value) or _KEYBIND_CHORD.fullmatch(value))

def _is_gui_breadcrumb(value: str) -> bool:
    return bool(
        re.search(r"\bryoku\s+(?:hub|settings)\b[^`\n]*>", value, re.IGNORECASE)
        or re.search(r"\s>\s", value)
    )


def _gui_precedes_command(text: str) -> bool:
    inline_matches = list(_INLINE_CODE.finditer(text))
    keybind_positions = [
        match.start() for match in inline_matches if _is_keybind(match.group(1))
    ]
    breadcrumb_positions = [
        match.start()
        for match in inline_matches
        if _is_gui_breadcrumb(match.group(1))
    ]
    command_positions = [
        match.start()
        for match in inline_matches
        if not (
            _is_keybind(match.group(1))
            or _is_gui_breadcrumb(match.group(1))
        )
    ]
    command_positions.extend(match.start() for match in _CODE_BLOCK.finditer(text))
    if not command_positions:
        return True
    gui_position = _first_match(_GUI_CUES, text)
    gui_positions = keybind_positions + breadcrumb_positions + (
        [gui_position] if gui_position is not None else []
    )
    return bool(gui_positions) and min(gui_positions) < min(command_positions)


def deterministic_checks(answer_text: str, expect: dict[str, Any]) -> dict[str, Any]:
    """Run exact, reproducible content and GUI-first checks on one answer."""

    folded = answer_text.casefold()
    groups = expect.get("must_mention_any", [])
    missing_groups = [
        group
        for group in groups
        if not any(str(alternative).casefold() in folded for alternative in group)
    ]
    forbidden_found = [
        phrase
        for phrase in expect.get("must_not_mention", [])
        if str(phrase).casefold() in folded
    ]
    gui_first = _gui_precedes_command(answer_text)
    gui_first_required = expect.get("gui_first") is True
    passed = not missing_groups and not forbidden_found and (
        gui_first or not gui_first_required
    )
    return {
        "pass": passed,
        "must_mention_pass": not missing_groups,
        "missing_groups": missing_groups,
        "must_not_mention_pass": not forbidden_found,
        "forbidden_found": forbidden_found,
        "gui_first": gui_first,
        "gui_first_required": gui_first_required,
        "gui_first_pass": gui_first or not gui_first_required,
    }


def _find_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    fallback: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if fallback is None:
            fallback = value
        if {"verdict", "reason", "gui_first"} <= set(value):
            return value
    return fallback


def _judge_error(reason: str) -> dict[str, Any]:
    return {"verdict": "judge_error", "reason": reason, "gui_first": False}


def _short_reason(value: str) -> str:
    words = value.split()
    return " ".join(words[:25])


async def llm_judge(
    cloud: CloudClient,
    question: str,
    ideal: str,
    answer: str,
) -> dict[str, Any]:
    """Judge an answer against a reference using a strict JSON-only rubric."""

    system = (
        "You are a strict evaluator for Ryoku Linux support answers. Ryoku is "
        "GUI-first: an answer should name the relevant Ryoku Hub page, picker, "
        "or keybind before any shell command. Compare only the submitted answer "
        "with the reference facts. The reference's key facts are its essential "
        "path, keybind, or command. Verdict is correct when the answer contains "
        "those key facts and nothing contradicts the reference. Extra details "
        "that the reference does not mention must not lower the verdict unless "
        "they are clearly wrong or contradict the reference. Verdict is "
        "partially_correct when the answer contains some key facts but misses "
        "others. Verdict is incorrect when the key facts are missing or "
        "contradicted, or when the answer gives up although the reference has "
        "an answer. Return exactly one JSON object with keys verdict, reason, "
        "and gui_first. verdict must be correct, partially_correct, or "
        "incorrect. reason must be at most 25 words. gui_first must be a JSON "
        "boolean. No markdown or extra text."
    )
    user = (
        f"QUESTION:\n{question}\n\nREFERENCE:\n{ideal}\n\n"
        f"ANSWER TO JUDGE:\n{answer}"
    )
    result = await cloud.complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=180,
    )
    if result.status != "ok":
        return _judge_error(result.error or "unavailable")

    value = _find_json_object(result.text)
    if value is None:
        return _judge_error("invalid_json")
    verdict = value.get("verdict")
    reason = value.get("reason")
    gui_first = value.get("gui_first")
    if verdict not in _VERDICTS:
        return _judge_error("invalid_verdict")
    if not isinstance(reason, str) or not reason.strip():
        return _judge_error("invalid_reason")
    if not isinstance(gui_first, bool):
        return _judge_error("invalid_gui_first")
    return {
        "verdict": verdict,
        "reason": _short_reason(reason.strip()),
        "gui_first": gui_first,
    }
