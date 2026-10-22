from __future__ import annotations

import re
import time
from dataclasses import dataclass

from prowl import ProwlClient, ProwlResult, is_source_query
from src.agent.cloud import CloudClient
from src.agent.prompt import build_messages
from src.knowledge.index import Evidence, KnowledgeIndex
from support import RetrievalDecision, SupportCard, normalize_text, requires_safety_confirmation


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class PipelineAnswer:
    text: str
    route: str
    title: str
    evidence_ids: tuple[str, ...]
    card_id: str | None
    docs_url: str | None
    risk: str | None
    model: str | None
    latency_ms: int


_SAFETY_TEXT = (
    "I won't recommend a destructive action from a broad source match. Describe the "
    "symptom, whether the desktop or a TTY still works, and what recovery steps "
    "you already tried."
)
_FORBIDDEN_OUTPUT = ("rm -rf", "mkfs", "dd if=", "--force", "reset --hard", "pacman -rdd")
_COMMANDS = frozenset(
    {"ryoku", "ryoku-shell", "ryoku-hub", "ryogami", "ryoku-rashin", "ryostore"}
)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_HUB_PATH = re.compile(r"ryoku\s+hub\s*>\s*[^\n`]+", re.IGNORECASE)


def normalize_path(path: str) -> str:
    parts = []
    for part in path.split(">"):
        cleaned = normalize_text(part).strip(" \t.,:;!?()[]{}'\"")
        if cleaned:
            parts.append(cleaned)
    return " > ".join(parts)


def _top_card(decision: RetrievalDecision) -> SupportCard | None:
    if decision.card is not None:
        return decision.card
    if decision.alternatives:
        return decision.alternatives[0]
    if decision.ranked:
        return decision.ranked[0].card
    return None


def _destructive_card(decision: RetrievalDecision) -> SupportCard | None:
    candidate = decision.card
    if candidate is None and decision.alternatives:
        candidate = decision.alternatives[0]
    return candidate if candidate is not None and candidate.risk == "destructive" else None


def _search_query(query: str, history: tuple[Turn, ...]) -> str:
    if len(query.split()) > 6:
        return query
    previous = next((turn.text for turn in reversed(history) if turn.role == "user"), "")
    return f"{previous} {query}".strip() if previous else query


def _source_evidence(result: ProwlResult) -> list[Evidence]:
    return [
        Evidence(
            kind="doc",
            id=f"source:{hit.citation}",
            title=hit.citation,
            text=hit.snippet,
            path=hit.citation,
            url=None,
            risk=None,
            score=2.0 - position / 100.0,
        )
        for position, hit in enumerate(result.hits)
    ]


def _deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    unique: list[Evidence] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    return unique


def _evidence_text(evidence: list[Evidence]) -> str:
    values: list[str] = []
    for item in evidence:
        values.extend((item.title, item.text, item.path or "", item.url or ""))
    return normalize_text("\n".join(values))


def _path_is_known(candidate: str, known_paths: frozenset[str], evidence_text: str) -> bool:
    normalized = normalize_path(candidate)
    if not normalized:
        return False
    for path in known_paths:
        known = normalize_path(path)
        if normalized == known or normalized.startswith(f"{known} >") or normalized.startswith(
            f"{known} "
        ):
            return True
    return normalized in normalize_path(evidence_text)


def _valid_model_text(
    text: str,
    evidence: list[Evidence],
    known_paths: frozenset[str],
) -> bool:
    if not text or len(text) > 1900:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in _FORBIDDEN_OUTPUT):
        return False

    packed_text = _evidence_text(evidence)
    for match in _HUB_PATH.finditer(text):
        if not _path_is_known(match.group(0), known_paths, packed_text):
            return False

    for inline in _INLINE_CODE.findall(text):
        words = inline.strip().split()
        if not words or words[0].lower() not in _COMMANDS:
            continue
        if len(words) < 2:
            return False
        first_two = normalize_text(" ".join(words[:2]))
        if first_two not in packed_text:
            return False
    return True


def _card_for_evidence(index: KnowledgeIndex, item: Evidence) -> SupportCard | None:
    card = index.card(item.id)
    if card is None and item.id.startswith("card:"):
        card = index.card(item.id.removeprefix("card:"))
    return card


def _answer_metadata(
    index: KnowledgeIndex, evidence: list[Evidence]
) -> tuple[str, str | None, str | None]:
    for item in evidence[:3]:
        if item.kind != "card":
            continue
        card = _card_for_evidence(index, item)
        return item.title, card.id if card is not None else item.id.removeprefix("card:"), (
            card.risk if card is not None else item.risk
        )
    if evidence:
        title = " ".join(evidence[0].title.split())[:80].rstrip()
        return title or "Ryoku help", None, None
    return "Ryoku help", None, None


def _first_docs_url(evidence: list[Evidence]) -> str | None:
    return next((item.url for item in evidence if item.url), None)


def _related_text(evidence: list[Evidence]) -> str:
    lines = ["I'm not sure I have a confident Ryoku answer for that."]
    related: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        locator = item.path or item.url or ""
        key = (item.title, locator)
        if key in seen:
            continue
        seen.add(key)
        related.append(f"- {item.title}" + (f" — {locator}" if locator else ""))
        if len(related) == 3:
            break
    if related:
        lines.append("Related:\n" + "\n".join(related))
    lines.append("Open a #help post with the output of `ryoku doctor --check`.")
    return "\n".join(lines)


class SupportPipeline:
    def __init__(
        self,
        index: KnowledgeIndex,
        cloud: CloudClient | None,
        prowl: ProwlClient | None,
        *,
        max_evidence_chars: int = 6000,
    ):
        self.index = index
        self.cloud = cloud
        self.prowl = prowl
        self.max_evidence_chars = max_evidence_chars

    async def answer(
        self, query: str, history: tuple[Turn, ...] = ()
    ) -> PipelineAnswer:
        started = time.monotonic()

        def finish(
            text: str,
            route: str,
            title: str,
            evidence: list[Evidence] | None = None,
            card: SupportCard | None = None,
            *,
            model: str | None = None,
            docs_url: str | None = None,
            risk: str | None = None,
            card_id: str | None = None,
        ) -> PipelineAnswer:
            items = evidence or []
            return PipelineAnswer(
                text=text,
                route=route,
                title=title,
                evidence_ids=tuple(item.id for item in items),
                card_id=card.id if card is not None else card_id,
                docs_url=card.docs_url if card is not None else docs_url,
                risk=card.risk if card is not None else risk,
                model=model,
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            )

        if requires_safety_confirmation(query):
            return finish(_SAFETY_TEXT, "safety", "I need more context")

        decision = self.index.card_decision(query)
        search_query = _search_query(query, history)
        indexed_evidence = self.index.search(search_query)
        evidence = indexed_evidence
        mode = "user"
        success_route = "cloud"

        if self.prowl is not None and is_source_query(query):
            card = _top_card(decision)
            source_hints = card.source_hints if card is not None else ()
            source_result = await self.prowl.search(query, source_hints)
            if source_result.status == "ok":
                evidence = _deduplicate_evidence(
                    _source_evidence(source_result) + indexed_evidence
                )
                mode = "source"
                success_route = "source"

        destructive = _destructive_card(decision)
        if destructive is not None:
            return finish(
                destructive.answer,
                "card",
                destructive.title,
                evidence,
                destructive,
            )

        used_model: str | None = None
        if self.cloud is not None:
            used_model = self.cloud.model
            messages = build_messages(
                query,
                evidence,
                history,
                mode=mode,
                max_evidence_chars=self.max_evidence_chars,
            )
            result = await self.cloud.complete(messages, max_tokens=450)
            if result.status == "ok":
                text = result.text.strip()
                if text == "NO_ANSWER":
                    return finish(
                        _related_text(evidence),
                        "no_match",
                        "No confident Ryoku answer",
                        evidence,
                        model=used_model,
                        docs_url=_first_docs_url(evidence),
                    )
                if text.startswith("CLARIFY:"):
                    question = text.removeprefix("CLARIFY:").strip()
                    if question:
                        return finish(
                            question,
                            "clarify",
                            "A quick question",
                            evidence,
                            model=used_model,
                            docs_url=_first_docs_url(evidence),
                        )
                elif _valid_model_text(text, evidence, self.index.known_paths()):
                    title, card_id, risk = _answer_metadata(self.index, evidence)
                    return finish(
                        text,
                        success_route,
                        title,
                        evidence,
                        model=used_model,
                        docs_url=_first_docs_url(evidence),
                        risk=risk,
                        card_id=card_id,
                    )

        if decision.kind == "answer" and decision.card is not None:
            return finish(
                decision.card.answer,
                "card",
                decision.card.title,
                evidence,
                decision.card,
                model=used_model,
            )
        if decision.kind == "clarify" and decision.alternatives:
            card = decision.alternatives[0]
            return finish(
                card.answer,
                "card",
                card.title,
                evidence,
                card,
                model=used_model,
            )
        return finish(
            _related_text(evidence),
            "no_match",
            "No confident Ryoku answer",
            evidence,
            model=used_model,
            docs_url=_first_docs_url(evidence),
        )
