from __future__ import annotations

from dataclasses import dataclass

from prowl import ProwlResult
from src.agent.ollama import LLMResult, OllamaClient
from support import SupportCard

@dataclass(frozen=True)
class RenderedAnswer:
    text: str
    route: str | None = None


def _prompt(query: str, card: SupportCard, sources: ProwlResult | None) -> str:
    parts = [
        f"Question: {query}",
        f"Reviewed support answer: {card.answer}",
        f"Risk: {card.risk}",
    ]
    if card.docs_url:
        parts.append(f"Documentation: {card.docs_url}")
    if sources is not None and sources.status == "ok":
        parts.append("Indexed source locations:")
        parts.extend(f"- {hit.citation}: {hit.snippet}" for hit in sources.hits)
    parts.append(
        "Reply with only one concise Discord-ready final answer. Do not ask follow-up or clarifying questions. "
        "Do not add facts beyond this evidence. If the reviewed answer begins with GUI:, lead with that exact "
        "Ryoku Settings path; mention a command only when the reviewed evidence explicitly says it is needed. "
        "For a contributor question, name whether the command is for a "
        "packaged install or a dev checkout. For state-changing recovery, lead with the safest read-only check "
        "unless the user explicitly asks to perform that action."
    )
    return "\n".join(parts)


def _requires_gui_guidance(card: SupportCard) -> bool:
    return (
        card.answer.startswith("GUI:")
        or "Ryoku Settings" in card.answer
        or "Press `Super" in card.answer
    )


def _includes_gui_guidance(text: str) -> bool:
    normalized = text.lower()
    return "ryoku settings" in normalized or "super +" in normalized or "super+" in normalized


class Answerer:
    def __init__(self, client: OllamaClient, *, rewrite_enabled: bool = True):
        self.client = client
        self.rewrite_enabled = rewrite_enabled

    async def render(
        self,
        query: str,
        card: SupportCard,
        sources: ProwlResult | None = None,
    ) -> RenderedAnswer:
        if card.risk == "destructive":
            return RenderedAnswer(card.answer)
        if not self.rewrite_enabled:
            return RenderedAnswer(card.answer)
        route = "lfm" if sources is not None and sources.status == "ok" else "gemma"
        result: LLMResult = await self.client.answer(_prompt(query, card, sources), route=route)
        if result.status != "ok" or "?" in result.text:
            return RenderedAnswer(card.answer)
        if _requires_gui_guidance(card) and not _includes_gui_guidance(result.text):
            return RenderedAnswer(card.answer)
        return RenderedAnswer(result.text, route=route)
