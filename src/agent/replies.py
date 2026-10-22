from __future__ import annotations

import discord

from feedback import FeedbackStore, FeedbackView
from src.agent.pipeline import PipelineAnswer

BRAND_COLOR = 0xEA5322
BOT_NAME = "Nero"

_RISK_LABELS = {
    "informational": "🟢 Read-only / informational",
    "state-changing": "🟡 Changes system state",
    "destructive": "🔴 Potentially destructive",
}


def answer_embed(answer: PipelineAnswer) -> discord.Embed:
    embed = discord.Embed(
        title=answer.title,
        description=answer.text,
        color=BRAND_COLOR,
    )
    embed.set_footer(text=f"{BOT_NAME} • Ryoku support")
    return embed


def answer_view(
    answer: PipelineAnswer,
    feedback: FeedbackStore | None,
) -> discord.ui.View:
    view = FeedbackView(feedback) if feedback is not None else discord.ui.View(timeout=None)
    if answer.risk is not None:
        view.add_item(
            discord.ui.Button(
                label=_RISK_LABELS.get(answer.risk, answer.risk),
                style=discord.ButtonStyle.secondary,
                disabled=True,
            )
        )
    if answer.docs_url:
        view.add_item(
            discord.ui.Button(
                label="Open Ryoku documentation →",
                style=discord.ButtonStyle.link,
                url=answer.docs_url,
            )
        )
    return view
