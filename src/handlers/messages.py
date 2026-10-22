from __future__ import annotations

import re
from pathlib import Path

import discord

from feedback import FeedbackStore
from src.agent.pipeline import SupportPipeline, Turn
from src.agent.replies import answer_embed, answer_view

_NON_REQUESTS = frozenset(
    {
        "thanks",
        "thank you",
        "ty",
        "thx",
        "ok",
        "okay",
        "nice",
        "lol",
        "gm",
        "hi",
        "hello",
        "obrigado",
        "obrigada",
        "gracias",
        "valeu",
    }
)


def _is_help_forum_thread(channel, help_forum_id: int | None) -> bool:
    return (
        help_forum_id is not None
        and help_forum_id > 0
        and getattr(channel, "parent_id", None) == help_forum_id
    )


def _is_reply_to_bot(message, bot_user_id: int) -> bool:
    resolved = getattr(getattr(message, "reference", None), "resolved", None)
    return (
        resolved is not None
        and getattr(getattr(resolved, "author", None), "id", None) == bot_user_id
    )


def _is_request(query: str) -> bool:
    normalized = re.sub(r"[^\w]+", " ", query.casefold()).strip()
    if normalized in _NON_REQUESTS:
        return False
    return "?" in query or len(re.findall(r"\b[\w'-]+\b", query)) >= 3


def extract_query(
    message,
    bot_user_id: int,
    support_channel_id: int | None = None,
    help_forum_id: int | None = None,
) -> str | None:
    if message.author.bot:
        return None

    channel = getattr(message, "channel", None)
    in_support_channel = (
        support_channel_id is not None
        and support_channel_id > 0
        and getattr(channel, "id", None) == support_channel_id
    )
    mentioned = any(
        getattr(user, "id", None) == bot_user_id
        for user in getattr(message, "mentions", ())
    )
    replied = _is_reply_to_bot(message, bot_user_id)
    forum_starter = (
        _is_help_forum_thread(channel, help_forum_id)
        and getattr(message, "id", None) == getattr(channel, "id", None)
    )
    if not in_support_channel and not mentioned and not replied and not forum_starter:
        return None

    content = re.sub(
        fr"<@!?{bot_user_id}>",
        "",
        getattr(message, "content", ""),
    ).strip()
    if forum_starter:
        thread_name = str(getattr(channel, "name", "")).strip()
        query = f"{thread_name}\n{content}".strip()
    else:
        query = content
    if not query:
        return None
    if in_support_channel and not mentioned and not replied and not _is_request(query):
        return None
    return query


def _turn_from_message(message, bot_user_id: int) -> Turn | None:
    author_id = getattr(getattr(message, "author", None), "id", None)
    if author_id == bot_user_id:
        embeds = getattr(message, "embeds", ())
        description = (
            getattr(embeds[0], "description", None)
            if embeds
            else None
        )
        text = description or getattr(message, "content", "")
        role = "nero"
    else:
        text = getattr(message, "content", "")
        role = "user"
    text = str(text or "").strip()
    return Turn(role, text) if text else None


async def _resolve_reference(message):
    reference = getattr(message, "reference", None)
    if reference is None:
        return None
    resolved = getattr(reference, "resolved", None)
    if getattr(resolved, "author", None) is not None:
        return resolved
    message_id = getattr(reference, "message_id", None)
    if not isinstance(message_id, int):
        return None
    try:
        return await message.channel.fetch_message(message_id)
    except Exception:
        return None


async def _reply_history(
    message,
    bot_user_id: int,
    help_forum_id: int | None,
) -> tuple[Turn, ...]:
    referenced = []
    current = message
    seen_ids: set[int] = set()
    for _ in range(3):
        current = await _resolve_reference(current)
        if current is None:
            break
        current_id = getattr(current, "id", None)
        if isinstance(current_id, int):
            if current_id in seen_ids:
                break
            seen_ids.add(current_id)
        referenced.append(current)

    channel = getattr(message, "channel", None)
    is_nonstarter = (
        _is_help_forum_thread(channel, help_forum_id)
        and getattr(message, "id", None) != getattr(channel, "id", None)
    )
    starter_id = getattr(channel, "id", None)
    turns = []
    for prior in reversed(referenced):
        if is_nonstarter and getattr(prior, "id", None) == starter_id:
            continue
        turn = _turn_from_message(prior, bot_user_id)
        if turn is not None:
            turns.append(turn)

    if is_nonstarter:
        try:
            starter = await channel.fetch_message(starter_id)
        except Exception:
            starter = None
        if starter is not None:
            name = str(getattr(channel, "name", "")).strip()
            content = str(getattr(starter, "content", "") or "").strip()
            text = f"{name}\n{content}".strip()
            if text:
                turns.insert(0, Turn("user", text))
    return tuple(turns)

async def _unresolved_reply_query(message, bot_user_id: int) -> str | None:
    if getattr(getattr(message, "author", None), "bot", False):
        return None
    referenced = await _resolve_reference(message)
    if (
        referenced is None
        or getattr(getattr(referenced, "author", None), "id", None) != bot_user_id
    ):
        return None
    content = re.sub(
        fr"<@!?{bot_user_id}>",
        "",
        getattr(message, "content", ""),
    ).strip()
    return content or None


async def handle_message(
    message,
    bot_user_id: int,
    pipeline: SupportPipeline,
    logo_path: Path,
    *,
    support_channel_id: int | None = None,
    help_forum_id: int | None = None,
    feedback: FeedbackStore | None = None,
) -> bool:
    query = extract_query(
        message,
        bot_user_id,
        support_channel_id,
        help_forum_id,
    )
    if query is None:
        query = await _unresolved_reply_query(message, bot_user_id)
    if query is None:
        return False

    history = await _reply_history(message, bot_user_id, help_forum_id)
    async with message.channel.typing():
        answer = await pipeline.answer(query, history)

    embed = answer_embed(answer)
    send = {
        "embed": embed,
        "view": answer_view(answer, feedback),
        "allowed_mentions": discord.AllowedMentions.none(),
    }
    if logo_path.is_file():
        send["file"] = discord.File(str(logo_path), filename="logo.png")
        embed.set_author(name="Nero", icon_url="attachment://logo.png")

    response = await message.reply(
        **send,
        mention_author=True,
    )
    response_id = getattr(response, "id", None)
    request_id = getattr(message, "id", None)
    requester_id = getattr(getattr(message, "author", None), "id", None)
    if (
        feedback is not None
        and isinstance(response_id, int)
        and isinstance(request_id, int)
        and isinstance(requester_id, int)
    ):
        feedback.record_answer(
            answer_message_id=response_id,
            request_message_id=request_id,
            requester_id=requester_id,
            decision_kind=answer.route,
            card_id=answer.card_id,
            model_route=f"cloud:{answer.model}" if answer.model else "deterministic",
            source_status="used" if answer.route == "source" else "not_requested",
        )
    return True
