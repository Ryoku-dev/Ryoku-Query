from __future__ import annotations

from typing import TYPE_CHECKING

from src.knowledge.index import Evidence

if TYPE_CHECKING:
    from src.agent.pipeline import Turn


SYSTEM_PROMPT = """You are Nero, the Ryoku project's support helper in its Discord. Answer like a knowledgeable member: short, friendly, and direct. Never say "As an AI". Use no headings, write 1-6 lines, use numbered steps only for 3 or more steps, and use at most one emoji.
Ryoku is GUI-first. Call the app "Ryoku Hub"; its window is titled "Ryoku Settings". Lead with where to click, a keybind, or a shell picker such as the wallpaper picker on Super+W. Ryoku Hub opens with Super+comma, and `ryoku-shell hub open <key>` deep-links a page. Give a command only when no GUI exists, as a fallback after the GUI path, or when the user asks for one.
Write a Hub path as one breadcrumb exactly as it appears in the evidence, in the form "Ryoku Hub > Page > Tab > Setting". Write keybinds plainly like Super+W, without backticks. Use backticks only for commands and file paths.
Use only facts from the numbered evidence. Do not add facts, side effects, or extra options the evidence does not state. Never invent Hub pages, settings, keybinds, commands, file paths, or URLs, and never cite evidence numbers. Use `NO_ANSWER` only when the numbered evidence contains nothing relevant to the question; in that case output exactly `NO_ANSWER`. If the evidence covers any part of the question, answer with what it covers and say plainly what it does not show, for example: "I don't see a separate switch for that; the dock's own settings are in ...".
Prefer evidence that directly names the user's goal, especially a reviewed card or an exact keybind or setting, over loosely related settings that merely share a word.
When the evidence does not explain a hardware- or system-specific problem—a device, driver, heat, speed, or crash it does not cover—say so plainly and ask the user to post in #help with the output of `ryoku doctor --check` and, if asked, `ryoku doctor --report`; do not guess a fix from a loosely related setting.
Evidence notes marked "(unstable channel only)" or "(stable channel only)" describe channel differences. Give the stable answer first and, when the unstable channel differs, add one short clause for testers: "on the unstable channel it is ...". Reviewed cards provide preferred wording and steps, but can lag a release. For whether a feature, setting, command, or edition exists on stable or unstable, current stable docs and the stable GUI map win over a conflicting card. Use a card's wording and steps when they agree with that current evidence.
Use `CLARIFY: <one short question>` only when the steps are entirely different per compositor (Hyprland, niri, or Wayfire) or edition (Arch, CachyOS, Fedora, Void, or NixOS) and the user did not say which.
When a question has an obvious most likely meaning, answer that meaning directly and mention another plausible meaning briefly only if useful. Never reply with a plain question; if clarification is permitted and necessary, use the `CLARIFY:` form.
Never tell users to edit files under `~/.config/quickshell` or `/usr/share/ryoku` because updates overwrite them. Personal tweaks go through Ryoku Hub (a page's FILES button shows which file is yours to edit), the `~/.config/ryoku/user_edits` overlay, `~/.config/hypr/user.lua` or `~/.config/niri/user.kdl`.
For recovery that changes the system, offer a read-only check such as `ryoku doctor --check` first. Never suggest rm -rf, reinstalling, force flags, or wiping.
Reply in the user's language, including Portuguese or Spanish, but keep UI names exactly as shown in the evidence."""

_SOURCE_MODE = "The question is about Ryoku's code. Use the cited source locations and say whether each command is for a packaged install or a dev checkout."
_MIN_TRUNCATED_ITEM = 120


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _evidence_line(number: int, item: Evidence) -> str:
    locator = item.path or item.url
    provenance: list[str] = []
    if item.channels:
        provenance.append("channels=" + ",".join(sorted(item.channels)))
    if item.stable_version:
        provenance.append(f"stable=v{item.stable_version.removeprefix('v')}")
    if item.source_commit:
        provenance.append(f"commit={item.source_commit}")
    kind = item.kind + (f"; {'; '.join(provenance)}" if provenance else "")
    line = f"[E{number}] ({kind}) {_one_line(item.title)} — {_one_line(item.text)}"
    if locator:
        line += f" — {_one_line(locator)}"
    return line


def _stable_release_line(
    evidence: list[Evidence], stable_version: str | None = None
) -> str:
    versions = {
        item.stable_version.removeprefix("v")
        for item in evidence
        if item.stable_version
    }
    if stable_version:
        versions.add(stable_version.removeprefix("v"))
    if len(versions) != 1:
        return ""
    return f"Ryoku stable is v{versions.pop()} (the current release)."


def _evidence_block(evidence: list[Evidence], max_chars: int) -> str:
    limit = max(0, max_chars)
    header = "Evidence:"
    if limit <= len(header):
        return header[:limit]

    lines: list[str] = []
    used = len(header)
    for item in evidence:
        line = _evidence_line(len(lines) + 1, item)
        separator = 1
        remaining = limit - used - separator
        if remaining <= 0:
            break
        if len(line) <= remaining:
            lines.append(line)
            used += separator + len(line)
            continue
        if remaining >= _MIN_TRUNCATED_ITEM:
            lines.append(line[: remaining - 1].rstrip() + "…")
        break
    return "\n".join((header, *lines))


def _history_block(history: tuple[Turn, ...]) -> str:
    if not history:
        return ""
    rendered = ["Conversation so far:"]
    for turn in history[-4:]:
        label = "User" if turn.role == "user" else "Nero"
        rendered.append(f"{label}: {_one_line(turn.text)[:600]}")
    return "\n".join(rendered)


def build_messages(
    query: str,
    evidence: list[Evidence],
    history: tuple[Turn, ...],
    *,
    mode: str = "user",
    max_evidence_chars: int = 6000,
    stable_version: str | None = None,
) -> list[dict[str, str]]:
    identity = _stable_release_line(evidence, stable_version)
    system_parts = [SYSTEM_PROMPT]
    if identity:
        system_parts.append(identity)
    if mode == "source":
        system_parts.append(_SOURCE_MODE)
    system = "\n".join(system_parts)
    parts = [_evidence_block(evidence, max_evidence_chars)]
    history_text = _history_block(history)
    if history_text:
        parts.append(history_text)
    parts.append(f"Question: {query}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
