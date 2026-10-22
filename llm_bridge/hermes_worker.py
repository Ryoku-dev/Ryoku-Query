# Vendored from Mika by the same author.
# Keep this file as a standalone script. It imports Hermes modules that exist
# only in Hermes' own interpreter, so the bot's lint cannot resolve them.
# pylint: disable=import-error,no-name-in-module
"""Single-shot, tool-free Hermes OAuth worker; stdin/stdout carry one JSON turn.

Run only with the Hermes Python environment, not the bot's Python environment.
Never print raw provider errors, messages, tokens or credentials to stderr.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _prepare_hermes_imports() -> None:  # noqa: C901
    """Expose only the installed editable Hermes source to this isolated child."""
    agent_spec = importlib.util.find_spec("agent")
    cli_spec = importlib.util.find_spec("hermes_cli")
    if agent_spec is None or cli_spec is None or not agent_spec.origin or not cli_spec.origin:
        raise ImportError("Hermes installation unavailable")
    agent_file = Path(agent_spec.origin)
    cli_file = Path(cli_spec.origin)
    if not agent_file.is_absolute() or not cli_file.is_absolute():
        raise ImportError("Hermes installation unavailable")
    root = agent_file.parent.parent
    expected = (root / "agent" / "__init__.py", root / "hermes_cli" / "__init__.py")
    if (
        (agent_file, cli_file) != expected
        or list(agent_spec.submodule_search_locations or []) != [str(root / "agent")]
        or list(cli_spec.submodule_search_locations or []) != [str(root / "hermes_cli")]
    ):
        raise ImportError("Hermes installation mismatch")
    yaml_file = root / "hermes_yaml.py"
    # The editable package finder exposes agent/hermes_cli but not root-level
    # hermes_yaml. Do not use cwd, PYTHONPATH, or a caller-supplied source path.
    for path in (root, *expected, agent_file.parent, cli_file.parent, yaml_file):
        if path.resolve(strict=True) != path or path.is_symlink():
            raise ImportError("Hermes installation mismatch")
        metadata = path.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & stat.S_IWOTH:
            raise ImportError("Hermes installation ownership mismatch")
        if not (path.is_dir() if path.suffix != ".py" else path.is_file()):
            raise ImportError("Hermes installation mismatch")
    sys.path.insert(0, str(root))
    yaml_spec = importlib.util.find_spec("hermes_yaml")
    if yaml_spec is None or yaml_spec.origin != str(yaml_file):
        sys.path.remove(str(root))
        raise ImportError("Hermes installation mismatch")
    # The editable finder may still point at an old Python venv after Hermes PM
    # committed a new ABI. Follow Hermes' read-only committed selection, not a
    # caller path; re-enter this exact worker with -I before importing any SDK.
    from pm.environments import committed_venv, venv_python  # type: ignore[import-not-found]  # noqa: I001, PLC0415

    environment = committed_venv(root)
    if environment is not None:
        python = venv_python(environment)
        if (
            not environment.is_absolute()
            or environment.resolve(strict=True) != environment
            or environment.stat().st_uid != os.geteuid()
            or not python.is_file()
            or python.resolve(strict=True).stat().st_uid != os.geteuid()
            or python.stat().st_mode & stat.S_IWOTH
        ):
            raise ImportError("Hermes interpreter mismatch")
        if Path(sys.prefix).resolve() != environment:
            os.execv(str(python), [str(python), "-I", str(Path(__file__).resolve())])  # noqa: S606
    # A deep Hermes import loads hermes_bootstrap. In this dedicated worker,
    # don't let its CLI lazy installer replace the pinned venv with a bare store
    # interpreter or mix a newer site's extension ABI into an older process.
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"


if __name__ == "__main__":
    try:
        _prepare_hermes_imports()
    except Exception:
        sys.stdout.write('{"error":"auth_unavailable"}')
        raise SystemExit(0) from None

try:
    from agent.auxiliary_client import (  # type: ignore[import-not-found]
        _CodexCompletionsAdapter,
        resolve_provider_client,
    )
    from hermes_cli import auth as hermes_auth  # type: ignore[import-not-found]
    from hermes_cli import auth_codex as codex_auth
except Exception:
    if __name__ != "__main__":
        raise
    sys.stdout.write('{"error":"auth_unavailable"}')
    raise SystemExit(0) from None

_MODEL_CODES = {"model_not_found", "unsupported_model", "model_unsupported"}
_HTTP_BAD_REQUEST = 400
_CODEX_ROUTE = "https://chatgpt.com/backend-api/codex"
_MAX_STREAM_CHARS = 120_000
_ALLOWED_OUTPUT_ITEMS = {"message", "reasoning"}


def _model_rejected(error: Exception, model: str) -> bool:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        details = body.get("error", body)
        if isinstance(details, Mapping) and details.get("code") in _MODEL_CODES:
            return True
    # The Codex Responses adapter sometimes exposes a flattened upstream body.
    text = str(error).lower()
    return bool(status == _HTTP_BAD_REQUEST and model.lower() in text and "not supported" in text)


def _is_hermes_oauth_row(row: Any) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("auth_type") == "oauth"
        and row.get("refresh_token")
        and row.get("access_token")
        and str(row.get("base_url") or "").rstrip("/") == _CODEX_ROUTE
    )


def _completed_text(output: list[Any], *, final_only: bool = False) -> str | None:
    parts: list[str] = []
    for item in output:
        if item.type != "message":
            continue
        if final_only and getattr(item, "phase", None) in {"commentary", "analysis"}:
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return None
        for part in content:
            text = getattr(part, "text", None)
            if getattr(part, "type", None) != "output_text" or not isinstance(text, str):
                return None
            parts.append(text)
    return "".join(parts)


def _item_finished(item: Any) -> bool:
    """Messages must say completed; upstream omits `status` on reasoning items."""
    status = getattr(item, "status", None)
    return status == "completed" or (item.type == "reasoning" and status is None)


def _verified_stream(  # noqa: C901, PLR0911, PLR0912, PLR0915
    client: Any, model: str, messages: list[dict[str, Any]]
) -> dict[str, str]:
    # Reuse Hermes' chat-to-Responses conversion, but inspect the raw upstream
    # events ourselves. The auxiliary ChatCompletion result fills `model` from
    # the REQUEST, so checking that synthesized field cannot verify the answer.
    params, selected, _ = _CodexCompletionsAdapter(client, model)._build_responses_kwargs(
        {"model": model, "messages": messages}
    )
    if selected != model or params.get("model") != model:
        return {"error": "model_rejected"}
    params.pop("_wire_aliases", None)
    params.pop("_issuer_kind", None)
    response_id: str | None = None
    completed = False
    items: dict[str, str] = {}
    phases: dict[str, str | None] = {}
    done_ids: set[str] = set()
    done_items: dict[str, Any] = {}
    chunks_by_item: dict[str, list[str]] = {}
    finalized_parts: dict[tuple[str, int], str] = {}
    text_size = 0
    final_content = ""
    with client.responses.create(**params, stream=True) as events:
        for event in events:
            kind = getattr(event, "type", "")
            # One response per turn. A later created/completed event must not
            # validate text from a different, unfinished response.
            if completed:
                return {"error": "invalid_response"}
            if kind in {"response.created", "response.completed"}:
                upstream = getattr(event, "response", None)
                if getattr(upstream, "model", None) != model:
                    return {"error": "model_rejected"}
                current_id = getattr(upstream, "id", None)
                if not isinstance(current_id, str) or not current_id:
                    return {"error": "invalid_response"}
                if kind == "response.created":
                    if response_id is not None:
                        return {"error": "invalid_response"}
                    response_id = current_id
                else:
                    if response_id != current_id:
                        return {"error": "invalid_response"}
                    if (
                        getattr(upstream, "status", None) != "completed"
                        or getattr(upstream, "incomplete_details", None) is not None
                    ):
                        return {"error": "incomplete_response"}
                    output = getattr(upstream, "output", None)
                    # Codex sometimes returns null here; the per-item .done
                    # frames are the authoritative completed output in that case.
                    if output is None:
                        output = list(done_items.values())
                    if not isinstance(output, list):
                        return {"error": "invalid_response"}
                    if any(
                        getattr(item, "type", None) not in _ALLOWED_OUTPUT_ITEMS for item in output
                    ):
                        return {"error": "unexpected_tool_call"}
                    if any(
                        getattr(item, "role", None) != "assistant"
                        for item in output
                        if item.type == "message"
                    ):
                        return {"error": "invalid_response"}
                    if not all(_item_finished(item) for item in output):
                        return {"error": "incomplete_response"}
                    if (
                        {getattr(item, "id", None): getattr(item, "type", None) for item in output}
                        != items
                        or done_ids != set(items)
                        or len(output) != len(items)
                    ):
                        return {"error": "invalid_response"}
                    if any(getattr(item, "phase", None) != phases.get(item.id) for item in output):
                        return {"error": "invalid_response"}
                    final_text = _completed_text(output)
                    if final_text is None:
                        return {"error": "invalid_response"}
                    if any(
                        _completed_text([item]) != _completed_text([done_items[item.id]])
                        for item in output
                        if item.type == "message"
                    ):
                        return {"error": "invalid_response"}
                    final_content = _completed_text(output, final_only=True) or ""
                    completed = True
            elif kind in {"response.failed", "response.incomplete"}:
                return {"error": "incomplete_response"}
            elif kind == "response.output_text.delta":
                delta_item_id = getattr(event, "item_id", None)
                if (
                    response_id is None
                    or not isinstance(delta_item_id, str)
                    or items.get(delta_item_id) != "message"
                    or delta_item_id in done_ids
                    or not isinstance(getattr(event, "delta", None), str)
                ):
                    return {"error": "invalid_response"}
                text_size += len(event.delta)
                if text_size > _MAX_STREAM_CHARS:
                    return {"error": "incomplete_response"}
                chunks_by_item.setdefault(delta_item_id, []).append(event.delta)
            elif kind in {"response.output_item.added", "response.output_item.done"}:
                if response_id is None:
                    return {"error": "invalid_response"}
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None)
                if item_type not in _ALLOWED_OUTPUT_ITEMS:
                    return {"error": "unexpected_tool_call"}
                if item_type == "message" and getattr(item, "role", None) != "assistant":
                    return {"error": "invalid_response"}
                phase = getattr(item, "phase", None)
                if item_type == "message" and phase not in {
                    None,
                    "final",
                    "final_answer",
                    "commentary",
                    "analysis",
                }:
                    return {"error": "invalid_response"}
                item_id = getattr(item, "id", None)
                if not isinstance(item_id, str) or not item_id:
                    return {"error": "invalid_response"}
                if kind == "response.output_item.added":
                    if item_id in items:
                        return {"error": "invalid_response"}
                    items[item_id] = item_type
                    phases[item_id] = phase
                elif items.get(item_id) != item_type or phases[item_id] != phase:
                    return {"error": "invalid_response"}
                elif not _item_finished(item):
                    return {"error": "incomplete_response"}
                elif item_id in done_ids:
                    return {"error": "invalid_response"}
                else:
                    if item_type == "message":
                        item_text = _completed_text([item])
                        if item_text is None:
                            return {"error": "invalid_response"}
                        deltas = chunks_by_item.get(item_id, [])
                        if deltas and "".join(deltas) != item_text:
                            return {"error": "incomplete_response"}
                        content = getattr(item, "content", None)
                        if not isinstance(content, list):
                            return {"error": "invalid_response"}
                        if any(
                            index >= len(content)
                            or getattr(content[index], "type", None) != "output_text"
                            or getattr(content[index], "text", None) != text
                            for (key, index), text in finalized_parts.items()
                            if key == item_id
                        ):
                            return {"error": "invalid_response"}
                    done_ids.add(item_id)
                    done_items[item_id] = item
            elif kind.startswith("response.output_item.") or response_id is None:
                return {"error": "invalid_response"}
            elif kind in {
                "response.content_part.added",
                "response.content_part.done",
                "response.output_text.done",
            }:
                item_id = getattr(event, "item_id", None)
                if not isinstance(item_id, str) or items.get(item_id) != "message":
                    return {"error": "invalid_response"}
                if kind in {"response.content_part.done", "response.output_text.done"}:
                    index = getattr(event, "content_index", None)
                    text = (
                        getattr(getattr(event, "part", None), "text", None)
                        if kind == "response.content_part.done"
                        else getattr(event, "text", None)
                    )
                    if (
                        item_id in done_ids
                        or not isinstance(index, int)
                        or index < 0
                        or not isinstance(text, str)
                        or (
                            kind == "response.content_part.done"
                            and getattr(getattr(event, "part", None), "type", None) != "output_text"
                        )
                    ):
                        return {"error": "invalid_response"}
                    key = (item_id, index)
                    if key in finalized_parts and finalized_parts[key] != text:
                        return {"error": "invalid_response"}
                    finalized_parts[key] = text
            elif kind not in {"response.in_progress", "response.metadata"}:
                # Unknown event types cannot cross the tool-free boundary.
                if "call" in kind or "tool" in kind:
                    return {"error": "unexpected_tool_call"}
                return {"error": "invalid_response"}
    if not completed:
        return {"error": "incomplete_response"}
    return {"model": model, "content": final_content.strip()}


async def _complete(request: dict[str, Any]) -> dict[str, str]:  # noqa: C901, PLR0911
    model, messages = request.get("model"), request.get("messages")
    if not isinstance(model, str) or not model or not isinstance(messages, list):
        return {"error": "invalid_request"}
    if not all(
        isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"} for m in messages
    ):
        return {"error": "invalid_request"}
    # Require the sole stored Hermes Codex OAuth grant, not a static key or an
    # alternative pool account. The exact route prevents a bearer from being
    # sent to an unreviewed third-party URL.
    try:
        rows = hermes_auth.read_credential_pool("openai-codex")
        if len(rows) != 1 or not _is_hermes_oauth_row(rows[0]):
            return {"error": "auth_unavailable"}
    except Exception:
        return {"error": "auth_unavailable"}
    try:
        client, selected = resolve_provider_client("openai-codex", model=model, raw_codex=True)
    except Exception:
        return {"error": "model_rejected"}
    if client is None or selected != model:
        if client is not None:
            client.close()
        return {"error": "model_rejected"}
    try:
        # Selection may refresh the single-use token under Hermes' own lock.
        # Compare AFTER selection; a read-only precheck would reject a valid
        # freshly rotated credential and strand Mika until another refresh.
        fresh = hermes_auth.read_credential_pool("openai-codex")
        credentials = codex_auth.resolve_codex_runtime_credentials(read_only=True)
        if len(fresh) != 1 or not _is_hermes_oauth_row(fresh[0]):
            return {"error": "auth_mismatch"}
        if (
            client.api_key != fresh[0]["access_token"]
            or client.api_key != credentials.get("api_key")
            or str(client.base_url).rstrip("/") != _CODEX_ROUTE
            or str(credentials.get("base_url")).rstrip("/") != _CODEX_ROUTE
        ):
            return {"error": "auth_mismatch"}
        return await asyncio.to_thread(_verified_stream, client, model, messages)
    except Exception as error:
        return {"error": "model_rejected" if _model_rejected(error, model) else "provider_failed"}
    finally:
        client.close()


def main() -> None:
    try:
        request = json.load(sys.stdin)
        result = (
            asyncio.run(_complete(request))
            if isinstance(request, dict)
            else {"error": "invalid_request"}
        )
    except Exception:
        result = {"error": "provider_failed"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
