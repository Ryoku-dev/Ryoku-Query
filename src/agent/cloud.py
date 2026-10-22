from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

_RESPONSE_LIMIT = 1024 * 1024
_MAX_TEXT_CHARS = 1900
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class LLMResult:
    status: str
    text: str = ""
    error: str | None = None


class CloudClient:
    def __init__(self, socket_path: str, model: str, *, timeout: float = 25.0) -> None:
        self.socket_path = socket_path
        self.model = model
        self.timeout = timeout

    async def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int = 500
    ) -> LLMResult:
        try:
            response = await asyncio.wait_for(
                self._exchange(messages, max_tokens), timeout=self.timeout
            )
        except TimeoutError:
            return LLMResult("unavailable", error="timeout")
        except (OSError, ConnectionError):
            return LLMResult("unavailable", error="connection_failed")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return LLMResult("unavailable", error="invalid_response")
        if not isinstance(response, dict):
            return LLMResult("unavailable", error="invalid_response")
        if response.get("status") != "ok":
            error = response.get("error")
            return LLMResult(
                "unavailable",
                error=error if isinstance(error, str) and error else "invalid_response",
            )
        text = response.get("text")
        if not isinstance(text, str):
            return LLMResult("unavailable", error="invalid_response")
        text = _THINK_BLOCK.sub("", text)
        if "<think" in text:
            text = ""
        text = text.strip()[:_MAX_TEXT_CHARS].rstrip()
        if not text:
            return LLMResult("unavailable", error="empty")
        return LLMResult("ok", text=text)

    async def _exchange(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> Any:
        reader, writer = await asyncio.open_unix_connection(
            self.socket_path, limit=_RESPONSE_LIMIT
        )
        try:
            request = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            writer.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            await writer.drain()
            raw = await reader.readline()
            if not raw or len(raw) > _RESPONSE_LIMIT:
                raise ValueError("invalid response")
            return json.loads(raw)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionError):
                pass
