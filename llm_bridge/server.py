from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUEST_LIMIT = 256 * 1024
_MAX_PROMPT_CHARS = 60_000
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_CONTRACT_ERRORS = frozenset(
    {
        "timeout",
        "busy",
        "bad_request",
        "model_not_allowed",
        "auth_unavailable",
        "provider_failed",
        "worker_failed",
    }
)
_DEFAULT_MODELS = "gpt-6-luna,gpt-6.1-sol,gpt-5.6-terra"


@dataclass(frozen=True)
class BridgeConfig:
    socket_path: Path
    models: frozenset[str]
    concurrency: int
    queue_wait: float
    timeout: float
    hermes_python: str
    worker_path: str
    home: str

    @classmethod
    def from_env(cls) -> BridgeConfig:
        concurrency = int(os.environ.get("NERO_LLM_CONCURRENCY", "2"))
        queue_wait = float(os.environ.get("NERO_LLM_QUEUE_WAIT", "10"))
        timeout = float(os.environ.get("NERO_LLM_TIMEOUT", "30"))
        if concurrency < 1 or queue_wait <= 0 or timeout <= 0:
            raise ValueError("LLM bridge limits must be positive")
        models = frozenset(
            model.strip()
            for model in os.environ.get("NERO_LLM_MODELS", _DEFAULT_MODELS).split(",")
            if model.strip()
        )
        default_worker = str(Path(__file__).with_name("hermes_worker.py").resolve())
        return cls(
            socket_path=Path(
                os.environ.get("NERO_LLM_SOCKET", "/run/nero-llm/llm.sock")
            ),
            models=models,
            concurrency=concurrency,
            queue_wait=queue_wait,
            timeout=timeout,
            hermes_python=os.environ.get(
                "NERO_HERMES_PYTHON",
                "/home/neur0map/.hermes/hermes-agent/venv/bin/python",
            ),
            worker_path=os.environ.get("NERO_LLM_WORKER", default_worker),
            home=os.environ.get("NERO_HERMES_HOME_DIR", "/home/neur0map"),
        )


@dataclass(frozen=True)
class _Request:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    prompt_chars: int


class LLMBridge:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self._slots = asyncio.Semaphore(config.concurrency)
        self._server: asyncio.AbstractServer | None = None
        self._connections: set[asyncio.Task[None]] = set()

    @classmethod
    def from_env(cls) -> LLMBridge:
        return cls(BridgeConfig.from_env())

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("LLM bridge is already running")
        try:
            self.config.socket_path.unlink()
        except FileNotFoundError:
            pass
        previous_umask = os.umask(0o007)
        try:
            self._server = await asyncio.start_unix_server(
                self._client_connected,
                path=str(self.config.socket_path),
                limit=_REQUEST_LIMIT,
            )
            os.chmod(self.config.socket_path, 0o660)
        finally:
            os.umask(previous_umask)

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            self.config.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        started = time.monotonic()
        model = "-"
        prompt_chars = 0
        response: dict[str, Any]
        try:
            try:
                raw = await reader.readline()
            except (ValueError, asyncio.LimitOverrunError):
                raw = b""
                response = {"status": "error", "error": "bad_request"}
            else:
                if not raw or len(raw) > _REQUEST_LIMIT:
                    response = {"status": "error", "error": "bad_request"}
                else:
                    request, error, model, prompt_chars = self._parse_request(raw)
                    if error is not None:
                        response = {"status": "error", "error": error}
                    else:
                        assert request is not None
                        response = await self._dispatch(request)
        except asyncio.CancelledError:
            response = {"status": "error", "error": "worker_failed"}
        except Exception:
            response = {"status": "error", "error": "worker_failed"}
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        if response.get("status") == "ok":
            response["latency_ms"] = latency_ms
            outcome = "status=ok"
        else:
            error = response.get("error", "worker_failed")
            outcome = f"error={error}"
        safe_model = json.dumps(model, ensure_ascii=True)
        print(
            f"model={safe_model} {outcome} latency_ms={latency_ms} "
            f"prompt_chars={prompt_chars}",
            file=sys.stderr,
            flush=True,
        )
        try:
            writer.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            await writer.drain()
        except (BrokenPipeError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionError):
                pass

    def _parse_request(
        self, raw: bytes
    ) -> tuple[_Request | None, str | None, str, int]:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "bad_request", "-", 0
        if not isinstance(value, dict):
            return None, "bad_request", "-", 0
        model_value = value.get("model")
        model = model_value if isinstance(model_value, str) else "-"
        messages_value = value.get("messages")
        prompt_chars = self._prompt_chars(messages_value)
        if set(value) != {"model", "messages", "max_tokens"}:
            return None, "bad_request", model, prompt_chars
        if not isinstance(model_value, str) or not model_value:
            return None, "bad_request", model, prompt_chars
        if model_value not in self.config.models:
            return None, "model_not_allowed", model, prompt_chars
        if not isinstance(messages_value, list) or not messages_value:
            return None, "bad_request", model, prompt_chars
        messages: list[dict[str, str]] = []
        total = 0
        for message in messages_value:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                return None, "bad_request", model, prompt_chars
            role = message.get("role")
            content = message.get("content")
            if role not in _ALLOWED_ROLES or not isinstance(content, str):
                return None, "bad_request", model, prompt_chars
            total += len(content)
            if total > _MAX_PROMPT_CHARS:
                return None, "bad_request", model, total
            messages.append({"role": role, "content": content})
        max_tokens = value.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 1500
        ):
            return None, "bad_request", model, total
        return _Request(model, messages, max_tokens, total), None, model, total

    @staticmethod
    def _prompt_chars(messages: Any) -> int:
        if not isinstance(messages, list):
            return 0
        return sum(
            len(content)
            for message in messages
            if isinstance(message, dict)
            and isinstance((content := message.get("content")), str)
        )

    async def _dispatch(self, request: _Request) -> dict[str, Any]:
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self.config.queue_wait
            )
        except TimeoutError:
            return {"status": "error", "error": "busy"}
        try:
            return await self._run_worker(request)
        finally:
            self._slots.release()

    async def _run_worker(self, request: _Request) -> dict[str, Any]:
        worker_env = {
            "HOME": self.config.home,
            "HERMES_HOME": str(Path(self.config.home) / ".hermes"),
            "PATH": "/usr/bin:/bin",
        }
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.hermes_python,
                "-I",
                self.config.worker_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=worker_env,
                start_new_session=True,
            )
        except (OSError, ValueError):
            return {"status": "error", "error": "worker_failed"}
        payload = json.dumps(
            {
                "model": request.model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=self.config.timeout
            )
        except TimeoutError:
            await self._kill_worker(process)
            return {"status": "error", "error": "timeout"}
        except asyncio.CancelledError:
            await self._kill_worker(process)
            raise
        except (BrokenPipeError, ConnectionError, OSError):
            await self._kill_worker(process)
            return {"status": "error", "error": "worker_failed"}
        if process.returncode != 0:
            return {"status": "error", "error": "worker_failed"}
        try:
            result = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "error", "error": "worker_failed"}
        if not isinstance(result, dict):
            return {"status": "error", "error": "worker_failed"}
        if set(result) == {"error"} and isinstance(result["error"], str):
            error = result["error"]
            mapped = error if error in _CONTRACT_ERRORS else "provider_failed"
            return {"status": "error", "error": mapped}
        if (
            set(result) != {"model", "content"}
            or result.get("model") != request.model
            or not isinstance(result.get("content"), str)
        ):
            return {"status": "error", "error": "worker_failed"}
        return {"status": "ok", "text": result["content"]}

    @staticmethod
    async def _kill_worker(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except ProcessLookupError:
            pass


async def _run() -> None:
    bridge = LLMBridge.from_env()
    await bridge.start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:
            continue
        installed_signals.append(signum)
    try:
        await stopped.wait()
    finally:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        await bridge.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
