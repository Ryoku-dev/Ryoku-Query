from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_bridge.server import LLMBridge


_FAKE_WORKER = '''from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

request = json.load(sys.stdin)
content = request["messages"][-1]["content"]
if content.startswith("sleep|"):
    _, seconds, marker = content.split("|", 2)
    Path(marker).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(float(seconds))
    content = "finished"
sys.stdout.write(json.dumps({"model": request["model"], "content": "answer:" + content}))
'''


class LLMBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.socket_path = root / "llm.sock"
        self.worker_path = root / "fake_worker.py"
        self.worker_path.write_text(_FAKE_WORKER, encoding="utf-8")
        self.bridges: list[LLMBridge] = []

    async def asyncTearDown(self) -> None:
        for bridge in reversed(self.bridges):
            await bridge.close()
        self.temporary.cleanup()

    async def _start_bridge(
        self,
        *,
        concurrency: int = 2,
        queue_wait: float = 0.1,
        timeout: float = 1.0,
    ) -> LLMBridge:
        environment = {
            "NERO_LLM_SOCKET": str(self.socket_path),
            "NERO_LLM_MODELS": "gpt-6-luna",
            "NERO_LLM_CONCURRENCY": str(concurrency),
            "NERO_LLM_QUEUE_WAIT": str(queue_wait),
            "NERO_LLM_TIMEOUT": str(timeout),
            "NERO_HERMES_PYTHON": sys.executable,
            "NERO_LLM_WORKER": str(self.worker_path),
            "NERO_HERMES_HOME_DIR": self.temporary.name,
        }
        with patch.dict(os.environ, environment):
            bridge = LLMBridge.from_env()
        await bridge.start()
        self.bridges.append(bridge)
        return bridge

    async def _request(self, value: object) -> dict[str, object]:
        reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
        try:
            if isinstance(value, bytes):
                raw = value
            else:
                raw = json.dumps(value).encode("utf-8") + b"\n"
            writer.write(raw)
            await writer.drain()
            response = await reader.readline()
            return json.loads(response)
        finally:
            writer.close()
            await writer.wait_closed()

    @staticmethod
    def _valid_request(content: str = "hello") -> dict[str, object]:
        return {
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 500,
        }

    async def _wait_for_file(self, path: Path) -> None:
        deadline = time.monotonic() + 1
        while not path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        self.assertTrue(path.exists())

    async def test_ok_worker_result_is_mapped(self) -> None:
        await self._start_bridge()

        response = await self._request(self._valid_request("hello"))

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["text"], "answer:hello")
        self.assertIsInstance(response["latency_ms"], int)

    async def test_model_not_allowed(self) -> None:
        await self._start_bridge()
        request = self._valid_request()
        request["model"] = "not-allowed"

        response = await self._request(request)

        self.assertEqual(
            response, {"status": "error", "error": "model_not_allowed"}
        )

    async def test_bad_request(self) -> None:
        await self._start_bridge()
        request = self._valid_request()
        request["messages"] = []

        response = await self._request(request)

        self.assertEqual(response, {"status": "error", "error": "bad_request"})

    async def test_timeout_kills_worker(self) -> None:
        await self._start_bridge(timeout=0.2)
        marker = Path(self.temporary.name) / "timeout.pid"

        response = await self._request(
            self._valid_request(f"sleep|30|{marker}")
        )

        self.assertEqual(response, {"status": "error", "error": "timeout"})
        pid = int(marker.read_text(encoding="utf-8"))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_busy_when_semaphore_is_saturated(self) -> None:
        await self._start_bridge(concurrency=1, queue_wait=0.02, timeout=1)
        marker = Path(self.temporary.name) / "busy.pid"
        first = asyncio.create_task(
            self._request(self._valid_request(f"sleep|0.15|{marker}"))
        )
        await self._wait_for_file(marker)

        second_response = await self._request(self._valid_request("second"))
        first_response = await first

        self.assertEqual(
            second_response, {"status": "error", "error": "busy"}
        )
        self.assertEqual(first_response["status"], "ok")

    async def test_log_does_not_include_prompt_text(self) -> None:
        await self._start_bridge()
        prompt = "private-prompt-marker-19f7"
        captured = io.StringIO()

        with contextlib.redirect_stderr(captured):
            response = await self._request(self._valid_request(prompt))

        self.assertEqual(response["status"], "ok")
        log_line = captured.getvalue()
        self.assertNotIn(prompt, log_line)
        self.assertIn('model="gpt-6-luna"', log_line)
        self.assertIn("status=ok", log_line)
        self.assertIn(f"prompt_chars={len(prompt)}", log_line)


if __name__ == "__main__":
    unittest.main()
