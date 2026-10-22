from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.agent.cloud import CloudClient, LLMResult


class CloudClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.temporary.name) / "cloud.sock")
        self.server: asyncio.AbstractServer | None = None
        self.handlers: set[asyncio.Task[None]] = set()

    async def asyncTearDown(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        if self.handlers:
            await asyncio.gather(*self.handlers, return_exceptions=True)
        self.temporary.cleanup()

    async def _serve(self, response: dict[str, object], *, delay: float = 0) -> None:
        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                await reader.readline()
                if delay:
                    await asyncio.sleep(delay)
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        def connected(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            task = asyncio.create_task(handle(reader, writer))
            self.handlers.add(task)

        self.server = await asyncio.start_unix_server(connected, self.socket_path)

    async def test_ok_response(self) -> None:
        await self._serve({"status": "ok", "text": "  Ready to help.  ", "latency_ms": 3})
        client = CloudClient(self.socket_path, "gpt-6-luna")

        result = await client.complete(
            [{"role": "user", "content": "How do I update?"}], max_tokens=321
        )

        self.assertEqual(result, LLMResult("ok", text="Ready to help."))

    async def test_error_status_is_unavailable(self) -> None:
        await self._serve({"status": "error", "error": "busy"})
        client = CloudClient(self.socket_path, "gpt-6-luna")

        result = await client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, LLMResult("unavailable", error="busy"))

    async def test_slow_server_times_out(self) -> None:
        await self._serve({"status": "ok", "text": "late"}, delay=0.05)
        client = CloudClient(self.socket_path, "gpt-6-luna", timeout=0.01)

        result = await client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, LLMResult("unavailable", error="timeout"))

    async def test_missing_socket_is_unavailable(self) -> None:
        client = CloudClient(self.socket_path, "gpt-6-luna")

        result = await client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, LLMResult("unavailable", error="connection_failed"))

    async def test_think_blocks_are_removed(self) -> None:
        await self._serve(
            {
                "status": "ok",
                "text": "<think>private\nreasoning</think>\nUse Ryoku Hub.",
            }
        )
        client = CloudClient(self.socket_path, "gpt-6-luna")

        result = await client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, LLMResult("ok", text="Use Ryoku Hub."))

    async def test_unclosed_think_discards_answer(self) -> None:
        await self._serve({"status": "ok", "text": "answer <think>unfinished"})
        client = CloudClient(self.socket_path, "gpt-6-luna")

        result = await client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, LLMResult("unavailable", error="empty"))


if __name__ == "__main__":
    unittest.main()
