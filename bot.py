from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import discord
from dotenv import load_dotenv

from feedback import FeedbackStore, FeedbackView
from src.agent.runtime import build_answerer, build_prowl, build_retriever
from src.handlers.messages import extract_query, handle_message


@dataclass(frozen=True)
class Config:
    token: str
    model_name: str
    support_path: Path
    ryoku_repo_path: Path | None
    prowl_timeout: float
    prowl_result_limit: int
    prowl_executable: str
    support_channel_id: int
    ollama_host: str
    gemma_model: str
    lfm_model: str
    ollama_timeout: float
    feedback_path: Path
    llm_rewrite: bool


def load_config() -> Config:
    load_dotenv()
    token = os.getenv("TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TOKEN in .env")
    timeout = float(os.getenv("PROWL_TIMEOUT_SECONDS", "4"))
    limit = int(os.getenv("PROWL_RESULT_LIMIT", "20"))
    support_channel_id = int(os.getenv("SUPPORT_CHANNEL_ID", "0"))
    ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    llm_rewrite_value = os.getenv("NERO_LLM_REWRITE", "0").strip()
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not ollama_host.startswith(("http://", "https://")):
        raise RuntimeError("OLLAMA_HOST must be an HTTP URL")
    if not 0.1 <= timeout <= 30:
        raise RuntimeError("PROWL_TIMEOUT_SECONDS must be between 0.1 and 30")
    if not 1 <= limit <= 20:
        raise RuntimeError("PROWL_RESULT_LIMIT must be between 1 and 20")
    if support_channel_id < 0:
        raise RuntimeError("SUPPORT_CHANNEL_ID must be a positive Discord channel ID")
    if not 10 <= ollama_timeout <= 240:
        raise RuntimeError("OLLAMA_TIMEOUT_SECONDS must be between 10 and 240")
    if llm_rewrite_value not in {"0", "1"}:
        raise RuntimeError("NERO_LLM_REWRITE must be 0 or 1")
    repo = os.getenv("RYOKU_REPO_PATH", "").strip()
    return Config(
        token=token,
        model_name=os.getenv(
            "MODEL_NAME", "minishlab/potion-base-32M"
        ),
        support_path=Path(os.getenv("SUPPORT_PATH", "data/support.json")),
        ryoku_repo_path=Path(repo) if repo else None,
        prowl_timeout=timeout,
        prowl_result_limit=limit,
        prowl_executable=os.getenv("PROWL_AGENT_PATH", "prowl-agent").strip(),
        support_channel_id=support_channel_id,
        ollama_host=ollama_host,
        gemma_model=os.getenv("GEMMA_MODEL", "gemma4:e4b").strip(),
        lfm_model=os.getenv("LFM_MODEL", "lfm2.5:latest").strip(),
        ollama_timeout=ollama_timeout,
        feedback_path=Path(
            os.getenv("FEEDBACK_DB_PATH", "runtime/nero-feedback.sqlite3")
        ),
        llm_rewrite=llm_rewrite_value == "1",
    )


def create_client(config: Config) -> discord.Client:
    retriever = build_retriever(config)
    prowl = build_prowl(config)
    answerer = build_answerer(config)
    feedback = FeedbackStore(config.feedback_path)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    client.add_view(FeedbackView(feedback))

    @client.event
    async def on_ready():
        print(
            f"Logged in as {client.user} (ID: {client.user.id}); "
            f"{len(retriever.cards)} support intents; "
            f"Prowl {'enabled' if prowl else 'disabled'}"
        )

    @client.event
    async def on_message(message):
        await handle_message(
            message,
            client.user.id,
            retriever,
            prowl,
            Path("data/logo.png"),
            answerer=answerer,
            support_channel_id=config.support_channel_id,
            feedback=feedback,
        )

    return client


async def run(config: Config) -> None:
    client = create_client(config)
    async with client:
        await client.start(config.token)


def main() -> None:
    try:
        asyncio.run(run(load_config()))
    except KeyboardInterrupt:
        print("\nNero stopped.")


__all__ = [
    "Config",
    "create_client",
    "extract_query",
    "handle_message",
    "load_config",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
