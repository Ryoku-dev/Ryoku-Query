from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import discord
from dotenv import load_dotenv

from feedback import FeedbackStore, FeedbackView
from src.agent.runtime import build_index, build_pipeline, build_prowl
from src.handlers.messages import extract_query, handle_message


@dataclass(frozen=True)
class Config:
    token: str
    model_name: str
    support_path: Path
    gui_map_path: Path
    docs_path: Path
    ryoku_repo_path: Path | None
    prowl_timeout: float
    prowl_result_limit: int
    prowl_executable: str
    support_channel_id: int
    help_forum_id: int
    feedback_path: Path
    cloud_enabled: bool
    llm_socket: str
    llm_model: str
    llm_timeout: float


def load_config() -> Config:
    load_dotenv()
    token = os.getenv("TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TOKEN in .env")
    prowl_timeout = float(os.getenv("PROWL_TIMEOUT_SECONDS", "4"))
    prowl_limit = int(os.getenv("PROWL_RESULT_LIMIT", "20"))
    support_channel_id = int(os.getenv("SUPPORT_CHANNEL_ID", "0"))
    help_forum_id = int(os.getenv("HELP_FORUM_ID", "0"))
    cloud_value = os.getenv("NERO_CLOUD", "1").strip()
    llm_timeout = float(os.getenv("NERO_LLM_TIMEOUT_SECONDS", "25"))
    if not 0.1 <= prowl_timeout <= 30:
        raise RuntimeError("PROWL_TIMEOUT_SECONDS must be between 0.1 and 30")
    if not 1 <= prowl_limit <= 20:
        raise RuntimeError("PROWL_RESULT_LIMIT must be between 1 and 20")
    if support_channel_id < 0:
        raise RuntimeError("SUPPORT_CHANNEL_ID must be a non-negative Discord channel ID")
    if help_forum_id < 0:
        raise RuntimeError("HELP_FORUM_ID must be a non-negative Discord channel ID")
    if cloud_value not in {"0", "1"}:
        raise RuntimeError("NERO_CLOUD must be 0 or 1")
    if not 5 <= llm_timeout <= 60:
        raise RuntimeError("NERO_LLM_TIMEOUT_SECONDS must be between 5 and 60")
    repo = os.getenv("RYOKU_REPO_PATH", "").strip()
    return Config(
        token=token,
        model_name=os.getenv("MODEL_NAME", "minishlab/potion-base-32M"),
        support_path=Path(os.getenv("SUPPORT_PATH", "data/support.json")),
        gui_map_path=Path(os.getenv("GUI_MAP_PATH", "data/gui_map.json")),
        docs_path=Path(os.getenv("DOCS_CHUNKS_PATH", "data/docs_chunks.json")),
        ryoku_repo_path=Path(repo) if repo else None,
        prowl_timeout=prowl_timeout,
        prowl_result_limit=prowl_limit,
        prowl_executable=os.getenv("PROWL_AGENT_PATH", "prowl-agent").strip(),
        support_channel_id=support_channel_id,
        help_forum_id=help_forum_id,
        feedback_path=Path(
            os.getenv("FEEDBACK_DB_PATH", "runtime/nero-feedback.sqlite3")
        ),
        cloud_enabled=cloud_value == "1",
        llm_socket=os.getenv(
            "NERO_LLM_SOCKET",
            "/run/nero-llm/llm.sock",
        ).strip(),
        llm_model=os.getenv("NERO_LLM_MODEL", "gpt-6.1-sol").strip(),
        llm_timeout=llm_timeout,
    )


def create_client(config: Config) -> discord.Client:
    index = build_index(config)
    prowl = build_prowl(config)
    pipeline = build_pipeline(config, index, prowl)
    feedback = FeedbackStore(config.feedback_path)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    client.add_view(FeedbackView(feedback))

    @client.event
    async def on_ready():
        counts = index.kind_counts
        cloud = config.llm_model if config.cloud_enabled else "cloud off"
        print(
            f"Logged in as {client.user} (ID: {client.user.id}); "
            f"{counts['card']} cards, {counts['gui']} GUI entries, "
            f"{counts['doc']} docs chunks; {cloud}; "
            f"Prowl {'enabled' if prowl else 'disabled'}"
        )

    @client.event
    async def on_message(message):
        await handle_message(
            message,
            client.user.id,
            pipeline,
            Path("data/logo.png"),
            support_channel_id=config.support_channel_id,
            help_forum_id=config.help_forum_id,
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
