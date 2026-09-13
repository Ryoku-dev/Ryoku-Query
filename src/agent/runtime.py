from __future__ import annotations

from typing import TYPE_CHECKING

from model2vec import StaticModel

from src.agent.answering import Answerer
from src.agent.ollama import OllamaClient
from prowl import ProwlClient
from support import SupportRetriever, load_support_cards

if TYPE_CHECKING:
    from bot import Config


def build_retriever(config: Config) -> SupportRetriever:
    model = StaticModel.from_pretrained(config.model_name)
    cards = load_support_cards(config.support_path)
    return SupportRetriever(cards, model)


def build_answerer(config: Config) -> Answerer:
    return Answerer(
        OllamaClient(
            config.ollama_host,
            config.gemma_model,
            config.lfm_model,
            timeout=config.ollama_timeout,
        ),
        rewrite_enabled=config.llm_rewrite,
    )


def build_prowl(config: Config) -> ProwlClient | None:
    if config.ryoku_repo_path is None:
        return None
    return ProwlClient(
        config.ryoku_repo_path,
        timeout=config.prowl_timeout,
        result_limit=config.prowl_result_limit,
        executable=config.prowl_executable,
    )
