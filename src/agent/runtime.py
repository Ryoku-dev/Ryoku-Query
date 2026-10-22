from __future__ import annotations

from typing import TYPE_CHECKING

from model2vec import StaticModel

from prowl import ProwlClient
from src.agent.cloud import CloudClient
from src.agent.pipeline import SupportPipeline
from src.knowledge.index import KnowledgeIndex

if TYPE_CHECKING:
    from bot import Config


def build_index(config: Config) -> KnowledgeIndex:
    model = StaticModel.from_pretrained(config.model_name)
    return KnowledgeIndex.from_paths(
        config.support_path,
        config.gui_map_path,
        config.docs_path,
        model,
    )


def build_cloud(config: Config) -> CloudClient | None:
    if not config.cloud_enabled:
        return None
    return CloudClient(
        config.llm_socket,
        config.llm_model,
        timeout=config.llm_timeout,
    )


def build_pipeline(
    config: Config,
    index: KnowledgeIndex,
    prowl: ProwlClient | None,
) -> SupportPipeline:
    return SupportPipeline(index, build_cloud(config), prowl)


def build_prowl(config: Config) -> ProwlClient | None:
    if config.ryoku_repo_path is None:
        return None
    return ProwlClient(
        config.ryoku_repo_path,
        timeout=config.prowl_timeout,
        result_limit=config.prowl_result_limit,
        executable=config.prowl_executable,
    )
