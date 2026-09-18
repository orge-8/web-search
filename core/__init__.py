"""web-search 插件的核心实现模块。

设计约束（重要）：
  本包内所有模块都是**纯逻辑**，不允许触碰 ``self.ctx``。
  所有与 Host 的交互（LLM、发送、日志、配置）都收敛在 ``plugin.py``。
  这样静态自检 ``check_plugin.py`` 扫描 ``ctx.*`` 调用时不会误判，
  也让核心逻辑可以脱离 MaiBot 环境直接单测。
"""

from .cache import TTLCache
from .engines import EngineChain, build_engine_chain
from .extract import extract_main_text, html_to_text
from .fetcher import FetchOutcome, Fetcher, is_safe_url
from .image_download import ImageDownload, ImageDownloader
from .image_search import BaiduImageEngine, ImageResult, parse_acjson
from .llm_params import (
    KNOWN_TASK_NAMES,
    build_llm_kwargs,
    describe_kwargs,
    explain_missing_model,
    generate_supports_task_name,
    needs_task_list_lookup,
    rejected_model_name,
)
from .models import SearchResult
from .relevance import extract_terms, is_low_relevance, relevance_ratio

__all__ = [
    "TTLCache",
    "EngineChain",
    "build_engine_chain",
    "extract_main_text",
    "html_to_text",
    "FetchOutcome",
    "Fetcher",
    "is_safe_url",
    "ImageDownload",
    "ImageDownloader",
    "BaiduImageEngine",
    "ImageResult",
    "parse_acjson",
    "KNOWN_TASK_NAMES",
    "build_llm_kwargs",
    "describe_kwargs",
    "explain_missing_model",
    "generate_supports_task_name",
    "needs_task_list_lookup",
    "rejected_model_name",
    "SearchResult",
    "extract_terms",
    "is_low_relevance",
    "relevance_ratio",
]
