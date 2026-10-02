"""联网搜索插件（web-search）。

给麦麦一个 ``web_search`` 工具：把用户的问题丢给搜索引擎，抓回结果页正文，
再让 LLM 读完材料后给出带来源引用的答案。另提供 ``fetch_page`` 抓取指定网页、
``search_image`` 搜图并直发聊天，以及 ``/websearch`` 命令用于诊断引擎链路。

设计取舍（相对社区同类插件）：
  * **依赖刻意压到两个**：httpx + beautifulsoup4。不引入 trafilatura /
    readability-lxml / lxml —— 它们需要编译，在真机 Windows 上装不上的概率
    不低，而依赖装不上就等于整个插件不可用。
  * **正文提取自己写**：见 ``core/extract.py`` 的启发式规则。
  * **引擎可降级**：Google 在多数国内网络不可达，所以默认链是
    duckduckgo → bing，Tavily 作为配了 Key 之后的追加项。
  * **LLM 任务名与模型名分开传**：Host 1.2.5 / SDK 2.8.1 起 ``task_name``（任务名）
    与 ``model``（具体模型名）语义拆分。任务名必须显式传——留空会让 Host 用
    ``plugin.<插件ID>`` 当任务名，该任务未配置时会 fallback 到 embedding 模型并持续
    报 400；而把任务名塞进 ``model`` 槽位，则会得到「未找到名为 'utils' 的模型」。
    传参构造见 ``core/llm_params.py``。
"""

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, ClassVar

import httpx
from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import CONFIG_RELOAD_SCOPE_SELF, ToolParamType, ToolParameterInfo

from .core import TTLCache
from .core.engines import EngineChain, build_engine_chain, supported_engine_names
from .core.fetcher import Fetcher
from .core.image_download import ImageDownloader
from .core.image_search import BaiduImageEngine, ImageResult
from .core.llm_params import (
    build_llm_kwargs,
    describe_kwargs,
    explain_missing_model,
    generate_supports_task_name,
    needs_task_list_lookup,
    rejected_model_name,
)
from .core.models import SearchResult
from .core.relevance import describe, is_low_relevance, relevance_ratio

SUPPORTED_CONFIG_VERSION = "0.3.6"  # 与 _manifest.json 的 version 保持同步

# 行为自检标记：真机"看起来部署了却没生效"时（依赖模块命中 sys.modules 缓存），
# 用 /websearch status 显示的这些值即可判断跑的是不是新代码。
BUILD_TAG = "2026-09-17.llm-param-split"
EXTRACTOR_TAG = "heuristic-v1"

#: Host 可用任务名清单的缓存时长。总结每次搜索都调一次 LLM，不加缓存就是每查一次多一次 RPC。
_TASK_NAMES_TTL_SECONDS = 60.0
#: 取清单的超时。Host RPC 无内建超时，诊断类调用卡住会把总结流程一起拖死。
_TASK_NAMES_TIMEOUT_SECONDS = 8.0

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_SUMMARY_PROMPT_TEMPLATE = """你是麦麦的资料整理助手。请根据下面检索到的材料回答用户的问题。

要求：
1. 直接给出结论，不要复述材料标题，不要写"根据材料"这类空话。
2. 关键事实后用 [编号] 标注来源，编号对应材料前的序号。
3. 材料之间如有冲突，说明分歧点，不要凭空调和。
4. 材料不足以回答时，明确说"没查到"，并说明缺什么，不要编造。
5. 用简体中文，控制在 500 字以内，可以分段或用短列表。

今天的日期：{today}
用户的问题：{query}
{extra_focus}
检索材料：
{materials}"""

_DETAIL_PROMPT_TEMPLATE = """你是麦麦的资料整理助手。用户想了解下面这个网页的内容。

要求：
1. 用简体中文概括这个页面的核心内容，说明它讲的是什么。
2. 如果用户有关注点，优先讲关注点相关的内容。
3. 控制在 500 字以内，不要逐段翻译。

今天的日期：{today}
用户关注点：{focus}
页面地址：{url}
页面标题：{title}
页面正文：
{content}"""


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（与插件版本同步）",
        json_schema_extra={"hidden": True, "disabled": True},
    )
    debug: bool = Field(default=False, description="输出调试日志（含每次搜索的引擎尝试过程）")


class SearchSectionConfig(PluginConfigBase):
    """搜索行为配置。"""

    __ui_label__ = "搜索"
    __ui_icon__ = "search"
    __ui_order__ = 1

    order: list[str] = Field(
        default_factory=lambda: ["baidu", "bing", "duckduckgo"],
        description=(
            "搜索引擎优先级顺序，前面的失败才降级到后面的。"
            "可选：baidu / bing / duckduckgo / searxng / tavily。"
            "默认把 baidu 放首位：实测 Bing 对「含数字 + 空格」的中文长查询会退化成"
            "单字匹配（查「洛天依 2026 演唱会」返回汉字「洛」的字典释义），百度则正常"
        ),
    )
    max_results: int = Field(default=6, description="每次搜索保留的结果条数")
    max_attempts: int = Field(default=3, description="最多尝试几个引擎，0 表示全部尝试")
    engine_cooldown: float = Field(
        default=300.0,
        description=(
            "引擎失败后的冷却时长（秒）。冷却期内跳过该引擎，"
            "避免每次搜索都先去等同一个挂掉的引擎；0 表示不启用"
        ),
    )
    timeout: float = Field(default=15.0, description="单个引擎的请求超时（秒）")
    language: str = Field(default="zh-CN", description="搜索语言偏好")
    proxy: str = Field(
        default="",
        description="搜索与抓取走的代理，例如 http://127.0.0.1:7890；留空则读系统环境变量",
    )
    user_agent: str = Field(default=DEFAULT_USER_AGENT, description="请求使用的 User-Agent")


class FetchSectionConfig(PluginConfigBase):
    """网页抓取配置。"""

    __ui_label__ = "抓取"
    __ui_icon__ = "download"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="是否抓取搜索结果页的正文（关闭则只返回搜索摘要）")
    max_pages: int = Field(default=3, description="最多抓取前几条结果的正文")
    timeout: float = Field(default=15.0, description="单页抓取超时（秒）")
    max_bytes_mb: float = Field(default=5.0, description="单页响应体积上限（MB），超出即截断")
    per_page_chars: int = Field(default=4000, description="单页正文送入 LLM 的字符上限")
    total_chars: int = Field(default=12000, description="所有页面正文合计字符上限（控制上下文开销）")
    concurrency: int = Field(default=3, description="并发抓取数，过高易被网站限流")
    allow_private_networks: bool = Field(
        default=False, description="允许抓取内网地址（危险，仅内网调试时开启）"
    )


class SummarizeSectionConfig(PluginConfigBase):
    """LLM 总结配置。"""

    __ui_label__ = "总结"
    __ui_icon__ = "sparkles"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="是否让 LLM 阅读材料后总结（关闭则直接返回材料片段）")
    task: str = Field(
        default="replyer",
        description=(
            "模型任务名（填任务，不是模型 ID），如 replyer / planner / utils。"
            "留空会导致 Host 回退到 embedding 模型并报错"
        ),
    )
    model: str = Field(
        default="",
        description=(
            "可选：具体模型名（需 SDK ≥2.8.1）。留空即用任务默认模型；"
            "填任务名会被自动忽略并告警"
        ),
    )
    temperature: float = Field(default=0.3, description="总结温度")
    max_tokens: int = Field(default=1200, description="总结输出上限 token")
    rpc_timeout_ms: int = Field(
        default=120000,
        description="LLM 调用的 RPC 层超时（毫秒）。Host 默认 30~60 秒，长材料会被截断，故显式放宽",
    )
    max_output_chars: int = Field(default=2000, description="总结结果回传给 LLM 的字符上限")


class CacheSectionConfig(PluginConfigBase):
    """缓存配置。"""

    __ui_label__ = "缓存"
    __ui_icon__ = "database"
    __ui_order__ = 4

    enabled: bool = Field(default=True, description="是否缓存搜索结果与已抓取的网页正文")
    ttl_seconds: int = Field(default=1800, description="缓存有效期（秒）")
    max_entries: int = Field(default=128, description="缓存条目上限")


class EnginesSectionConfig(PluginConfigBase):
    """需要密钥或自建实例的引擎配置。"""

    __ui_label__ = "引擎凭据"
    __ui_icon__ = "key"
    __ui_order__ = 5

    searxng_base_url: str = Field(
        default="", description="SearXNG 实例地址，例如 https://searx.example.com"
    )
    tavily_api_key: str = Field(default="", description="Tavily API Key（敏感信息，勿提交到版本库）")


class ImageSectionConfig(PluginConfigBase):
    """图片搜索与发送配置。"""

    __ui_label__ = "图片"
    __ui_icon__ = "image"
    __ui_order__ = 6

    enabled: bool = Field(default=True, description="是否启用图片搜索工具（关闭后工具返回未启用提示）")
    default_count: int = Field(default=2, description="每次默认发送几张图片")
    max_count: int = Field(default=4, description="单次发送张数上限（LLM 传参会被截断到此值）")
    candidate_pool: int = Field(default=12, description="每次搜索抓取的候选图数量（给死链留余量）")
    min_width: int = Field(default=200, description="剔除宽度过小的图（图标/碎图）；尺寸未知（0）的放行")
    download_timeout: float = Field(default=10.0, description="单张图片下载超时（秒）")
    max_download_bytes_mb: float = Field(
        default=8.0, description="单张图片下载体积上限（MB），超出即放弃该图换下一张"
    )
    max_send_bytes_mb: float = Field(
        default=4.0,
        description=(
            "单张图片发送体积上限（MB）。base64 过 RPC 会膨胀 1.33 倍，"
            "4MB 原图约 5.3MB 传输，超出直接拒图换下一张"
        ),
    )
    allow_private_networks: bool = Field(
        default=False, description="允许下载内网图片地址（危险，仅调试开启）"
    )
    safe_search: bool = Field(
        default=True,
        description=(
            "安全搜索开关（预留位）。百度接口无公开的成人内容过滤参数，"
            "当前版本内容审核依赖 LLM 层措辞，此开关仅在状态页展示"
        ),
    )


class WebSearchConfig(PluginConfigBase):
    """插件配置总入口。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    search: SearchSectionConfig = Field(default_factory=SearchSectionConfig)
    fetch: FetchSectionConfig = Field(default_factory=FetchSectionConfig)
    summarize: SummarizeSectionConfig = Field(default_factory=SummarizeSectionConfig)
    cache: CacheSectionConfig = Field(default_factory=CacheSectionConfig)
    engines: EnginesSectionConfig = Field(default_factory=EnginesSectionConfig)
    image: ImageSectionConfig = Field(default_factory=ImageSectionConfig)


class WebSearchPlugin(MaiBotPlugin):
    """联网搜索插件主类。

    辅助方法一律定义在装饰器组件之前——装饰器只绑定紧邻其后的那个 ``def``，
    中间插入别的函数会静默注册错组件（本地全绿、真机报参数不匹配）。
    """

    config_model: ClassVar[type[PluginConfigBase] | None] = WebSearchConfig

    def get_webui_config_schema(self, **kwargs) -> dict:
        """覆写 SDK 的 WebUI 配置 Schema：做可视化模式的显示层补丁。

        Runner 调这个方法拿配置页 Schema 且异常会被吞掉（变成空 Schema、
        配置页整页空白），所以这里自己兜底：补丁失败就原样返回 SDK 输出。
        """

        schema = super().get_webui_config_schema(**kwargs)
        try:
            return _apply_webui_display_polish(schema)
        except Exception:  # noqa: BLE001 —— 显示补丁失败绝不能让配置页变空白
            logging.getLogger(__name__).exception("修正 WebUI 配置 Schema 失败，回退 SDK 原样输出")
            return schema

    def __init__(self) -> None:
        """初始化插件状态。"""
        super().__init__()
        self._cache: TTLCache | None = None
        self._fetcher: Fetcher | None = None
        self._search_client: httpx.AsyncClient | None = None
        self._chain: EngineChain | None = None
        self._image_engine: BaiduImageEngine | None = None
        self._image_downloader: ImageDownloader | None = None
        # 跨请求图片去重：stream_id -> (指纹集合, 最后活跃时刻)。
        # 「再来几张」场景下百度会返回同一批 top 结果，无此记忆必然重复发图。
        self._image_sent_history: dict[str, tuple[set[str], float]] = {}
        self._last_error: str = ""
        self._started_at: float = 0.0
        # --- LLM 传参自愈状态（见 core/llm_params.py）---
        # 「任务名被填进模型名槽位」属于**配置性**错误，重试不会自愈；正确做法是
        # 调用前纠偏一次，并把被拒的参数在本次会话内停用，别让每个请求都白跑一遍。
        # 这些位不写回 config.toml，且在 on_config_update 里复位，保证用户改完配置即刻生效。
        self._llm_ignore_model: bool = False
        self._llm_ignore_task: bool = False
        self._llm_notice_reported: bool = False
        # Host 可用任务名清单缓存 (monotonic 时刻, 清单)；取失败不写缓存。
        self._llm_task_names_cache: tuple[float, list[str]] | None = None
        # generate 是否支持 task_name（None = 尚未探测）。
        self._llm_supports_task_name: bool | None = None

    # ------------------------------------------------------------------ 辅助

    def _debug(self, message: str, *args: Any) -> None:
        """按配置输出调试日志。"""
        if self.config.plugin.debug:
            self.ctx.logger.info(message, *args)

    def _build_cache(self) -> TTLCache | None:
        """按配置创建缓存实例。"""
        if not self.config.cache.enabled:
            return None
        return TTLCache(
            max_entries=self.config.cache.max_entries,
            ttl_seconds=self.config.cache.ttl_seconds,
        )

    def _build_search_client(self) -> httpx.AsyncClient:
        """创建搜索引擎共用的 httpx 客户端。"""
        search = self.config.search
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(search.timeout, connect=min(10.0, search.timeout)),
            "follow_redirects": True,
            "max_redirects": 5,
            "headers": {
                "User-Agent": search.user_agent or DEFAULT_USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            "trust_env": True,
        }
        proxy = (search.proxy or "").strip()
        if proxy:
            try:
                return httpx.AsyncClient(proxy=proxy, **kwargs)
            except TypeError:  # httpx < 0.28
                return httpx.AsyncClient(proxies=proxy, **kwargs)
        return httpx.AsyncClient(**kwargs)

    def _build_fetcher(self) -> Fetcher:
        """创建网页抓取器。"""
        fetch = self.config.fetch
        return Fetcher(
            timeout=fetch.timeout,
            user_agent=self.config.search.user_agent or DEFAULT_USER_AGENT,
            max_bytes=int(fetch.max_bytes_mb * 1024 * 1024),
            allow_private_networks=fetch.allow_private_networks,
            proxy=self.config.search.proxy,
            cache=self._cache,
        )

    def _build_chain(self, client: httpx.AsyncClient) -> EngineChain:
        """构建引擎降级链。"""
        return build_engine_chain(
            client=client,
            order=list(self.config.search.order or []),
            language=self.config.search.language,
            searxng_base_url=self.config.engines.searxng_base_url,
            tavily_api_key=self.config.engines.tavily_api_key,
            max_attempts=self.config.search.max_attempts,
            cooldown_seconds=self.config.search.engine_cooldown,
        )

    def _build_image_engine(self, client: httpx.AsyncClient) -> BaiduImageEngine:
        """构建图片搜索引擎（与网页引擎共享客户端与代理配置）。"""
        return BaiduImageEngine(client, language=self.config.search.language)

    def _build_image_downloader(self) -> ImageDownloader:
        """构建图片下载器（独立客户端：超时/体积预算与网页抓取不同）。"""
        image = self.config.image
        return ImageDownloader(
            timeout=image.download_timeout,
            user_agent=self.config.search.user_agent or DEFAULT_USER_AGENT,
            max_download_bytes=int(image.max_download_bytes_mb * 1024 * 1024),
            max_send_bytes=int(image.max_send_bytes_mb * 1024 * 1024),
            allow_private_networks=image.allow_private_networks,
            proxy=self.config.search.proxy,
        )

    async def _rebuild_runtime(self) -> None:
        """（重）建缓存、客户端、抓取器、引擎链与图片运行时，旧资源先释放。"""
        await self._release_runtime()

        self._cache = self._build_cache()
        self._search_client = self._build_search_client()
        self._fetcher = self._build_fetcher()
        self._chain = self._build_chain(self._search_client)
        # 图片引擎无独立资源（复用 _search_client），随客户端生命周期走；
        # 下载器持有独立客户端，须纳入释放流程
        self._image_engine = self._build_image_engine(self._search_client)
        self._image_downloader = self._build_image_downloader()

        self._debug(
            "运行时已重建：引擎链=%s 抓取=%s 缓存=%s 图片=%s",
            self._chain.names,
            self.config.fetch.enabled,
            bool(self._cache),
            self.config.image.enabled,
        )

    async def _release_runtime(self) -> None:
        """释放运行时资源。"""
        if self._image_downloader is not None:
            await self._image_downloader.aclose()
            self._image_downloader = None
        if self._fetcher is not None:
            await self._fetcher.aclose()
            self._fetcher = None
        if self._search_client is not None:
            try:
                await self._search_client.aclose()
            except Exception:
                pass
            self._search_client = None
        self._chain = None
        self._image_engine = None
        self._cache = None

    def _ensure_chain(self) -> EngineChain:
        """取引擎链，缺失时惰性重建（保证工具调用不会因状态空转失败）。"""
        if self._chain is None:
            self._search_client = self._build_search_client()
            self._chain = self._build_chain(self._search_client)
        return self._chain

    def _ensure_fetcher(self) -> Fetcher:
        """取抓取器，缺失时惰性重建。"""
        if self._fetcher is None:
            self._fetcher = self._build_fetcher()
        return self._fetcher

    def _ensure_image_engine(self) -> BaiduImageEngine:
        """取图片引擎，缺失时惰性重建。"""
        if self._image_engine is None:
            if self._search_client is None:
                self._search_client = self._build_search_client()
            self._image_engine = self._build_image_engine(self._search_client)
        return self._image_engine

    def _ensure_image_downloader(self) -> ImageDownloader:
        """取图片下载器，缺失时惰性重建。"""
        if self._image_downloader is None:
            self._image_downloader = self._build_image_downloader()
        return self._image_downloader

    async def _fetch_pages(self, results: list[SearchResult]) -> None:
        """并发抓取结果正文，就地写入 ``content`` / ``fetch_error``。

        Args:
            results: 搜索结果列表（按相关度排序）。
        """
        fetch = self.config.fetch
        if not fetch.enabled or not results:
            return

        targets = results[: max(0, fetch.max_pages)]
        # 已自带正文的（例如 tavily）不必再抓
        pending = [item for item in targets if not item.has_content]
        if not pending:
            return

        fetcher = self._ensure_fetcher()
        semaphore = asyncio.Semaphore(max(1, fetch.concurrency))

        async def _one(item: SearchResult) -> None:
            async with semaphore:
                outcome = await fetcher.fetch(item.url)
            if outcome.ok:
                item.content = outcome.text
                if not item.title and outcome.title:
                    item.title = outcome.title
                # 用跟随重定向后的最终地址回写来源：百度的结果是 /link?url=...
                # 跳转链接，直接展示给用户毫无意义，而这一步顺手就解开了。
                if outcome.final_url and outcome.final_url != item.url:
                    item.url = outcome.final_url
            else:
                item.fetch_error = outcome.error
                self._debug("抓取失败 %s：%s", item.url, outcome.error)

        await asyncio.gather(*(_one(item) for item in pending), return_exceptions=True)

    def _build_materials(self, results: list[SearchResult]) -> str:
        """把搜索结果渲染成受总预算约束的材料文本。"""
        budget = max(500, self.config.fetch.total_chars)
        per_page = max(200, self.config.fetch.per_page_chars)

        blocks: list[str] = []
        used = 0
        for index, item in enumerate(results, start=1):
            block = item.to_prompt_block(index, max_chars=per_page)
            if used + len(block) > budget and blocks:
                break
            blocks.append(block)
            used += len(block)

        if not blocks:
            return "(没有可用材料)"
        return "\n\n".join(blocks)

    def _build_sources(self, results: list[SearchResult]) -> str:
        """渲染来源清单。"""
        if not results:
            return ""
        lines = [item.to_source_line(index) for index, item in enumerate(results, start=1)]
        return "\n".join(lines)

    async def _available_task_names(self) -> list[str]:
        """取 Host 可用任务名清单（60s 缓存 + 超时保护），取不到返回空列表。

        只在「疑似把任务名填进了模型名槽位」时才会被调用，正常配置不产生额外 RPC。
        """
        now = time.monotonic()
        cached = self._llm_task_names_cache
        if cached is not None and now - cached[0] < _TASK_NAMES_TTL_SECONDS:
            return cached[1]

        try:
            names = [
                str(name)
                for name in await asyncio.wait_for(
                    self.ctx.llm.get_available_models(),
                    timeout=_TASK_NAMES_TIMEOUT_SECONDS,
                )
            ]
        except Exception as exc:  # 取不到清单只是失去纠偏依据，不影响正常调用
            self._debug("获取可用任务名失败：%s", exc)
            return []

        self._llm_task_names_cache = (now, names)
        return names

    def _llm_param_notice(self, message: str, *args: Any) -> None:
        """LLM 参数类告警：每次会话只警告一次，避免高频群里刷屏。"""
        if self._llm_notice_reported:
            self._debug(message, *args)
            return
        self._llm_notice_reported = True
        self.ctx.logger.warning(message, *args)

    def _resolve_llm_params(self) -> tuple[str, str]:
        """解析本次要传出的 (任务名, 具体模型名)，被停用的参数在此落空。"""
        summarize = self.config.summarize
        task = (summarize.task or "").strip() or "replyer"
        model = (summarize.model or "").strip()
        if self._llm_ignore_task:
            task = ""
        if self._llm_ignore_model:
            model = ""
        return task, model

    async def _precheck_llm_model(self, model: str) -> str:
        """调用前纠偏：``model`` 槽位若填的其实是任务名，就丢掉它。

        判定要**双重确认**——值在内置白名单里 **且** 在 Host 实际任务清单里：
        只用白名单会误伤同名的真模型，只用 Host 清单又无法排除它是真模型名。
        ``needs_task_list_lookup`` 的左侧短路保证默认配置（空值）与真模型名都不产生额外 RPC。
        """
        if not model or not needs_task_list_lookup(model):
            return model

        available = await self._available_task_names()
        if not available or model not in available:
            return model

        self._llm_ignore_model = True
        self._llm_param_notice(
            "配置项 summarize.model 填的是任务名「%s」，已忽略该值；"
            "任务名请填在 summarize.task，summarize.model 留空即可用任务默认模型",
            model,
        )
        return ""

    async def _handle_llm_failure(self, reason: str, kwargs: dict[str, Any]) -> None:
        """LLM 软失败（``success=False``）的诊断与自愈。

        「未找到名为 X 的模型」是**配置性**错误——重试不会自愈，所以第 1 次就把
        *实际传出*的参数、Host 可用任务名清单和修复步骤一起打出来，不等重试累计。
        """
        head = f"LLM 返回失败（{describe_kwargs(kwargs)}）：{reason or '未知原因'}"
        rejected = rejected_model_name(reason)

        if rejected and kwargs.get("model") == rejected and not self._llm_ignore_model:
            self._llm_ignore_model = True
            available = await self._available_task_names()
            self.ctx.logger.error(
                "%s\n已停用 summarize.model（本次运行内不再传该参数）。修复建议：%s",
                head,
                explain_missing_model(rejected, available),
            )
            return

        if rejected and kwargs.get("task_name") == rejected and not self._llm_ignore_task:
            self._llm_ignore_task = True
            available = await self._available_task_names()
            self.ctx.logger.error(
                "%s\n已停用 summarize.task（回落到 Host 默认任务）。修复建议：%s",
                head,
                explain_missing_model(rejected, available),
            )
            return

        self.ctx.logger.warning("%s", head)

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成总结，失败返回空字符串。

        任务名与模型名分走 ``task_name`` / ``model`` 两个槽位（Host 1.2.5 起语义拆分）：
        把任务名塞进 ``model`` 会得到「未找到名为 'utils' 的模型」。传参构造在
        ``core/llm_params``，这里只管纠偏、日志与降级。
        """
        summarize = self.config.summarize
        task, model = self._resolve_llm_params()
        model = await self._precheck_llm_model(model)

        if self._llm_supports_task_name is None:
            self._llm_supports_task_name = generate_supports_task_name(self.ctx.llm.generate)

        kwargs, notices = build_llm_kwargs(
            task=task,
            model=model,
            supports_task_name=self._llm_supports_task_name,
            temperature=summarize.temperature,
            max_tokens=summarize.max_tokens,
            timeout_ms=max(10000, int(summarize.rpc_timeout_ms)),
        )
        for notice in notices:
            self._llm_param_notice("%s", notice)
        # 打**实际传出**的参数，而不是读配置：两者之间隔着纠偏与默认值注入。
        self._debug("LLM 调用参数：%s", describe_kwargs(kwargs))

        try:
            result = await asyncio.wait_for(
                self.ctx.llm.generate(prompt, **kwargs),
                timeout=max(10.0, summarize.rpc_timeout_ms / 1000.0) + 5.0,
            )
        except asyncio.TimeoutError:
            self._last_error = "LLM 总结超时"
            self.ctx.logger.warning("LLM 总结超时（%s）", describe_kwargs(kwargs))
            return ""
        except Exception as exc:
            self._last_error = f"LLM 调用异常：{exc}"
            self.ctx.logger.warning("LLM 调用失败（%s）：%s", describe_kwargs(kwargs), exc)
            return ""

        if not isinstance(result, dict) or not result.get("success"):
            reason = ""
            if isinstance(result, dict):
                reason = str(result.get("error") or result.get("message") or "")
            self._last_error = f"LLM 返回失败：{reason or '未知原因'}"
            await self._handle_llm_failure(reason, kwargs)
            return ""

        text = str(result.get("response") or "").strip()
        # 净化：剥控制字符（防 LLM 输出被页面内容诱导夹带角色/系统标记注入聊天）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        limit = max(200, self.config.summarize.max_output_chars)
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    def _fallback_answer(self, results: list[SearchResult]) -> str:
        """LLM 不可用时的兜底输出：直接给摘要片段。"""
        if not results:
            return ""
        lines = ["(未能调用模型总结，以下是检索到的原始片段)"]
        for index, item in enumerate(results[:5], start=1):
            body = (item.content or item.snippet or "").strip()
            if len(body) > 300:
                body = body[:300].rstrip() + "…"
            lines.append(f"[{index}] {item.title or '(无标题)'}\n{body or '(无摘要)'}")
        return "\n\n".join(lines)

    def _render_result(
        self,
        results: list[SearchResult],
        failures: list[tuple[str, str]],
        answer: str,
    ) -> str:
        """拼装最终回传文本：答案 + 来源 + 失败提示。"""
        parts: list[str] = []
        if answer:
            parts.append(answer)
        if results:
            sources = self._build_sources(results[:5])
            if sources:
                parts.append("来源：\n" + sources)
        if not answer and not results:
            parts.append("没有查到可用结果。")

        if failures:
            detail = "；".join(f"{name}: {reason}" for name, reason in failures)
            parts.append(f"（部分引擎不可用 → {detail}）")
        return "\n\n".join(part for part in parts if part).strip()

    async def _run_search(
        self,
        query: str,
        limit: int,
        fetch_pages: bool,
        focus: str,
    ) -> str:
        """完整搜索链路：搜索 → 抓取 → 总结。"""
        chain = self._ensure_chain()
        self._last_error = ""

        results, failures = await chain.search(query, limit)
        if not results:
            self._last_error = "所有引擎均未返回结果"
            reason = "；".join(f"{name}: {reason}" for name, reason in failures) or "未知原因"
            return f"没搜到「{query}」的相关结果。引擎情况 → {reason}"

        if fetch_pages:
            await self._fetch_pages(results)

        # 抓完正文再评估相关性：此时 content 参与匹配，判断更准
        ratio = relevance_ratio(query, results)

        answer = ""
        if self.config.summarize.enabled:
            materials = self._build_materials(results)
            extra_focus = f"额外关注：{focus}\n" if focus.strip() else ""
            prompt = _SUMMARY_PROMPT_TEMPLATE.format(
                today=datetime.now().strftime("%Y-%m-%d"),
                query=query,
                extra_focus=extra_focus,
                materials=materials,
            )
            answer = await self._call_llm(prompt)
        if not answer:
            answer = self._fallback_answer(results)

        rendered = self._render_result(results, failures, answer)
        if is_low_relevance(ratio):
            rendered = f"{rendered}\n\n{self._low_relevance_hint(query, ratio)}"
        return rendered

    def _low_relevance_hint(self, query: str, ratio: float) -> str:
        """结果与查询明显对不上时，给 LLM 一段可执行的改查建议。

        这段提示是必须的：真机日志显示，如果只返回"没找到"而不给方向，
        LLM 会不停换说法重搜，连续空转 9 轮、累计 200+ 秒工具执行时间。
        把"搜索引擎把长查询拆散了"这个事实直接讲明，它才会转向别的策略。
        """
        return (
            f"检索质量提示：{describe(ratio)}，搜索引擎很可能把长查询拆散成了单字匹配。"
            f"请不要用原句重复检索。建议改为：\n"
            f"1) 只保留核心实体重搜，例如把「{query[:24]}」缩短成一个专有名词；\n"
            f"2) 或改用 fetch_page 直接抓取已知官方页面（官网 / 百科 / 官方账号空间页）；\n"
            f"3) 若换过两次仍无结果，请如实告诉用户没查到，并说明缺什么信息。"
        )

    # ------------------------------------------------------ 图片搜索辅助

    #: 已发图片记忆的保留时长（秒）与单会话指纹上限
    _IMG_HISTORY_TTL = 1800.0
    _IMG_HISTORY_MAX = 200

    #: 用户原话里的数量词（「发一张」「来两张」「3张」）；解析不出返回 0
    _USER_COUNT_RE = re.compile(r"([一两二三四五六七八九十\d]+)\s*张")
    _CN_DIGIT = {
        "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9,
    }

    @classmethod
    def _parse_user_count(cls, text: str) -> int:
        """从用户原话解析明确张数。

        「发一张」→1、「来两张」→2、「3张」→3；含「十」做简单合成（十一→11）。
        解析不出（如「来几张」「发点」）返回 0，表示无明确数量、用默认值。
        真机教训：Planner 收到「发一张」却自作主张传 count=3——LLM 传参
        不可信，必须由代码从原话兜底解析。
        """
        match = cls._USER_COUNT_RE.search(text or "")
        if not match:
            return 0
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if "十" in token:
            parts = token.split("十")
            tens = cls._CN_DIGIT.get(parts[0], 1) if parts[0] else 1
            ones = cls._CN_DIGIT.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        return cls._CN_DIGIT.get(token, 0)

    def _image_history_for(self, stream_id: str) -> set[str]:
        """取（并惰性清理）某个会话的已发图片指纹集合。"""
        now = time.monotonic()
        stale = [
            sid for sid, (_fp, ts) in self._image_sent_history.items()
            if now - ts > self._IMG_HISTORY_TTL
        ]
        for sid in stale:
            del self._image_sent_history[sid]
        entry = self._image_sent_history.get(stream_id)
        if entry is None:
            fingerprints: set[str] = set()
            self._image_sent_history[stream_id] = (fingerprints, now)
            return fingerprints
        fingerprints, _ = entry
        self._image_sent_history[stream_id] = (fingerprints, now)
        # 防爆炸：超上限时无差别清空（正常使用到不了这个量级）
        if len(fingerprints) > self._IMG_HISTORY_MAX:
            fingerprints.clear()
        return fingerprints

    @staticmethod
    def _image_fingerprint(item: ImageResult) -> tuple[str, ...]:
        """提取一条候选的 URL 指纹（thumb/hover/来源图任一命中即视为重复）。"""
        return tuple(
            url for url in (item.thumb_url, item.hover_url, item.image_url) if url
        )

    def _resolve_image_count(self, count: int, user_request: str = "") -> int:
        """把张数收敛到配置区间 [1, max_count]。

        优先级：用户原话里的明确数量词 > LLM 传的 count > 配置默认值。
        用户说「发一张」时 LLM 传什么都以 1 为准（真机教训）。
        """
        image = self.config.image
        default_count = max(1, image.default_count)
        max_count = max(1, image.max_count)
        user_said = self._parse_user_count(user_request)
        if user_said > 0:
            requested = user_said
        else:
            try:
                requested = int(count or 0)
            except (TypeError, ValueError):
                requested = 0
            if requested <= 0:
                requested = default_count
        return max(1, min(requested, max_count))

    def _filter_image_candidates(
        self,
        results: list[ImageResult],
        *,
        wanted: int,
        sent_history: set[str] | None = None,
    ) -> list[ImageResult]:
        """候选过滤：单请求内 thumb_url 去重 + 剔除过小图 + 跳过本会话近期已发。"""
        min_width = max(0, self.config.image.min_width)
        seen: set[str] = set()
        pool: list[ImageResult] = []
        for item in results:
            if not item.thumb_url or item.thumb_url in seen:
                continue
            if min_width and item.width and item.width < min_width:
                continue
            # 跨请求去重：任一 URL 指纹在历史里就跳过（「再来几张」场景）
            if sent_history:
                fingerprints = self._image_fingerprint(item)
                if fingerprints and any(fp in sent_history for fp in fingerprints):
                    continue
            seen.add(item.thumb_url)
            pool.append(item)
            # 超采样封顶：给死链留余量即可，不必全量下载校验
            if len(pool) >= max(wanted * 3 + 6, self.config.image.candidate_pool):
                break
        return pool

    async def _send_image(self, data: bytes, stream_id: str) -> tuple[bool, str]:
        """把图片字节以纯 base64 发进聊天（带 data: 前缀实测会发送失败）。"""
        import base64

        try:
            sent = await self.ctx.send.image(
                base64.b64encode(data).decode("ascii"), stream_id
            )
        except Exception as exc:
            self.ctx.logger.warning("send.image 调用异常：%s", exc)
            return False, f"发送异常：{exc}"
        if not sent:
            return False, "宿主返回发送失败"
        return True, ""

    async def _run_image_search(
        self, query: str, count: int, stream_id: str, user_request: str = ""
    ) -> str:
        """图片搜索完整链路：搜索 → 过滤 → 下载（降级链）→ 发送 → 渲染。"""
        image = self.config.image
        if not image.enabled:
            return "图片搜索未启用（插件配置 image.enabled=false）。"

        wanted = self._resolve_image_count(count, user_request)
        engine = self._ensure_image_engine()
        downloader = self._ensure_image_downloader()

        try:
            raw = await engine.search(
                query, max(image.candidate_pool, wanted * 3 + 6)
            )
        except Exception as exc:  # EngineError 已带中文原因
            self._last_error = f"图片搜索失败：{exc}"
            self.ctx.logger.warning("图片搜索失败：%s", exc)
            return f"图片搜索失败：{exc}。可以换个关键词重试。"

        sent_history = self._image_history_for(stream_id)
        pool = self._filter_image_candidates(
            raw, wanted=wanted, sent_history=sent_history
        )
        dedup_skipped = len(raw) - len(pool) if sent_history else 0
        if not pool:
            return (
                f"搜到了关于「{query}」的图片结果，但没有可用的下载地址"
                f"{'（近期已发过的已自动跳过）' if sent_history else ''}。"
                "可以换个关键词重试。"
            )

        skip_send = not stream_id
        sent: list[ImageResult] = []
        last_error = ""
        skipped = 0
        # 总尝试上限防全灭循环：每个候选最多消耗 thumb+hover 两次下载机会
        attempt_cap = wanted * 2 + 4

        for candidate in pool[:attempt_cap]:
            if len(sent) >= wanted:
                break
            data: bytes | None = None
            for url in (candidate.thumb_url, candidate.hover_url):
                if not url:
                    continue
                outcome = await downloader.download(url)
                if outcome.ok:
                    data = outcome.data
                    break
                last_error = outcome.error
                self._debug("图片下载失败 %s：%s", url, outcome.error)

            if data is None:
                skipped += 1
                continue

            # 内容 hash 兜底：不同 URL 指向同一张图时（百度 thumb 常见），仍不重发
            # 注意 sent_history 是「集合可能为空」——空集合也必须记录，否则第一次
            # 发送不入历史，第二次就无据可查（falsy 陷阱，用 is not None 判断）
            if sent_history is not None:
                content_hash = f"sha:{hashlib.sha256(data).hexdigest()}"
                if content_hash in sent_history:
                    self._debug("图片内容与近期已发重复，跳过 %s", candidate.thumb_url)
                    skipped += 1
                    continue
                sent_history.add(content_hash)
                for fp in self._image_fingerprint(candidate):
                    sent_history.add(fp)

            if skip_send:
                # 无聊天上下文：只收集描述，不发图
                sent.append(candidate)
                continue

            ok, reason = await self._send_image(data, stream_id)
            if ok:
                sent.append(candidate)
            else:
                last_error = reason
                skipped += 1

        return self._render_image_reply(
            sent=sent,
            skipped=skipped,
            pool=pool,
            query=query,
            error="" if sent else last_error,
            skip_send=skip_send,
            dedup_skipped=dedup_skipped,
        )

    def _render_image_reply(
        self,
        *,
        sent: list[ImageResult],
        skipped: int,
        pool: list[ImageResult],
        query: str,
        error: str,
        skip_send: bool,
        dedup_skipped: int = 0,
    ) -> str:
        """渲染给 LLM 的返回文案：图片已直发聊天，LLM 只需口头介绍。"""
        if not sent:
            hint = error or "未知原因"
            links = "；".join(
                item.source_url for item in pool[:2] if item.source_url
            )
            return (
                f"搜到了 {len(pool)} 张「{query}」的图，但全部下载或发送失败（原因：{hint}）。"
                f"{('可以把来源页链接告诉用户：' + links) if links else ''}"
            )

        lines = [f"已向当前聊天发送 {len(sent)} 张关于「{query}」的图片："]
        lines.extend(item.to_prompt_line(i) for i, item in enumerate(sent, start=1))
        lines.append("图片已直接出现在聊天里，请根据上面的描述向用户介绍，无需再发链接。")
        if dedup_skipped > 0:
            lines.append(
                f"（已自动避开 {dedup_skipped} 张近期发过的图，均为新图）"
            )
        if skip_send:
            lines.append("（当前无聊天上下文，图片未实际发送）")
        if skipped:
            lines.append(f"（{skipped} 张因下载失败/尺寸不符被跳过）")
        return "\n".join(lines)

    # -------------------------------------------------------------- 生命周期

    async def on_load(self) -> None:
        """插件加载：建立运行时资源。"""
        self._started_at = time.time()
        await self._rebuild_runtime()
        self.ctx.logger.info(
            "联网搜索插件已加载 v%s（build=%s）引擎链=%s 抓取=%s 图片=%s 总结任务=%s",
            SUPPORTED_CONFIG_VERSION,
            BUILD_TAG,
            self._chain.names if self._chain else [],
            self.config.fetch.enabled,
            self.config.image.enabled,
            self.config.summarize.task,
        )

    async def on_unload(self) -> None:
        """插件卸载：释放所有连接资源。"""
        await self._release_runtime()
        self.ctx.logger.info("联网搜索插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        """配置热重载：重建运行时（引擎链、超时、代理都可能变化）。"""
        del config_data
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        # 自愈状态必须复位：用户改完配置要立刻生效，否则被停用的参数会一直停用。
        self._llm_ignore_model = False
        self._llm_ignore_task = False
        self._llm_notice_reported = False
        self._llm_task_names_cache = None
        self._llm_supports_task_name = None
        await self._rebuild_runtime()
        self.ctx.logger.info("联网搜索插件配置已更新：version=%s", version)

    # ------------------------------------------------------------------ 组件

    @Tool(
        "web_search",
        description=(
            "联网搜索。当需要获取实时信息、最新事件、你不掌握的事实，或用户明确要求"
            "上网查一下时使用。会搜索多个来源、抓取网页正文并汇总成带来源引用的答案。"
            "注意：这需要消耗数秒到数十秒，不要对已知常识或闲聊使用。"
        ),
        detailed_description=(
            "参数说明：\n"
            "- query：string，必填。搜索关键词，用自然语言描述想要查什么，"
            "例如「2026 年诺贝尔物理学奖得主」。不要传 URL 或命令。\n"
            "- max_results：integer，可选。保留的结果条数，默认读插件配置（通常 6）。\n"
            "- fetch_pages：boolean，可选。是否进一步抓取网页正文再总结，默认 true；"
            "只想要快速标题列表时传 false。\n"
            "- focus：string，可选。希望材料重点覆盖的方面，例如「价格和发布时间」。\n"
            "返回：一段中文答案，末尾附来源链接清单。"
        ),
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="搜索关键词（自然语言描述）",
                required=True,
            ),
            ToolParameterInfo(
                name="max_results",
                param_type=ToolParamType.INTEGER,
                description="保留的结果条数，0 表示使用插件配置默认值",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="fetch_pages",
                param_type=ToolParamType.BOOLEAN,
                description="是否抓取正文后再总结，默认 true",
                required=False,
                default=True,
            ),
            ToolParameterInfo(
                name="focus",
                param_type=ToolParamType.STRING,
                description="总结时希望重点覆盖的方面，可为空",
                required=False,
                default="",
            ),
        ],
    )
    async def tool_web_search(
        self,
        query: str = "",
        max_results: int = 0,
        fetch_pages: bool = True,
        focus: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行联网搜索。"""
        del kwargs
        keyword = (query or "").strip()
        if not keyword:
            return {"content": "搜索失败：query 参数为空，请提供要查询的关键词。"}

        default_limit = max(1, self.config.search.max_results)
        try:
            limit = int(max_results) if int(max_results or 0) > 0 else default_limit
        except (TypeError, ValueError):
            limit = default_limit
        limit = max(1, min(limit, 10))

        try:
            content = await self._run_search(
                keyword, limit, bool(fetch_pages), str(focus or "")
            )
        except Exception as exc:  # 工具层绝不把堆栈抛给 LLM
            self.ctx.logger.exception("web_search 执行异常")
            return {"content": f"搜索过程中出错：{exc}。可以稍后重试，或改用更具体的关键词。"}

        return {"content": content}

    @Tool(
        "fetch_page",
        description=(
            "抓取指定网页的正文并总结。当用户直接给出链接，或需要查看某个已知页面的"
            "具体内容（而不是搜索）时使用。长页面支持分段读取。"
        ),
        detailed_description=(
            "参数说明：\n"
            "- url：string，必填。要抓取的 http/https 地址。\n"
            "- start_char：integer，可选。从正文第几个字符开始读取，默认 0。\n"
            "- end_char：integer，可选。读到第几个字符为止，-1 表示不限制。\n"
            "返回：该页面的中文内容概括；如需原文片段可按字符区间分段调用。"
        ),
        parameters=[
            ToolParameterInfo(
                name="url",
                param_type=ToolParamType.STRING,
                description="要抓取的网页地址",
                required=True,
            ),
            ToolParameterInfo(
                name="start_char",
                param_type=ToolParamType.INTEGER,
                description="正文起始字符位置，默认 0",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="end_char",
                param_type=ToolParamType.INTEGER,
                description="正文结束字符位置，-1 表示不限制",
                required=False,
                default=-1,
            ),
        ],
    )
    async def tool_fetch_page(
        self,
        url: str = "",
        start_char: int = 0,
        end_char: int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """抓取单个网页并总结。"""
        del kwargs
        target = (url or "").strip()
        if not target:
            return {"content": "抓取失败：url 参数为空。"}

        try:
            fetcher = self._ensure_fetcher()
            outcome = await fetcher.fetch(target)
        except Exception as exc:
            self.ctx.logger.exception("fetch_page 执行异常")
            return {"content": f"抓取过程中出错：{exc}"}

        if not outcome.ok:
            return {
                "content": (
                    f"抓取失败：{outcome.error}。"
                    "如果该地址需要登录或被反爬拦截，可以换个来源，或改用 web_search 搜索。"
                )
            }

        text = outcome.text or ""
        try:
            begin = max(0, int(start_char or 0))
        except (TypeError, ValueError):
            begin = 0
        try:
            finish = int(end_char if end_char is not None else -1)
        except (TypeError, ValueError):
            finish = -1

        window = text[begin:] if finish < 0 else text[begin:finish]
        total = len(text)
        window = window.strip()
        if not window:
            return {
                "content": (
                    f"抓到了 {outcome.final_url}，但正文为空"
                    f"（总长度 {total} 字符，当前区间 {begin}~{finish}）。"
                    "该页面可能是纯脚本渲染，建议换一个来源。"
                )
            }

        focus = "" if finish < 0 else f"第 {begin}~{finish} 字符区间"
        answer = ""
        if self.config.summarize.enabled:
            prompt = _DETAIL_PROMPT_TEMPLATE.format(
                today=datetime.now().strftime("%Y-%m-%d"),
                focus=focus or "（无特定关注点）",
                url=outcome.final_url or target,
                title=outcome.title or "(无标题)",
                content=window[: self.config.fetch.per_page_chars * 2],
            )
            answer = await self._call_llm(prompt)
        if not answer:
            answer = window[:1000]

        suffix = f"\n\n（原文总长 {total} 字符，本次读取 {begin}~{begin + len(window)}）"
        return {"content": f"{answer}{suffix}"}

    @Tool(
        "search_image",
        description=(
            "搜索图片并直接发送到当前聊天。当用户想看某事物的图片、表情包、截图或"
            "视觉参考，或明确要求「发张图」时使用。图片会直接出现在聊天里，返回内容"
            "包含每张图的描述与来源，可据此口头介绍。需数秒，勿对闲聊滥用。"
        ),
        detailed_description=(
            "参数说明：\n"
            "- query：string，必填。要搜的图片关键词，如「麦麦 表情包」「洛天依 舞台照」。\n"
            "- count：integer，可选。发送几张，0 表示用默认值（通常 2），上限 4。"
            "【重要】用户明确说了数量（如「发一张」「来两张」）时必须按用户说的传，"
            "不要自行加大；用户没说数量时传 0。\n"
            "- user_request：string，可选。触发本次搜索的【用户原话】（逐字引用，不要改写），"
            "如「发一张乐正绫的图片」。插件会从原话里解析数量词并以它为准——"
            "这是防止你传错 count 的保险，务必尽量带上。\n"
            "返回：已发送图片的数量与每张的描述+来源页链接。图片已直接发进聊天，"
            "你不需要再发链接，根据描述介绍即可；部分图片下载失败时会自动换下一张。"
        ),
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="图片搜索关键词",
                required=True,
            ),
            ToolParameterInfo(
                name="count",
                param_type=ToolParamType.INTEGER,
                description="发送张数，0=默认，上限 4；用户明确说了数量时必须照传",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="user_request",
                param_type=ToolParamType.STRING,
                description="触发搜索的用户原话（逐字引用），用于解析用户想要的张数",
                required=False,
                default="",
            ),
        ],
    )
    async def tool_search_image(
        self,
        query: str = "",
        count: int = 0,
        user_request: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """搜索图片、下载并发送到当前聊天。"""
        keyword = (query or "").strip()
        if not keyword:
            return {"content": "图片搜索失败：query 参数为空，请提供要搜的图片关键词。"}

        stream_id = str(kwargs.get("stream_id") or "")
        try:
            content = await self._run_image_search(
                keyword, int(count or 0), stream_id, user_request
            )
        except Exception as exc:  # 工具层绝不把堆栈抛给 LLM
            self.ctx.logger.exception("search_image 执行异常")
            return {"content": f"图片搜索过程中出错：{exc}。可以稍后重试或换个关键词。"}
        return {"content": content}

    @Command(
        "websearch",
        description="联网搜索插件诊断：查看引擎链路与缓存状态，或实测一次搜索",
        pattern=(
            r"(?<!\S)/?websearch"
            r"(?:\s+(?P<action>status|test|clear))?"
            r"(?:\s+(?P<arg>.+?))?\s*$"
        ),
        aliases=["联网搜索"],
    )
    async def cmd_websearch(
        self, matched_groups: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[bool, str, bool]:
        """诊断命令入口。

        返回值不会自动发到群里，必须显式 ``ctx.send.text``。
        """
        groups = matched_groups or {}
        action = str(groups.get("action") or "status").strip().lower()
        arg = str(groups.get("arg") or "").strip()
        stream_id = str(kwargs.get("stream_id") or "")

        if action == "test":
            reply = await self._cmd_test(arg)
        elif action == "clear":
            reply = self._cmd_clear()
        else:
            reply = await self._cmd_status()

        if stream_id:
            await self.ctx.send.text(reply, stream_id)
        return True, reply, True

    # -------------------------------------------------------- 命令内部实现

    async def _cmd_status(self) -> str:
        """渲染插件状态。

        会顺带拉一次宿主可用的模型任务名——总结任务名配错是本插件最容易踩的坑
        （配错会回退到 embedding 模型并持续报 400；把任务名填进模型名槽位则会
        报「未找到名为 'xxx' 的模型」），把可用值直接列出来最省事。
        """
        chain = self._ensure_chain()
        cache_stats = self._cache.stats() if self._cache else None
        uptime = int(time.time() - self._started_at) if self._started_at else 0

        # 诊断要的是此刻的真相，所以绕过缓存直接拉（内部仍有超时保护）。
        self._llm_task_names_cache = None
        available_models = await self._available_task_names()

        lines = [
            f"联网搜索插件 v{SUPPORTED_CONFIG_VERSION}",
            f"行为自检：build={BUILD_TAG} / 提取器={EXTRACTOR_TAG}",
            f"引擎链：{' → '.join(chain.names) if chain.names else '(空)'}"
            f"（最多尝试 {chain.max_attempts} 个）",
            f"抓取正文：{'开' if self.config.fetch.enabled else '关'}"
            f"（前 {self.config.fetch.max_pages} 条，每页 {self.config.fetch.per_page_chars} 字）",
            f"总结任务：{self.config.summarize.task or '(空，会回退到 embedding！)'}"
            f" / 具体模型：{self.config.summarize.model or '(跟随任务默认)'}"
            f"（{'开' if self.config.summarize.enabled else '关'}）",
            f"缓存：{'开' if cache_stats else '关'}",
            f"图片搜索：{'开' if self.config.image.enabled else '关'}"
            f"（默认 {self.config.image.default_count} 张/次，上限 {self.config.image.max_count}）",
            f"内网抓取：{'允许' if self.config.fetch.allow_private_networks else '已禁止（SSRF 防护）'}",
            f"代理：{self.config.search.proxy or '(跟随系统环境变量)'}",
            f"Tavily Key：{'已配置' if self.config.engines.tavily_api_key else '未配置'}",
            f"SearXNG：{self.config.engines.searxng_base_url or '未配置'}",
        ]
        snapshot = getattr(chain, "cooldown_snapshot", None)
        cooling = snapshot() if callable(snapshot) else {}
        if cooling:
            lines.append(
                "引擎冷却中："
                + "，".join(f"{name}（剩 {secs:.0f}s）" for name, secs in cooling.items())
            )

        if cache_stats:
            lines.append(
                "缓存统计：条目 {entries}/{max_entries}，命中 {hits}，未命中 {misses}，"
                "淘汰 {evictions}".format(**cache_stats)
            )
        if uptime:
            lines.append(f"已运行：{uptime} 秒")
        if self._last_error:
            lines.append(f"最近一次错误：{self._last_error}")

        if available_models:
            lines.append("可用模型任务名：" + " / ".join(available_models[:15]))
            task_name = (self.config.summarize.task or "").strip()
            if task_name and task_name not in available_models:
                lines.append(
                    f"注意：总结任务「{task_name}」不在上面的列表里，"
                    "请从列表中选一个，否则总结会失败"
                )
            model_name = (self.config.summarize.model or "").strip()
            if model_name and model_name in available_models:
                lines.append(
                    f"注意：summarize.model 填的「{model_name}」是任务名而非模型名，"
                    "调用时会被自动忽略；任务名请填 summarize.task"
                )
        else:
            lines.append(
                "可用模型任务名：(未取到) 若总结持续失败，请到 WebUI 确认模型列表非空"
                "——先保存提供商，再添加模型，最后把任务指向具体模型"
            )

        stopped = [
            name
            for name, flag in (
                ("summarize.model", self._llm_ignore_model),
                ("summarize.task", self._llm_ignore_task),
            )
            if flag
        ]
        if stopped:
            lines.append(
                "已停用的 LLM 参数：" + " / ".join(stopped)
                + "（本次运行内不再传出；改完配置重载插件即恢复）"
            )

        lines.append("可用引擎名：" + " / ".join(supported_engine_names()))
        return "\n".join(lines)

    def _cmd_clear(self) -> str:
        """清空缓存。"""
        if self._cache is None:
            return "缓存未启用，无需清理。"
        removed = self._cache.clear()
        return f"已清空搜索与抓取缓存（{removed} 条）。"

    async def _cmd_test(self, keyword: str) -> str:
        """实测搜索链路（不调用 LLM），逐个引擎汇报结果并给出配置建议。

        刻意绕过引擎链的冷却状态、直接遍历全部引擎：诊断要看的是每个引擎的
        真实状况，而不是"当前会用到哪几个"。所以本命令不读也不写冷却。
        """
        if not keyword:
            return "用法：/websearch test <关键词>，例如 /websearch test 麦麦"

        chain = self._ensure_chain()
        lines = [f"引擎链路实测：{keyword}"]
        ok_engines: list[str] = []
        bad_engines: list[str] = []

        for engine in chain.engines[: chain.max_attempts]:
            started = time.monotonic()
            try:
                results = await engine.search(keyword, 3)
                elapsed = (time.monotonic() - started) * 1000
                lines.append(f"[成功] {engine.name}：{len(results)} 条 / {elapsed:.0f} ms")
                ok_engines.append(engine.name)
                for item in results[:2]:
                    lines.append(f"    · {(item.title or '(无标题)')[:40]}")
            except Exception as exc:
                elapsed = (time.monotonic() - started) * 1000
                lines.append(f"[失败] {engine.name}：{exc}（{elapsed:.0f} ms）")
                bad_engines.append(engine.name)

        if self.config.fetch.enabled:
            fetcher = self._ensure_fetcher()
            probe = "https://example.com/"
            outcome = await fetcher.fetch(probe)
            lines.append(
                f"[抓取] {probe} → 成功，正文 {len(outcome.text)} 字符"
                if outcome.ok
                else f"[抓取] {probe} → 失败：{outcome.error}"
            )

        # 直接给出下一步动作，省去"看到失败原因后还要自己推算怎么配"这一步
        lines.append("")
        configured = list(self.config.search.order or [])
        if ok_engines and bad_engines:
            lines.append(
                f"可用引擎：{' / '.join(ok_engines)}；不可用：{' / '.join(bad_engines)}。"
                f"建议 search.order = {ok_engines}"
            )
        elif ok_engines:
            lines.append(f"全部可用，保持 search.order = {ok_engines} 即可。")
        else:
            lines.append(
                "所有引擎都不通。若本机需要代理才能出网，请把代理地址填到 search.proxy；"
                "或配置 engines.tavily_api_key 后把 tavily 加进 search.order。"
            )

        if configured and ok_engines and configured[0] not in ok_engines:
            lines.append(
                f"注意：当前 order 首位「{configured[0]}」实测不可用，"
                "每次搜索都会先等它超时一次（失败后 5 分钟内会被自动跳过）。"
            )
        return "\n".join(lines)


# ================================================================ WebUI 显示层补丁
#
# 可视化模式的 FieldRenderer（dashboard/src/routes/plugin-config.tsx:163-361，
# 1.3.1 布局）按 ui_type 渲染控件时只输出 label / hint / placeholder，
# **从不渲染 description**；本插件字段的填法说明都写在 description 里，
# 不搬进 hint，用户在配置页上一个字都看不到（源代码模式才见得到）。
# 这里只改展示元数据，不碰任何配置键与校验语义。

#: 默认收起的 section：标题自带「可选 / 默认关闭」的功能节。收起后标题
#: 与说明仍可见，点开即可配置，避免整页卡片全开淹没常用配置。
_WEBUI_COLLAPSED_SECTIONS: frozenset = frozenset({})

#: 手工指定的 section 标题（键 = section 名）；仅在 SDK 输出标题等于
#: 节名（未配置 __ui_label__）时采用。
_WEBUI_SECTION_TITLES: dict = {}

#: 手工指定的字段 label（键 = 字段名）；仅在自动推导不可用时采用。
_WEBUI_LABEL_OVERRIDES: dict = {
    "tavily_api_key": "Tavily API Key",
    "model": "总结模型",
}


#: 推导 label 用的中文分隔符（取最靠左的一个）
_WEBUI_CJK_SEPS = "。！？；，：（(、"


def _webui_label_from_description(description: str) -> str:
    """从中文 description 里取第一小句当显示标题（取不到返回空串）。

    在中英文标点里找**最靠左**的分隔符，取它前面的短语——通常是字段
    本身的名字；超长时截到 12 字符并避免把英文单词截一半。
    """
    text = (description or "").strip()
    if not text:
        return ""
    cut = len(text)
    for sep in _WEBUI_CJK_SEPS:
        idx = text.find(sep)
        if 0 < idx < cut:
            cut = idx
    text = text[:cut].strip()
    if len(text) > 12:
        text = text[:12]
        if " " in text[4:]:
            text = text[: text.rfind(" ")].rstrip() or text
    while text and text[-1] in "（(\"“：:，,；;、":
        text = text[:-1].rstrip()
    return text if len(text) >= 2 else ""


def _apply_webui_display_polish(schema: dict) -> dict:
    """把 description 抄进 hint、补中文 label / 节标题、收起可选功能节。"""
    if not isinstance(schema, dict):
        return schema
    sections = schema.get("sections")
    if not isinstance(sections, dict):
        return schema
    for name, section in sections.items():
        if not isinstance(section, dict):
            continue
        if name in _WEBUI_COLLAPSED_SECTIONS:
            section["collapsed"] = True
        title = section.get("title")
        if not title or title == name:
            new_title = _WEBUI_SECTION_TITLES.get(name) or _webui_label_from_description(
                section.get("description") or ""
            )
            if new_title and new_title != name:
                section["title"] = new_title
        for fname, field in (section.get("fields") or {}).items():
            if not isinstance(field, dict):
                continue
            if fname == "config_version":
                field["hidden"] = True  # 插件自维护字段：可视化模式不渲染，源代码模式仍可见
            if not field.get("hint") and field.get("description"):
                field["hint"] = field["description"]
            label = field.get("label")
            if (not label or label == fname) and field.get("description"):
                new_label = _WEBUI_LABEL_OVERRIDES.get(fname) or _webui_label_from_description(
                    field["description"]
                )
                if new_label and new_label != fname:
                    field["label"] = new_label
    return schema


def create_plugin() -> WebSearchPlugin:
    """Runner 加载入口。"""
    return WebSearchPlugin()
