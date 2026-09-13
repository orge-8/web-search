"""搜索引擎实现与降级链。

设计原则：**引擎之间互相独立，任一可用即返回。**

参考了两个社区插件的做法后，这里做了一个明确取舍——支持两类引擎：

* **免 Key 引擎**：DuckDuckGo HTML 版、Bing 网页版、SearXNG 实例。
  不需要申请密钥，但依赖解析对方 HTML，结构变动时会失效。
* **API 引擎**：Tavily。返回结构化 JSON 且自带正文摘要，最稳，但需要 Key。

国内网络环境下 Google 直连不可用，所以默认引擎链是 ``duckduckgo → bing``，
把 Tavily 放在可配置的追加位置（配了 key 就能用）。

每个引擎失败时抛出 :class:`EngineError`（含中文原因），由 :class:`EngineChain`
记录后继续尝试下一个；最终把每个引擎的失败原因一并交给诊断命令展示——
这样"搜不到"和"引擎挂了"能被区分开，而不是笼统报一句失败。
"""

import asyncio
import base64
import binascii
import json
import re
import time
from typing import Any, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from .models import SearchResult
from .relevance import describe, is_low_relevance, relevance_ratio

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 百度对请求指纹比较敏感：只带 UA 的裸请求更容易被风控盯上
_BAIDU_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

# 搜索结果里常见的跳转/追踪域名，抓到这类链接没有意义
_UNUSABLE_HOSTS = {
    "www.bing.com",
    "bing.com",
    "duckduckgo.com",
    "html.duckduckgo.com",
    "r.bing.com",
    "go.microsoft.com",
}

_WHITESPACE_RE = re.compile(r"\s+")


class EngineError(RuntimeError):
    """引擎级失败，携带可直接展示给用户的中文原因。"""


class SearchEngine(Protocol):
    """搜索引擎协议。"""

    name: str

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """执行搜索。失败时抛 EngineError。"""
        ...


def _clean(text: str) -> str:
    """压缩空白。"""
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """按 URL 去重，保留先出现的（引擎排序即相关度排序）。"""
    seen: set[str] = set()
    output: list[SearchResult] = []
    for item in results:
        key = (item.url or "").rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _unwrap_bing_url(href: str) -> str:
    """解开 Bing 的 ck/a 跳转包装链接。

    Bing 有时把真实地址 base64 进 ``u=a1<base64>`` 参数，直接返回会拿到
    ``www.bing.com/ck/a?...`` 这种无法抓取的地址。
    """
    if not href:
        return ""
    if "bing.com/ck/a" not in href:
        return href
    try:
        query = parse_qs(urlparse(href).query)
        raw = (query.get("u") or [""])[0]
        if raw.startswith("a1"):
            raw = raw[2:]
            padding = "=" * (-len(raw) % 4)
            decoded = base64.urlsafe_b64decode(raw + padding).decode("utf-8", "replace")
            if decoded.startswith("http"):
                return decoded
    except (ValueError, binascii.Error, UnicodeDecodeError):
        pass
    return href


def _unwrap_ddg_url(href: str) -> str:
    """解开 DuckDuckGo 的 ``/l/?uddg=`` 跳转链接。"""
    if not href:
        return ""
    if "uddg=" in href:
        try:
            query = parse_qs(urlparse(href).query)
            target = (query.get("uddg") or [""])[0]
            if target:
                return unquote(target)
        except Exception:
            pass
    if href.startswith("//"):
        return "https:" + href
    return href


def _is_usable(url: str) -> bool:
    """判断 URL 是否值得作为结果返回。"""
    if not url or not url.lower().startswith(("http://", "https://")):
        return False
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return host not in _UNUSABLE_HOSTS


def _absolute_url(href: str, base: str) -> str:
    """把搜索结果里的相对链接补成绝对地址。

    百度会给出 ``/link?url=...`` 形式的跳转链接：这里先补全协议与主机，
    真实落地地址由抓取阶段跟随 302 解出（见 plugin 侧用 final_url 回写）。
    """
    raw = (href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("/"):
        return base.rstrip("/") + raw
    return raw


class _HttpEngine:
    """共享 httpx 客户端的引擎基类。"""

    name = "base"

    def __init__(self, client: httpx.AsyncClient, *, language: str = "zh-CN") -> None:
        self._client = client
        self.language = language or "zh-CN"

    async def _get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """发起 GET 请求并返回解码后的文本。"""
        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise EngineError("请求超时") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"网络错误：{exc}") from exc

        if response.status_code >= 400:
            raise EngineError(f"返回 HTTP {response.status_code}")
        return response.text

    async def search(self, query: str, limit: int) -> list[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class DuckDuckGoEngine(_HttpEngine):
    """DuckDuckGo HTML 版（免 Key，国内可达性较好）。"""

    name = "duckduckgo"
    _ENDPOINT = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """执行 DuckDuckGo 搜索。"""
        region = "cn-zh" if self.language.lower().startswith("zh") else "wt-wt"
        html = await self._get_text(self._ENDPOINT, {"q": query, "kl": region})
        soup = BeautifulSoup(html, "html.parser")

        blocks = soup.select("div.result, div.web-result, div.result__body")
        if not blocks:
            # 反爬页面通常既没有结果也没有报错，这里显式区分
            if "anomaly" in html.lower() or "unusual traffic" in html.lower():
                raise EngineError("触发反爬验证")
            raise EngineError("页面结构无法识别（可能已改版）")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for block in blocks:
            link = block.select_one("a.result__a") or block.select_one("a.result__url")
            if link is None:
                continue
            href = _unwrap_ddg_url(str(link.get("href") or ""))
            if not _is_usable(href):
                continue
            # 容器选择器同时命中外层 result 与内层 result__body，同一结果会被
            # 解析两次，必须在这里去重，否则降级链看到的是重复项。
            key = href.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            snippet_node = block.select_one(".result__snippet") or block.select_one(
                "a.result__snippet"
            )
            results.append(
                SearchResult(
                    title=_clean(link.get_text(" ", strip=True)),
                    url=href,
                    snippet=_clean(snippet_node.get_text(" ", strip=True))
                    if snippet_node
                    else "",
                    engine=self.name,
                )
            )
            if len(results) >= limit:
                break

        if not results:
            raise EngineError("未解析出任何结果")
        return results


class BingEngine(_HttpEngine):
    """Bing 网页版（免 Key）。"""

    name = "bing"
    _ENDPOINT = "https://www.bing.com/search"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """执行 Bing 搜索。"""
        lang = self.language.split("-")[0] or "zh"
        html = await self._get_text(
            self._ENDPOINT,
            {"q": query, "count": max(limit, 10), "setlang": lang, "ensearch": "0"},
        )
        soup = BeautifulSoup(html, "html.parser")

        items = soup.select("li.b_algo")
        if not items:
            items = soup.select("li.b_algo, div.b_algo")
        if not items:
            raise EngineError("页面结构无法识别（可能已改版或被要求验证）")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in items:
            link = item.select_one("h2 a") or item.select_one("a")
            if link is None:
                continue
            href = _unwrap_bing_url(str(link.get("href") or ""))
            if not _is_usable(href):
                continue
            key = href.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)

            snippet_node = (
                item.select_one("div.b_caption p")
                or item.select_one("p.b_lineclamp2")
                or item.select_one("p.b_lineclamp3")
                or item.select_one("div.b_caption")
            )
            results.append(
                SearchResult(
                    title=_clean(link.get_text(" ", strip=True)),
                    url=href,
                    snippet=_clean(snippet_node.get_text(" ", strip=True))
                    if snippet_node
                    else "",
                    engine=self.name,
                )
            )
            if len(results) >= limit:
                break

        if not results:
            raise EngineError("未解析出任何结果")
        return results


class BaiduEngine(_HttpEngine):
    """百度搜索 —— 国内网络下的首选引擎。

    实测依据：查询 ``洛天依 2026 演唱会`` 时，百度返回的是 2026 巡回演唱会各站信息，
    而 Bing 退化成对"洛"这个单字的匹配（返回汉字字典、洛谷、LOL 英雄）。
    根因是 Bing 对"含数字 + 空格分隔"的中文多词查询做逐词降级，
    而这恰好是 LLM 最常生成的查询形式，所以不能把 Bing 放在首位。

    已知代价：百度的频率风控比 Bing 严，裸请求连续触发会弹"安全验证"。
    对策是三层的：首次搜索前先访问首页取 Cookie（BAIDUID）、请求带完整浏览器头、
    触发验证时抛 EngineError 走降级链并进入冷却，等风控窗口过去自动恢复。
    """

    name = "baidu"
    _ENDPOINT = "https://www.baidu.com/s"
    _HOME = "https://www.baidu.com/"

    def __init__(self, client: httpx.AsyncClient, *, language: str = "zh-CN") -> None:
        super().__init__(client, language=language)
        self._warmed = False
        self._warm_lock = asyncio.Lock()

    async def _ensure_warm(self) -> None:
        """首次搜索前先访问首页取 Cookie，降低触发安全验证的概率。"""
        if self._warmed:
            return
        async with self._warm_lock:
            if self._warmed:
                return
            try:
                await self._client.get(self._HOME, headers=_BAIDU_HEADERS)
            except httpx.HTTPError:
                pass  # 预热失败不值得阻断搜索，让真正的搜索自己去试
            self._warmed = True

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """执行百度搜索。"""
        await self._ensure_warm()
        try:
            response = await self._client.get(
                self._ENDPOINT,
                params={"wd": query, "rn": max(limit, 10), "ie": "utf-8"},
                headers=_BAIDU_HEADERS,
            )
        except httpx.TimeoutException as exc:
            raise EngineError("请求超时") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"网络错误：{exc}") from exc

        if response.status_code >= 400:
            raise EngineError(f"返回 HTTP {response.status_code}")

        html = response.text
        if "安全验证" in html or "wappass.baidu.com" in html:
            raise EngineError("触发百度安全验证（请求过频），稍后会自动恢复")

        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one("#content_left")
        if container is None:
            raise EngineError("页面结构无法识别（可能已改版）")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in container.find_all("div", recursive=False):
            # 广告位带 data-tuiguang 标记，不能混进自然结果
            if item.get("data-tuiguang") is not None:
                continue
            link = item.select_one("h3 a")
            if link is None:
                continue
            href = _absolute_url(str(link.get("href") or ""), "https://www.baidu.com")
            title = _clean(link.get_text(" ", strip=True))
            if not href or not title:
                continue

            key = href.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)

            snippet_node = (
                item.select_one(".c-abstract")
                or item.select_one('[class*="content-right"]')
                or item.select_one('[class*="c-span-last"]')
            )
            snippet = _clean(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
            if not snippet:
                whole = _clean(item.get_text(" ", strip=True))
                snippet = whole.replace(title, "", 1).strip()[:300]

            results.append(
                SearchResult(title=title, url=href, snippet=snippet, engine=self.name)
            )
            if len(results) >= limit:
                break

        if not results:
            raise EngineError("未解析出任何结果")
        return results


class SearxngEngine(_HttpEngine):
    """自建 / 公共 SearXNG 实例（免 Key，但需实例开启 JSON 输出）。"""

    name = "searxng"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        language: str = "zh-CN",
    ) -> None:
        super().__init__(client, language=language)
        self.base_url = (base_url or "").rstrip("/")

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """通过 SearXNG JSON 接口搜索。"""
        if not self.base_url:
            raise EngineError("未配置 searxng 实例地址")

        url = f"{self.base_url}/search"
        try:
            response = await self._client.get(
                url,
                params={
                    "q": query,
                    "format": "json",
                    "language": self.language,
                    "safesearch": "0",
                },
            )
        except httpx.TimeoutException as exc:
            raise EngineError("请求超时") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"网络错误：{exc}") from exc

        if response.status_code >= 400:
            raise EngineError(f"返回 HTTP {response.status_code}")

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise EngineError("未返回 JSON（实例可能禁用了 json 格式）") from exc

        raw_items = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            raise EngineError("返回结构异常")

        results: list[SearchResult] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url") or "")
            if not _is_usable(link):
                continue
            results.append(
                SearchResult(
                    title=_clean(str(item.get("title") or "")),
                    url=link,
                    snippet=_clean(str(item.get("content") or "")),
                    engine=self.name,
                    score=float(item.get("score") or 0.0),
                )
            )
            if len(results) >= limit:
                break

        if not results:
            raise EngineError("未返回可用结果")
        return results


class TavilyEngine(_HttpEngine):
    """Tavily 搜索 API（需 Key；自带正文摘要，最稳）。"""

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        language: str = "zh-CN",
    ) -> None:
        super().__init__(client, language=language)
        self.api_key = (api_key or "").strip()

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """调用 Tavily 搜索接口。"""
        if not self.api_key:
            raise EngineError("未配置 tavily api_key")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max(1, min(limit, 10)),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = await self._client.post(self._ENDPOINT, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise EngineError("请求超时") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"网络错误：{exc}") from exc

        if response.status_code in (401, 403):
            raise EngineError("鉴权失败（api_key 无效或额度用尽）")
        if response.status_code == 429:
            raise EngineError("触发限流")
        if response.status_code >= 400:
            raise EngineError(f"返回 HTTP {response.status_code}")

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise EngineError("返回内容不是合法 JSON") from exc

        raw_items = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            raise EngineError("返回结构异常")

        results: list[SearchResult] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url") or "")
            if not _is_usable(link):
                continue
            results.append(
                SearchResult(
                    title=_clean(str(item.get("title") or "")),
                    url=link,
                    # Tavily 的 content 是页面的实质内容，可当作已抓取正文使用
                    snippet=_clean(str(item.get("content") or ""))[:400],
                    content=_clean(str(item.get("content") or "")),
                    engine=self.name,
                    score=float(item.get("score") or 0.0),
                )
            )
            if len(results) >= limit:
                break

        if not results:
            raise EngineError("未返回可用结果")
        return results


_SUPPORTED_ENGINES = ("baidu", "bing", "duckduckgo", "searxng", "tavily")


class EngineChain:
    """按配置顺序尝试的搜索引擎链。

    带**失败冷却**：某引擎失败后会在冷却期内被跳过。这不是小题大做——
    国内网络下 DuckDuckGo 的连接超时要烧掉十几秒，而它每次搜索都排在最前面
    的话，麦麦每次搜索都要先干等一轮。冷却期内若所有引擎都不可用，则忽略冷却
    强制再试一次，避免一次网络抖动让插件"永久失灵"。
    """

    def __init__(
        self,
        engines: list[SearchEngine],
        *,
        max_attempts: int = 0,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """初始化引擎链。

        Args:
            engines: 按优先级排列的引擎；顺序即降级顺序。
            max_attempts: 最多尝试几个引擎，0 表示全部尝试。
            cooldown_seconds: 引擎失败后的冷却时长（秒），0 表示不启用。
        """
        self.engines = engines
        self.max_attempts = max_attempts if max_attempts > 0 else len(engines)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._cooldown_until: dict[str, float] = {}

    @property
    def names(self) -> list[str]:
        """引擎名列表。"""
        return [engine.name for engine in self.engines]

    def cooldown_snapshot(self) -> dict[str, float]:
        """返回仍处于冷却中的引擎及剩余秒数（供诊断展示）。"""
        now = time.monotonic()
        return {
            name: round(until - now, 1)
            for name, until in self._cooldown_until.items()
            if until > now
        }

    def _candidates(self) -> list[SearchEngine]:
        """给出本轮可尝试的引擎；全部冷却中时强制返回全部。"""
        if self.cooldown_seconds <= 0:
            return list(self.engines)
        now = time.monotonic()
        fresh = [
            engine
            for engine in self.engines
            if self._cooldown_until.get(engine.name, 0.0) <= now
        ]
        return fresh or list(self.engines)

    def _mark_failed(self, name: str) -> None:
        """标记引擎失败并进入冷却。"""
        if self.cooldown_seconds > 0:
            self._cooldown_until[name] = time.monotonic() + self.cooldown_seconds

    def _mark_ok(self, name: str) -> None:
        """引擎成功后清除其冷却状态。"""
        self._cooldown_until.pop(name, None)

    async def search(
        self, query: str, limit: int
    ) -> tuple[list[SearchResult], list[tuple[str, str]]]:
        """依次尝试各引擎，返回首个"结果可信"的引擎输出。

        "可信"的判据不只是"有结果"，还包括相关性：搜索引擎对长查询做降级时
        会返回一批完全无关的结果（实测查「洛天依 2026 演唱会」拿到的是汉字"洛"
        的字典释义），此时应当继续降级到下一个引擎，而不是把垃圾喂给 LLM。

        若所有引擎的结果都低相关，则返回其中相对最好的那份——有噪声的材料
        强于完全没有材料，但调用方需要据此明确告知 LLM。

        Args:
            query: 搜索词。
            limit: 期望结果条数。

        Returns:
            tuple[list[SearchResult], list[tuple[str, str]]]:
                结果列表，以及 [(引擎名, 失败或可疑原因), ...]。
        """
        failures: list[tuple[str, str]] = []
        if not self.engines:
            return [], [("(未配置)", "引擎链为空，请检查 engines.order 配置")]

        fallback: tuple[float, list[SearchResult], str] | None = None

        for engine in self._candidates()[: self.max_attempts]:
            try:
                results = await engine.search(query, limit)
            except EngineError as exc:
                self._mark_failed(engine.name)
                failures.append((engine.name, str(exc)))
                continue
            except Exception as exc:  # 引擎解析代码的任何意外都不应阻断降级
                self._mark_failed(engine.name)
                failures.append((engine.name, f"未预期错误：{exc}"))
                continue

            results = _dedupe(results)
            if not results:
                failures.append((engine.name, "未返回结果"))
                continue

            ratio = relevance_ratio(query, results)
            if not is_low_relevance(ratio):
                self._mark_ok(engine.name)
                return results[:limit], failures

            # 引擎工作正常（返回了结构化结果），只是内容不对路：记入可疑列表，
            # 但不进冷却——避免把"查询词没选好"误判成"引擎坏了"。
            failures.append((engine.name, f"结果疑似与查询无关（{describe(ratio)}）"))
            if fallback is None or ratio > fallback[0]:
                fallback = (ratio, results[:limit], engine.name)

        if fallback is not None:
            self._mark_ok(fallback[2])
            return fallback[1], failures

        return [], failures


def build_engine_chain(
    *,
    client: httpx.AsyncClient,
    order: list[str],
    language: str = "zh-CN",
    searxng_base_url: str = "",
    tavily_api_key: str = "",
    max_attempts: int = 0,
    cooldown_seconds: float = 300.0,
) -> EngineChain:
    """按配置构建引擎链。

    未知引擎名会被静默跳过（配置里写错名字不应导致插件加载失败），
    由诊断命令展示实际生效的引擎列表。

    Args:
        client: 共享的 httpx 客户端。
        order: 引擎优先级顺序。
        language: 语言偏好。
        searxng_base_url: SearXNG 实例地址。
        tavily_api_key: Tavily 密钥。
        max_attempts: 最多尝试的引擎数。
        cooldown_seconds: 引擎失败后的冷却时长（秒）。

    Returns:
        EngineChain: 引擎链实例。
    """
    engines: list[SearchEngine] = []
    for raw_name in order:
        name = (raw_name or "").strip().lower()
        if name == "baidu":
            engines.append(BaiduEngine(client, language=language))
        elif name == "duckduckgo":
            engines.append(DuckDuckGoEngine(client, language=language))
        elif name == "bing":
            engines.append(BingEngine(client, language=language))
        elif name == "searxng":
            engines.append(
                SearxngEngine(client, base_url=searxng_base_url, language=language)
            )
        elif name == "tavily":
            engines.append(TavilyEngine(client, api_key=tavily_api_key, language=language))

    if not engines:
        # 一个有效引擎名都没配出来时，给出最可能在中文场景下工作的组合
        engines.append(BaiduEngine(client, language=language))
        engines.append(BingEngine(client, language=language))

    return EngineChain(
        engines, max_attempts=max_attempts, cooldown_seconds=cooldown_seconds
    )


def supported_engine_names() -> list[str]:
    """返回本插件支持的引擎名，供配置校验与文档使用。"""
    return list(_SUPPORTED_ENGINES)


def build_search_url(engine_name: str, query: str) -> str:
    """构造引擎的可点击搜索地址（诊断输出用）。"""
    encoded = quote_plus(query)
    mapping = {
        "baidu": f"https://www.baidu.com/s?wd={encoded}",
        "duckduckgo": f"https://duckduckgo.com/?q={encoded}",
        "bing": f"https://www.bing.com/search?q={encoded}",
        "tavily": "https://app.tavily.com/",
    }
    return mapping.get(engine_name, "")
