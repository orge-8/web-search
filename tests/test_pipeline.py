"""离线单测：核心模块（不联网、不启动 MaiBot Host）。

运行: python tests/test_pipeline.py

刻意不依赖 pytest —— 开发机与真机都未必装。自带极简断言运行器，退出码非 0
即表示有失败，可直接被 run_gates.py 采集。

覆盖范围：
  * 正文提取的降噪与容器选择
  * SSRF 防护的各个拒绝分支（这是本插件唯一按外部输入主动外呼的入口）
  * 缓存 TTL / LRU 行为
  * 各引擎的结果解析（用离线固件，不真实请求网络）
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from core.cache import TTLCache  # noqa: E402
from core.engines import (  # noqa: E402
    BaiduEngine,
    BingEngine,
    DuckDuckGoEngine,
    EngineError,
    SearxngEngine,
    TavilyEngine,
    build_engine_chain,
)
from core.extract import extract_main_text, format_json_response, html_to_text  # noqa: E402
from core.fetcher import is_safe_url  # noqa: E402
from core.models import SearchResult  # noqa: E402

_TESTS: list = []


def case(fn):
    """注册一个测试函数。

    刻意不叫 ``test`` —— pytest 默认收集所有 ``test*`` 开头的可调用对象，
    那个名字会被当成"需要一个叫 fn 的 fixture 的用例"，导致收集报错。
    """
    _TESTS.append(fn)
    return fn


# ------------------------------------------------------------------ 固件

_ARTICLE_HTML = """
<!DOCTYPE html>
<html><head>
  <title>测试文章标题 - 某资讯站</title>
  <meta property="og:title" content="OG 标题优先">
  <script>var tracking = "abcdefg";</script>
  <style>.ad { display: block; }</style>
</head>
<body>
  <nav class="navbar">首页 关于我们 联系方式 隐私政策</nav>
  <header class="site-header">某某资讯网</header>
  <div class="sidebar">热门推荐 点击这里</div>
  <article class="article-content">
    <h1>正文里的一级标题</h1>
    <p>这是第一段正文内容，需要足够长才能通过最小长度阈值的判断，否则提取器会认为没有拿到正文而回退到全页文本。</p>
    <p>这是第二段正文内容，同样需要比较多的文字来保证整体长度超过阈值，同时验证多个块级元素都能被收集到。</p>
    <p>这是第三段正文内容，用于确认真实段落文本被完整保留下来，而导航和页脚这类噪声已经被剥离干净。</p>
  </article>
  <aside class="related">相关阅读 猜你喜欢</aside>
  <footer class="site-footer">版权所有 侵权必究 备案号 123456</footer>
</body></html>
"""

_DDG_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1&amp;rut=xyz">结果标题一</a>
    </h2>
    <a class="result__snippet">这是结果一的摘要内容。</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage2">结果标题二</a>
    </h2>
    <a class="result__snippet">这是结果二的摘要内容。</a>
  </div>
</div>
</body></html>
"""

_BING_HTML = """
<html><body><ol id="b_results">
  <li class="b_algo">
    <h2><a href="https://example.com/bing-one">Bing 结果标题一</a></h2>
    <div class="b_caption"><p>Bing 的摘要内容一。</p></div>
  </li>
  <li class="b_algo">
    <h2><a href="https://www.bing.com/ck/a?!&&p=abc&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9kZWNvZGVk&amp;ntb=1">Bing 结果标题二</a></h2>
    <div class="b_caption"><p>Bing 的摘要内容二。</p></div>
  </li>
</ol></body></html>
"""


class _FakeResponse:
    """伪装 httpx.Response。"""

    def __init__(self, *, text: str = "", payload=None, status_code: int = 200) -> None:
        self.text = text
        self._payload = payload
        self.status_code = status_code

    def json(self):
        """返回预设 JSON。"""
        if self._payload is None:
            raise ValueError("没有 JSON 载荷")
        return self._payload


class _FakeClient:
    """伪装 httpx.AsyncClient，记录请求但不发出去。"""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.requests: list = []

    async def get(self, url, params=None, **kwargs):
        """记录 GET 请求。"""
        self.requests.append(("GET", url, params))
        return self._response

    async def post(self, url, json=None, headers=None, **kwargs):
        """记录 POST 请求。"""
        self.requests.append(("POST", url, json))
        return self._response


# ------------------------------------------------------------------ 正文提取


@case
def test_extract_strips_noise() -> None:
    """正文提取应剥离导航/页脚/脚本，并保留真实段落。"""
    title, text = extract_main_text(_ARTICLE_HTML)
    assert "OG 标题优先" in title, f"og:title 未优先采用: {title!r}"
    assert "第一段正文内容" in text, "正文段落缺失"
    assert "第三段正文内容" in text, "正文段落不完整"
    assert "首页 关于我们" not in text, f"导航未被剥离: {text[:120]!r}"
    assert "版权所有" not in text, f"页脚未被剥离: {text[:200]!r}"
    assert "tracking" not in text, "脚本内容未被剥离"
    assert "display: block" not in text, "样式内容未被剥离"


@case
def test_extract_fallback_for_plain_page() -> None:
    """没有 article 容器的页面应回退到全页文本。"""
    html = "<html><body><div><p>" + "一段没有语义容器的正文。" * 30 + "</p></div></body></html>"
    _, text = extract_main_text(html)
    assert "一段没有语义容器的正文" in text, "兜底提取失败"


@case
def test_extract_handles_garbage_input() -> None:
    """空输入与非法 HTML 都不应抛异常。"""
    assert extract_main_text("") == ("", "")
    assert extract_main_text("   ") == ("", "")
    assert html_to_text("") == ""
    assert isinstance(html_to_text("<html><body><p>ok</p></body></html>"), str)


@case
def test_extract_truncates() -> None:
    """max_chars 应生效。"""
    html = "<html><body><p>" + "很长的一段文字。" * 200 + "</p></body></html>"
    _, text = extract_main_text(html, max_chars=50)
    assert len(text) <= 50, f"截断未生效: {len(text)}"


@case
def test_format_json_response() -> None:
    """合法 JSON 格式化，非法 JSON 原样返回。"""
    pretty = format_json_response('{"a": 1, "b": [1, 2]}')
    assert '"a": 1' in pretty, f"JSON 未格式化: {pretty!r}"
    assert format_json_response("not json at all") == "not json at all"


# ------------------------------------------------------------------ SSRF


@case
def test_ssrf_blocks_private_and_metadata() -> None:
    """内网、环回、云元数据地址必须被拒绝。"""
    blocked = [
        "http://127.0.0.1/",
        "http://127.0.0.1:8001/dashboard",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://localhost/",
    ]
    for url in blocked:
        allowed, reason = is_safe_url(url)
        assert allowed is False, f"未拦截: {url}"
        assert reason, f"未给出拦截原因: {url}"


@case
def test_ssrf_blocks_bad_schemes() -> None:
    """非 http/https 协议必须被拒绝。"""
    for url in ["file:///etc/passwd", "ftp://example.com/a", "gopher://x/", "javascript:alert(1)"]:
        allowed, reason = is_safe_url(url)
        assert allowed is False, f"未拦截非法协议: {url}"
        assert reason, f"未给出原因: {url}"


@case
def test_ssrf_allows_public_ip() -> None:
    """公网 IP 应放行。"""
    allowed, reason = is_safe_url("https://93.184.216.34/page")
    assert allowed is True, f"公网地址被误拦: {reason}"


@case
def test_ssrf_empty_and_malformed() -> None:
    """空值与畸形地址不应通过。"""
    for url in ["", "   ", "http://", "not-a-url"]:
        allowed, _ = is_safe_url(url)
        assert allowed is False, f"未拦截: {url!r}"


@case
def test_ssrf_allow_private_switch() -> None:
    """开启 allow_private 后内网放行（开关本身要有效）。"""
    allowed, _ = is_safe_url("http://127.0.0.1/", allow_private=True)
    assert allowed is True, "allow_private=True 未放行内网地址"


# ------------------------------------------------------------------ 缓存


@case
def test_cache_basic() -> None:
    """写入后应能读到。"""
    cache = TTLCache(max_entries=4, ttl_seconds=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert cache.get("missing") is None
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1, f"统计异常: {stats}"


@case
def test_cache_lru_eviction() -> None:
    """超出容量应淘汰最久未使用者。"""
    cache = TTLCache(max_entries=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # a 变成最近使用
    cache.set("c", 3)  # 应淘汰 b
    assert cache.get("a") == 1, "最近使用的条目被误淘汰"
    assert cache.get("b") is None, "最久未使用条目未被淘汰"
    assert cache.get("c") == 3


@case
def test_cache_expiry() -> None:
    """超过 TTL 应失效。"""
    cache = TTLCache(max_entries=4, ttl_seconds=1)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(1.05)
    assert cache.get("k") is None, "过期条目仍可读取"


@case
def test_cache_clear() -> None:
    """clear 应清空并返回条数。"""
    cache = TTLCache(max_entries=4, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.clear() == 2
    assert cache.stats()["entries"] == 0


# ------------------------------------------------------------------ 数据结构


@case
def test_search_result_rendering() -> None:
    """材料块与来源行的渲染格式。"""
    item = SearchResult(title="标题", url="https://example.com/x", content="正文内容")
    assert item.has_content is True
    block = item.to_prompt_block(1)
    assert block.startswith("[1] 标题"), f"材料块编号异常: {block!r}"
    assert "URL: https://example.com/x" in block
    assert item.to_source_line(3) == "[3] 标题 — https://example.com/x"

    empty = SearchResult(title="只有标题", url="https://example.com/y")
    assert empty.has_content is False
    assert "(未能获取正文" in empty.to_prompt_block(2), "无正文时未给出占位说明"


@case
def test_search_result_prefers_content_over_snippet() -> None:
    """有正文时用正文，没有才退回摘要。"""
    item = SearchResult(title="T", url="u", snippet="摘要文本", content="正文文本")
    block = item.to_prompt_block(1)
    assert "正文文本" in block and "摘要文本" not in block


# ------------------------------------------------------------------ 引擎解析


@case
def test_duckduckgo_parsing() -> None:
    """DDG 结果解析与跳转链接解包。"""
    client = _FakeClient(_FakeResponse(text=_DDG_HTML))
    engine = DuckDuckGoEngine(client)
    results = asyncio.run(engine.search("测试", 5))
    assert len(results) == 2, f"结果条数异常: {len(results)}"
    assert results[0].url == "https://example.com/page1", f"跳转链接未解包: {results[0].url}"
    assert results[0].title == "结果标题一", f"标题解析异常: {results[0].title}"
    assert "摘要内容" in results[0].snippet, "摘要未解析"
    assert results[0].engine == "duckduckgo"


@case
def test_bing_parsing_and_ck_unwrap() -> None:
    """Bing 结果解析，含 ck/a 包装链接解包。"""
    client = _FakeClient(_FakeResponse(text=_BING_HTML))
    engine = BingEngine(client)
    results = asyncio.run(engine.search("测试", 5))
    assert len(results) == 2, f"结果条数异常: {len(results)}"
    assert results[0].url == "https://example.com/bing-one"
    assert results[0].title == "Bing 结果标题一"
    assert results[1].url == "https://example.com/decoded", f"ck/a 未解包: {results[1].url}"


@case
def test_engine_raises_on_unrecognized_page() -> None:
    """页面结构无法识别时应抛 EngineError（以便降级），而不是返回空列表。"""
    client = _FakeClient(_FakeResponse(text="<html><body>nothing here</body></html>"))
    try:
        asyncio.run(DuckDuckGoEngine(client).search("测试", 5))
    except EngineError:
        return
    raise AssertionError("结构异常时未抛出 EngineError，降级链会误判为成功")


@case
def test_engine_raises_on_http_error() -> None:
    """HTTP 4xx/5xx 应抛 EngineError。"""
    client = _FakeClient(_FakeResponse(text="", status_code=503))
    try:
        asyncio.run(BingEngine(client).search("测试", 5))
    except EngineError as exc:
        assert "503" in str(exc), f"错误信息未包含状态码: {exc}"
        return
    raise AssertionError("HTTP 503 未抛出 EngineError")


@case
def test_searxng_parsing() -> None:
    """SearXNG JSON 解析。"""
    payload = {
        "results": [
            {"title": "S1", "url": "https://example.com/s1", "content": "内容一", "score": 0.8},
            {"title": "S2", "url": "https://example.com/s2", "content": "内容二", "score": 0.5},
        ]
    }
    client = _FakeClient(_FakeResponse(payload=payload))
    engine = SearxngEngine(client, base_url="https://searx.example.com")
    results = asyncio.run(engine.search("测试", 5))
    assert len(results) == 2
    assert results[0].url == "https://example.com/s1"
    assert results[0].score == 0.8


@case
def test_searxng_requires_base_url() -> None:
    """未配置实例地址时应抛 EngineError。"""
    client = _FakeClient(_FakeResponse(payload={}))
    try:
        asyncio.run(SearxngEngine(client, base_url="").search("测试", 5))
    except EngineError as exc:
        assert "实例地址" in str(exc), f"错误提示不明确: {exc}"
        return
    raise AssertionError("未配置 base_url 时未抛 EngineError")


@case
def test_tavily_parsing() -> None:
    """Tavily 解析，且 content 直接作为正文。"""
    payload = {
        "results": [
            {"title": "T1", "url": "https://example.com/t1", "content": "正文一", "score": 0.9}
        ]
    }
    client = _FakeClient(_FakeResponse(payload=payload))
    engine = TavilyEngine(client, api_key="tvly-test")
    results = asyncio.run(engine.search("测试", 5))
    assert len(results) == 1
    assert results[0].has_content is True, "tavily 的 content 未作为正文保留"


@case
def test_tavily_requires_api_key() -> None:
    """未配置 Key 时应抛 EngineError。"""
    client = _FakeClient(_FakeResponse(payload={}))
    try:
        asyncio.run(TavilyEngine(client, api_key="").search("测试", 5))
    except EngineError as exc:
        assert "api_key" in str(exc), f"错误提示不明确: {exc}"
        return
    raise AssertionError("未配置 api_key 时未抛 EngineError")


# ------------------------------------------------------------------ 引擎链


@case
def test_engine_chain_build_order() -> None:
    """引擎链按配置顺序构建，未知名字被跳过。"""
    chain = build_engine_chain(client=object(), order=["bing", "duckduckgo"],
                               max_attempts=2)
    assert chain.names == ["bing", "duckduckgo"], f"顺序异常: {chain.names}"

    chain2 = build_engine_chain(client=object(), order=["unknown-engine", "bing"])
    assert chain2.names == ["bing"], f"未知引擎未被跳过: {chain2.names}"


@case
def test_engine_chain_default_fallback() -> None:
    """配置为空时应给出默认引擎，而不是空链。"""
    chain = build_engine_chain(client=object(), order=[])
    assert chain.names, "空配置未给出默认引擎"
    assert "baidu" in chain.names, "默认兜底应包含 baidu（中文长查询质量实测最好）"


@case
def test_engine_chain_reports_failures() -> None:
    """全部引擎失败时，应返回失败原因列表而不是抛异常。"""
    class _DeadEngine:
        name = "dead"

        async def search(self, query: str, limit: int):
            raise EngineError("模拟失败")

    from core.engines import EngineChain

    chain = EngineChain([_DeadEngine()])
    results, failures = asyncio.run(chain.search("测试", 5))
    assert results == [], "失败时不应有结果"
    assert failures and failures[0][0] == "dead", f"失败信息未记录: {failures}"
    assert "模拟失败" in failures[0][1]


@case
def test_engine_chain_falls_through_to_next() -> None:
    """前一个引擎失败时应继续尝试下一个。"""

    class _DeadEngine:
        name = "dead"

        async def search(self, query: str, limit: int):
            raise EngineError("挂了")

    class _GoodEngine:
        name = "good"

        async def search(self, query: str, limit: int):
            # 标题里带上查询词：否则结果会被相关性检查判为"疑似无关"而走兜底分支，
            # 测不到"降到下一个引擎并立即返回"这条主路径
            return [
                SearchResult(
                    title=f"{query}的结果", url="https://example.com/ok", engine="good"
                )
            ]

    from core.engines import EngineChain

    chain = EngineChain([_DeadEngine(), _GoodEngine()])
    results, failures = asyncio.run(chain.search("测试", 5))
    assert len(results) == 1, f"降级失败: {results}"
    assert results[0].url == "https://example.com/ok"
    assert len(failures) == 1 and failures[0][0] == "dead", "前置失败未被记录"


@case
def test_engine_chain_dedupes_by_url() -> None:
    """不同引擎给出的相同 URL 应去重。"""

    class _DupEngine:
        name = "dup"

        async def search(self, query: str, limit: int):
            return [
                SearchResult(title="A", url="https://example.com/same", engine="dup"),
                SearchResult(title="B", url="https://example.com/same/", engine="dup"),
                SearchResult(title="C", url="https://example.com/other", engine="dup"),
            ]

    from core.engines import EngineChain

    chain = EngineChain([_DupEngine()])
    results, _ = asyncio.run(chain.search("测试", 5))
    assert len(results) == 2, f"去重未生效: {[r.url for r in results]}"


@case
def test_engine_chain_cooldown_skips_failed_engine() -> None:
    """失败引擎进入冷却后，后续搜索不应再尝试它。"""

    class _DeadEngine:
        name = "dead"
        attempts = 0

        async def search(self, query: str, limit: int):
            type(self).attempts += 1
            raise EngineError("挂了")

    class _GoodEngine:
        name = "good"

        async def search(self, query: str, limit: int):
            return [SearchResult(title="T", url="https://example.com/ok", engine="good")]

    from core.engines import EngineChain

    dead = _DeadEngine()
    chain = EngineChain([dead, _GoodEngine()], cooldown_seconds=300)

    asyncio.run(chain.search("测试", 5))
    assert dead.attempts == 1, "首次搜索未尝试该引擎"

    results, _ = asyncio.run(chain.search("测试", 5))
    assert dead.attempts == 1, f"冷却期内的引擎仍被重复调用：{dead.attempts} 次"
    assert len(results) == 1, "冷却期间未正常使用健康引擎"
    assert chain.cooldown_snapshot().get("dead", 0) > 0, "冷却状态未记录"


@case
def test_engine_chain_cooldown_all_down_forces_retry() -> None:
    """所有引擎都在冷却中时必须强制重试，否则插件会永久失灵。"""

    class _DeadEngine:
        name = "dead"
        attempts = 0

        async def search(self, query: str, limit: int):
            type(self).attempts += 1
            raise EngineError("挂了")

    from core.engines import EngineChain

    dead = _DeadEngine()
    chain = EngineChain([dead], cooldown_seconds=300)

    asyncio.run(chain.search("测试", 5))
    assert dead.attempts == 1
    asyncio.run(chain.search("测试", 5))
    assert dead.attempts == 2, "全部引擎冷却时未强制重试，插件将永久无法搜索"


@case
def test_engine_chain_cooldown_can_be_disabled() -> None:
    """cooldown_seconds=0 时不做冷却，每次都重试。"""

    class _DeadEngine:
        name = "dead"
        attempts = 0

        async def search(self, query: str, limit: int):
            type(self).attempts += 1
            raise EngineError("挂了")

    from core.engines import EngineChain

    dead = _DeadEngine()
    chain = EngineChain([dead], cooldown_seconds=0)

    asyncio.run(chain.search("测试", 5))
    asyncio.run(chain.search("测试", 5))
    assert dead.attempts == 2, "关闭冷却后未重试"
    assert chain.cooldown_snapshot() == {}, "关闭冷却后仍记录了冷却状态"


@case
def test_engine_chain_recovers_after_success() -> None:
    """引擎恢复成功后，其冷却状态应被清除。"""

    class _FlakyEngine:
        name = "flaky"
        attempts = 0

        async def search(self, query: str, limit: int):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise EngineError("第一次失败")
            return [SearchResult(title="T", url="https://example.com/ok", engine="flaky")]

    from core.engines import EngineChain

    engine = _FlakyEngine()
    chain = EngineChain([engine], cooldown_seconds=300)

    asyncio.run(chain.search("测试", 5))
    assert chain.cooldown_snapshot().get("flaky", 0) > 0, "失败后未进入冷却"

    # 全部引擎冷却 → 强制重试 → 这次成功，冷却应被清除
    asyncio.run(chain.search("测试", 5))
    assert chain.cooldown_snapshot() == {}, "成功后未清除冷却状态"


# ------------------------------------------------------------- 相关性检测


@case
def test_relevance_terms_extraction() -> None:
    """查询切词：长串按虚词切开，单字噪声被丢弃。"""
    from core.relevance import extract_terms

    assert extract_terms("洛天依 2026 演唱会") == ["洛天依", "2026", "演唱会"]
    # "最近的活动" 应被切成「最近」「活动」，"中V" 的单字不进词表
    assert "活动" in extract_terms("中V最近的活动")
    assert "最近" in extract_terms("中V最近的活动")
    assert "中" not in extract_terms("中V最近的活动")
    assert extract_terms("") == []
    assert extract_terms("的 了 在") == []


@case
def test_relevance_ratio_identifies_garbage() -> None:
    """引擎降级产物（单字字典条目）应被判为疑似无关。"""
    from core.relevance import is_low_relevance, relevance_ratio

    garbage = [
        SearchResult(title="洛 （汉字）_百度百科", url="https://baike.baidu.com/item/x", snippet="洛，汉语规范汉字"),
        SearchResult(title="题目列表 - 洛 谷", url="https://www.luogu.com.cn/problem/list", snippet="洛谷题目列表"),
        SearchResult(title="幻翎（英雄联盟英雄）_百度百科", url="https://baike.baidu.com/item/y", snippet="幻翎·洛是英雄联盟英雄"),
    ]
    ratio = relevance_ratio("洛天依 2026 演唱会", garbage)
    assert ratio == 0.0, f"垃圾结果命中率应为 0: {ratio}"
    assert is_low_relevance(ratio), "垃圾结果未被判定为低相关"


@case
def test_relevance_ratio_passes_good_results() -> None:
    """与查询实体匹配的正常结果不应被误杀。"""
    from core.relevance import is_low_relevance, relevance_ratio

    good = [
        SearchResult(title="洛天依 2026「无限共鸣·纯蓝幻乐」巡回演唱会", url="https://vcpedia.cn/x", snippet="洛天依2026巡回演唱会北京站"),
        SearchResult(title="2026 洛天依「无限共鸣·纯蓝幻乐」巡回演唱会_百度百科", url="https://baike.baidu.com/item/z", snippet="2026年洛天依巡回演唱会"),
    ]
    ratio = relevance_ratio("洛天依 2026 演唱会", good)
    assert ratio == 1.0, f"正常结果命中率为 1: {ratio}"
    assert not is_low_relevance(ratio), "正常结果被误判为低相关"


@case
def test_relevance_undeterminable_passes_through() -> None:
    """查询无可切分片段时应返回 -1（无法判断），调用方放行。"""
    from core.relevance import is_low_relevance, relevance_ratio

    assert relevance_ratio("的", []) == -1.0
    assert relevance_ratio("", [SearchResult(title="x", url="https://example.com")]) == -1.0
    assert not is_low_relevance(-1.0), "无法判断不应被判为低相关"


@case
def test_engine_chain_low_relevance_falls_through() -> None:
    """引擎返回"有结果但完全无关"时应继续降级，而不是直接采信。"""

    class _GarbageEngine:
        name = "garbage"

        async def search(self, query: str, limit: int):
            return [
                SearchResult(
                    title="无关结果一", url="https://example.com/g1", engine="garbage"
                ),
                SearchResult(
                    title="无关结果二", url="https://example.com/g2", engine="garbage"
                ),
            ]

    class _GoodEngine:
        name = "good"

        async def search(self, query: str, limit: int):
            return [
                SearchResult(
                    title=f"{query}的结果", url="https://example.com/ok", engine="good"
                )
            ]

    from core.engines import EngineChain

    chain = EngineChain([_GarbageEngine(), _GoodEngine()], cooldown_seconds=0)
    results, failures = asyncio.run(chain.search("测试", 5))
    assert results and results[0].url == "https://example.com/ok", (
        f"低相关结果未触发降级: {results}"
    )
    assert any(name == "garbage" and "无关" in reason for name, reason in failures), (
        f"低相关原因未被记录: {failures}"
    )
    # 引擎本身没坏（返回了结构化结果），不应进冷却
    assert chain.cooldown_snapshot() == {}, "低相关不应触发冷却"


@case
def test_engine_chain_all_low_relevance_returns_best() -> None:
    """所有引擎都低相关时，应返回相对最好的那份，而不是空手而归。"""

    class _WeakEngine:
        name = "weak"

        async def search(self, query: str, limit: int):
            return [SearchResult(title="完全无关", url="https://example.com/w", engine="weak")]

    from core.engines import EngineChain

    chain = EngineChain([_WeakEngine()], cooldown_seconds=0)
    results, failures = asyncio.run(chain.search("测试", 5))
    assert len(results) == 1, f"兜底结果丢失: {results}"
    assert any("无关" in reason for _, reason in failures), f"可疑原因未记录: {failures}"


# ------------------------------------------------------------------ 百度


_BAIDU_HTML = """
<html><body>
<div id="content_left">
  <div class="result c-container" data-tuiguang="1">
    <h3><a href="/link?url=ad1">广告位标题</a></h3><div class="c-abstract">广告摘要</div>
  </div>
  <div class="result c-container">
    <h3><a href="/link?url=abc">洛天依 2026 巡回演唱会 官方公告</a></h3>
    <div class="c-abstract">洛天依2026「无限共鸣·纯蓝幻乐」巡回演唱会的场次与开票信息。</div>
  </div>
  <div class="result c-container">
    <h3><a href="https://vsinger.com/live">VSINGER LIVE 官方站</a></h3>
    <div class="c-abstract">Vsinger Live 全息演唱会官方页面。</div>
  </div>
  <div class="c-container"><div>没有 h3 的杂项模块</div></div>
</div>
</body></html>
"""


@case
def test_baidu_parsing() -> None:
    """百度结果解析：跳转链接补全、广告剔除、无标题模块跳过。"""
    client = _FakeClient(_FakeResponse(text=_BAIDU_HTML))
    engine = BaiduEngine(client)
    results = asyncio.run(engine.search("洛天依 演唱会", 5))
    assert len(results) == 2, f"结果条数异常: {len(results)}"
    assert "广告位标题" not in [item.title for item in results], "广告未被剔除"

    first = results[0]
    assert first.title == "洛天依 2026 巡回演唱会 官方公告"
    # 相对跳转链接应补全成绝对地址
    assert first.url == "https://www.baidu.com/link?url=abc", f"相对链接未补全: {first.url}"
    assert "演唱会" in first.snippet
    assert first.engine == "baidu"

    # 绝对链接原样保留
    assert results[1].url == "https://vsinger.com/live"


@case
def test_baidu_detects_captcha() -> None:
    """百度安全验证页应抛 EngineError 以触发降级，而不是静默返回空。"""
    client = _FakeClient(_FakeResponse(text="<html>百度安全验证 wappass.baidu.com</html>"))
    try:
        asyncio.run(BaiduEngine(client).search("测试", 5))
    except EngineError as exc:
        assert "安全验证" in str(exc), f"错误提示不明确: {exc}"
        return
    raise AssertionError("安全验证页未抛出 EngineError")


@case
def test_engine_chain_build_includes_baidu() -> None:
    """默认引擎链应包含 baidu 且排在首位。"""
    chain = build_engine_chain(client=object(), order=["baidu", "bing", "duckduckgo"])
    assert chain.names == ["baidu", "bing", "duckduckgo"], f"顺序异常: {chain.names}"

    fallback = build_engine_chain(client=object(), order=[])
    assert fallback.names[0] == "baidu", f"空配置兜底未以 baidu 开头: {fallback.names}"


# ------------------------------------------------------------------ 图片搜索


from core.image_download import ImageDownload, ImageDownloader  # noqa: E402
from core.image_search import (  # noqa: E402
    BaiduImageEngine,
    ImageResult,
    _decode_baidu_url,
    _decode_objurl,
    _looks_encrypted_url,
    parse_acjson,
)

_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd400000000"
    "49454e44ae426082"
)

_ACJSON_SAMPLE = {
    "data": [
        {
            "fromPageTitle": "洛天依 舞台照_第一张",
            "thumbURL": "https://img0.baidu.com/it/u=1,100&fm=253",
            "hoverURL": "https://img1.baidu.com/it/u=1,200&fm=253",
            "objURL": "https://example.com/original-1.jpg",
            "fromURL": "https://source.example.com/page1",
            "width": 1024,
            "height": 768,
        },
        {
            "fromPageTitle": "洛天依 &amp; 官方 <剧照>",
            "thumbURL": "https://img0.baidu.com/it/u=2,300&fm=253",
            "hoverURL": "",
            "objURL": "aHR0cHM6Ly9leGFtcGxlLmNvbS9vcmlnaW5hbC0yLmpwZw==",
            "fromURL": "https://source.example.com/page2",
            "width": 800,
            "height": 600,
        },
        {},  # 百度返回末尾常带空项
        {
            "fromPageTitle": "无缩略图的项应被剔除",
            "hoverURL": "https://img1.baidu.com/hover-only.jpg",
            "fromURL": "https://source.example.com/page3",
        },
    ]
}


@case
def test_image_parse_acjson() -> None:
    """acjson 解析：字段映射正确，空项与无缩略图项被跳过。"""
    results = parse_acjson(_ACJSON_SAMPLE)
    assert len(results) == 2, f"应解析出 2 条，实际 {len(results)}"

    first = results[0]
    assert first.title == "洛天依 舞台照_第一张"
    assert first.thumb_url == "https://img0.baidu.com/it/u=1,100&fm=253"
    assert first.hover_url == "https://img1.baidu.com/it/u=1,200&fm=253"
    assert first.image_url == "https://example.com/original-1.jpg"
    assert first.source_url == "https://source.example.com/page1"
    assert first.width == 1024 and first.height == 768
    assert first.engine == "baidu_image"

    second = results[1]
    assert second.image_url == "https://example.com/original-2.jpg", "base64 objURL 未解码"


@case
def test_image_parse_title_cleanup() -> None:
    """标题里的 HTML 实体应解码为纯文本；成对标签被剥离。"""
    results = parse_acjson(_ACJSON_SAMPLE)
    assert results[1].title == "洛天依 & 官方", f"标题未清洗: {results[1].title!r}"


@case
def test_image_parse_requires_thumb() -> None:
    """无 thumbURL 的项没有可用下载源，必须剔除。"""
    payload = {"data": [{"fromPageTitle": "无图", "fromURL": "https://x.com/a"}]}
    assert parse_acjson(payload) == []
    assert parse_acjson({"data": []}) == []
    assert parse_acjson({}) == []
    assert parse_acjson({"data": "not-a-list"}) == []


@case
def test_image_objurl_decode() -> None:
    """objURL 解码分支：明文直返 / URL 编码 / base64 / 替换加密；乱码返回空。"""
    assert _decode_objurl("https://a.com/x.jpg") == "https://a.com/x.jpg"
    assert (
        _decode_objurl("https%3A%2F%2Fa.com%2Fx.jpg") == "https://a.com/x.jpg"
    )
    assert (
        _decode_objurl("aHR0cHM6Ly9leGFtcGxlLmNvbS9pLmpwZw==")
        == "https://example.com/i.jpg"
    )
    assert _decode_objurl("@@@garbage@@@") == ""
    assert _decode_objurl("") == ""
    # 替换加密串（真机日志样本）：走 _decode_baidu_url 分支
    encrypted = (
        "ippr_z2C$qAzdH3FAzdH3Fooo_z&e3Bktstktst_z&e3Bv54"
        "AzdH3F6jw1AzdH3F45ktsj?t1=n888mdn0"
    )
    decoded = _decode_objurl(encrypted)
    assert decoded.startswith("http://"), f"替换加密 objURL 应解出 http: {decoded!r}"
    assert "bilibili" in decoded, f"解码结果应含 bilibili: {decoded!r}"


@case
def test_image_baidu_url_decode() -> None:
    """百度替换加密 URL 解码：多字符标记 + 单字符表 + 特征判定。"""
    # 真机日志样本（第七轮 search_image 输出的来源页）→ 百度百科
    raw = "ippr_z2C$qAzdH3FAzdH3Fkwthj_z&e3Bkwt17_z&e3Bv54"
    decoded = _decode_baidu_url(raw)
    assert decoded == "http://baike.baidu.com", f"解码结果异常: {decoded!r}"

    # 单字符表验证（t4f->ims, w6ptg2nmc->arting365，真实设计网站）
    sample2 = _decode_baidu_url("ippr_z2C$qAzdH3FAzdH3Ft4f_z&e3Bw6ptg2nmc_z&e3Bv54")
    assert sample2 == "http://ims.arting365.com", f"样本2解码异常: {sample2!r}"

    # 特征判定：明文 URL 不命中，不误伤
    assert not _looks_encrypted_url("https://example.com/page")
    assert not _looks_encrypted_url("http://a.com/x?q=1")
    assert _looks_encrypted_url("ippr_z2C$qAzdH3F...")
    assert _looks_encrypted_url("xxAzdH3Fyy")
    assert _looks_encrypted_url("xx_z&e3Byy")
    assert _looks_encrypted_url("xx_z2C$qyy")

    # 解不出 http 的返回空串
    assert _decode_baidu_url("") == ""
    assert _decode_baidu_url("plain_text_no_marker") == ""
    # 明文 URL（无加密特征）直接返回空（该函数只服务加密串）
    assert _decode_baidu_url("https://example.com") == ""

    # parse_acjson 集成：source_url 应为解码后的明文
    payload = {
        "data": [
            {
                "thumbURL": "https://img0.baidu.com/it/u=1,a&fm=253",
                "fromURL": "ippr_z2C$qAzdH3FAzdH3Fkwthj_z&e3Bkwt17_z&e3Bv54",
                "fromPageTitle": "样本",
            },
            {
                "thumbURL": "https://img0.baidu.com/it/u=2,b&fm=253",
                "fromURL": "https://plain.example.com/page",  # 明文 fromURL（未来兼容）
                "fromPageTitle": "明文",
            },
        ]
    }
    results = parse_acjson(payload)
    assert len(results) == 2
    assert results[0].source_url == "http://baike.baidu.com", results[0].source_url
    assert results[1].source_url == "https://plain.example.com/page"


@case
def test_image_engine_search_ok() -> None:
    """引擎请求参数与结果封装。"""
    client = _FakeClient(_FakeResponse(payload=_ACJSON_SAMPLE))
    engine = BaiduImageEngine(client)
    results = asyncio.run(engine.search("洛天依", 10))

    # requests[0] 是预热，requests[1] 才是搜索
    assert len(client.requests) >= 2, "缺少预热或搜索请求"
    method, url, params = client.requests[1]
    assert url.endswith("/search/acjson"), f"端点异常: {url}"
    # tn=resultjson_com + ipn=rj 才返回 JSON；tn=baiduimage 会返回「页面不存在」HTML
    # （真机翻车证据：v0.3.0 首版用 baiduimage，三连报「返回结构异常（非 JSON）」）
    assert params["tn"] == "resultjson_com", f"tn 参数异常: {params['tn']}"
    assert params["ipn"] == "rj"
    assert params["word"] == "洛天依"
    assert params["queryWord"] == "洛天依"
    assert params["lm"] == -1

    assert len(results) == 2
    assert all(r.engine == "baidu_image" for r in results)


@case
def test_image_engine_error_branches() -> None:
    """HTTP 400 / 非 JSON / data 为空 → 各抛 EngineError 且为中文。"""
    for bad in (
        _FakeResponse(payload=None, status_code=403),
        _FakeResponse(text="not json"),
        _FakeResponse(payload={"data": []}),
        _FakeResponse(payload={"status": "安全验证", "data": []}),
    ):
        client = _FakeClient(bad)
        try:
            asyncio.run(BaiduImageEngine(client).search("测试", 5))
        except EngineError as exc:
            assert str(exc), "错误信息为空"
        else:
            raise AssertionError(f"应抛 EngineError，实际没有：{bad!r}")


@case
def test_image_engine_antiflag_cooldown() -> None:
    """antiFlag 风控响应 → 抛 EngineError 并进入冷却，冷却期内快速失败不发请求。

    真机/开发机实测：百度 acjson 对连续请求返回 200 + JSON + antiFlag:1，
    无 data 字段，必须靠 payload 内容识别。
    """
    client = _FakeClient(
        _FakeResponse(payload={"antiFlag": 1, "message": "Forbid spider access"})
    )
    engine = BaiduImageEngine(client)
    try:
        asyncio.run(engine.search("测试", 5))
    except EngineError as exc:
        assert "反爬" in str(exc), f"错误信息未提示反爬: {exc}"
    else:
        raise AssertionError("antiFlag 响应应抛 EngineError")

    # 冷却期内再搜：直接快速失败，且 client 不应收到新的搜索请求
    warmed_requests = len(client.requests)
    try:
        asyncio.run(engine.search("测试", 5))
    except EngineError as exc:
        assert "恢复" in str(exc) or "稍后" in str(exc), f"冷却期错误信息异常: {exc}"
    else:
        raise AssertionError("冷却期内应快速失败")
    assert len(client.requests) == warmed_requests, "冷却期内不应发出真实请求"


@case
def test_image_engine_cooldown_resets_on_success() -> None:
    """冷却结束后请求恢复，成功结果应重置冷却状态。"""
    responses = [
        _FakeResponse(payload={"antiFlag": 1, "message": "Forbid spider access"}),
        _FakeResponse(payload=_ACJSON_SAMPLE),
    ]

    class _QueueClient(_FakeClient):
        """按顺序返回多个响应的 Fake 客户端。"""

        def __init__(self, queue: list) -> None:
            super().__init__(queue[0])
            self._queue = queue
            self._idx = 0

        async def get(self, url, params=None, **kwargs):  # noqa: D401
            self.requests.append(("GET", url, params))
            response = self._queue[min(self._idx, len(self._queue) - 1)]
            self._idx += 1
            return response

    client = _QueueClient(responses)
    engine = BaiduImageEngine(client)

    try:
        asyncio.run(engine.search("测试", 5))
    except EngineError:
        pass  # 预期：第一发命中风控

    # 模拟冷却期结束
    engine._forbid_until = 0.0
    results = asyncio.run(engine.search("测试", 5))
    assert len(results) == 2, "冷却结束后应正常搜索"
    assert engine._forbid_until == 0.0, "成功后不应残留冷却状态"


@case
def test_image_user_count_parse() -> None:
    """用户原话数量词解析 + _resolve_image_count 优先级。

    真机教训（第九轮日志）：用户说「发一张」，Planner 自作主张传 count=3，
    发了 3 张。LLM 传参不可信，必须由代码从原话兜底。
    """
    plugin = _make_plugin_for_count()
    parse = plugin._parse_user_count
    assert parse("发一张乐正绫的图片") == 1
    assert parse("来两张洛天依") == 2
    assert parse("发两张") == 2
    assert parse("来 3 张图") == 3
    assert parse("发十一张") == 11
    assert parse("发二十张") == 20
    # 模糊数量 → 0（用默认）
    assert parse("发点洛天依的图片") == 0
    assert parse("再来几张") == 0
    assert parse("") == 0

    resolve = plugin._resolve_image_count
    # 优先级 1：用户原话明确数量 > LLM 乱传的 count
    assert resolve(3, "发一张乐正绫的图片") == 1, "用户说一张却被 LLM 传 3 覆盖"
    assert resolve(4, "来两张") == 2
    # 优先级 2：无原话数量时尊重 LLM 传参
    assert resolve(3, "发点图") == 3
    # 优先级 3：都没有 → 默认值
    assert resolve(0, "") == plugin.config.image.default_count
    # clamp 仍生效
    assert resolve(99, "发二十张") == plugin.config.image.max_count


def _make_plugin_for_count():
    """构造最小插件实例供 _parse_user_count / _resolve_image_count 测试。"""
    from fakehost import (
        bind_context,
        build_context,
        get_default_config,
        load_plugin_module,
    )

    plugin_root = pathlib.Path(__file__).resolve().parent.parent
    plugin_mod = load_plugin_module(plugin_root, module_name="websearch_under_test")
    plugin = plugin_mod.create_plugin()
    ctx = build_context("test.count")
    bind_context(plugin, ctx, get_default_config(getattr(type(plugin), "config_model", None)))
    return plugin


@case
def test_image_engine_warms_once() -> None:
    """首次 search 前有一次对 image.baidu.com 首页的 GET，之后不再预热。"""
    client = _FakeClient(_FakeResponse(payload=_ACJSON_SAMPLE))
    engine = BaiduImageEngine(client)

    asyncio.run(engine.search("洛天依", 5))
    assert client.requests[0][1] == "https://image.baidu.com/", "首次未预热"
    first_count = len(client.requests)

    asyncio.run(engine.search("洛天依", 5))
    assert len(client.requests) == first_count + 1, "第二次搜索不应再预热"


class _FakeStreamResponse:
    """伪装 httpx stream 上下文。"""

    def __init__(self, *, status_code=200, content_type="image/png",
                 content: bytes = b"", final_url="https://93.184.216.34/x.png",
                 location: str = ""):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        if location:
            self.headers["location"] = location
        self.url = final_url
        self._content = content

    async def aiter_bytes(self):
        yield self._content


class _FakeStreamClient:
    """伪装带 stream 的 httpx.AsyncClient。

    支持多响应队列（按请求次序弹出；耗尽后复用最后一个），供重定向
    逐跳跟随场景使用。
    """

    is_closed = False  # ImageDownloader._ensure_client 会检查该属性

    def __init__(self, *responses: _FakeStreamResponse) -> None:
        self._responses = list(responses)
        self.requests: list = []

    def stream(self, method, url, **kwargs):
        self.requests.append((method, url))
        if len(self._responses) > 1:
            return _FakeStreamCtx(self._responses.pop(0))
        return _FakeStreamCtx(self._responses[0])


class _FakeStreamCtx:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


def _make_downloader(client: _FakeStreamClient, **overrides) -> ImageDownloader:
    """构造注入 FakeClient 的下载器。"""
    downloader = ImageDownloader(
        max_download_bytes=overrides.pop("max_download_bytes", 1024 * 1024),
        max_send_bytes=overrides.pop("max_send_bytes", 512 * 1024),
    )
    downloader._client = client  # noqa: SLF001 - 测试注入
    return downloader


@case
def test_image_downloader_ssrf() -> None:
    """内网/保留地址在发请求前即被拒。"""
    for url in ("http://127.0.0.1/x.png", "http://169.254.169.254/x.jpg"):
        client = _FakeStreamClient(_FakeStreamResponse())
        downloader = _make_downloader(client)
        outcome = asyncio.run(downloader.download(url))
        assert not outcome.ok, f"{url} 不应下载成功"
        assert outcome.error, "应给出中文原因"
        assert not client.requests, "SSRF 拦截不应发出任何请求"


@case
def test_image_downloader_redirect_guard() -> None:
    """302 落地到内网应被逐跳校验拦截，且拦截前最多只发两跳请求。"""
    client = _FakeStreamClient(
        _FakeStreamResponse(
            status_code=302, content_type="text/html",
            location="http://192.168.1.10/evil.png",
        )
    )
    downloader = _make_downloader(client)
    outcome = asyncio.run(downloader.download("https://example.com/redirect"))
    assert not outcome.ok
    assert "重定向目标被拒绝" in outcome.error, f"原因异常: {outcome.error}"
    assert len(client.requests) == 1, (
        f"被拒的跳转目标不应再发请求: {client.requests}"
    )

    # 变体：跳进内网再跳回公网（只校验 final_url 的旧方案拦不住）
    client2 = _FakeStreamClient(
        _FakeStreamResponse(
            status_code=302, content_type="text/html",
            location="http://10.0.0.5/mid.png",
        )
    )
    outcome2 = asyncio.run(downloader.download("https://example.com/redirect2"))
    assert not outcome2.ok
    assert "重定向目标被拒绝" in outcome2.error, f"中间跳内网未被拦截: {outcome2.error}"


@case
def test_image_downloader_redirect_follow_ok() -> None:
    """302 跳到公网合法图片应正常跟随并下载成功。"""
    client = _FakeStreamClient(
        _FakeStreamResponse(
            status_code=302, content_type="text/html",
            location="https://93.184.216.34/real.png",
        ),
        _FakeStreamResponse(content=_PNG_1PX),
    )
    downloader = _make_downloader(client)
    outcome = asyncio.run(downloader.download("https://example.com/jump"))
    assert outcome.ok, f"合法重定向应下载成功: {outcome.error}"
    assert outcome.data == _PNG_1PX
    assert len(client.requests) == 2, f"应发起两跳请求: {client.requests}"
    assert client.requests[1][1] == "https://93.184.216.34/real.png"


@case
def test_image_downloader_content_type() -> None:
    """content-type 非 image/* 应拒绝。"""
    client = _FakeStreamClient(
        _FakeStreamResponse(content_type="text/html", content=b"<html>error</html>")
    )
    downloader = _make_downloader(client)
    outcome = asyncio.run(downloader.download("https://example.com/x.png"))
    assert not outcome.ok
    assert "非图片内容" in outcome.error, f"原因异常: {outcome.error}"


@case
def test_image_downloader_size_cap() -> None:
    """流式体积超下载上限应中止并报中文原因。"""
    big = _PNG_1PX + b"\x00" * (2 * 1024 * 1024)
    client = _FakeStreamClient(_FakeStreamResponse(content=big))
    downloader = _make_downloader(client, max_download_bytes=1024 * 1024)
    outcome = asyncio.run(downloader.download("https://example.com/big.png"))
    assert not outcome.ok
    assert "下载上限" in outcome.error, f"原因异常: {outcome.error}"


@case
def test_image_downloader_magic_bytes() -> None:
    """content-type 是 image/jpeg 但 body 是 HTML → 魔数校验拦截。"""
    client = _FakeStreamClient(
        _FakeStreamResponse(content_type="image/jpeg", content=b"<html>fake</html>")
    )
    downloader = _make_downloader(client)
    outcome = asyncio.run(downloader.download("https://example.com/fake.jpg"))
    assert not outcome.ok
    assert "不是有效图片" in outcome.error, f"原因异常: {outcome.error}"


@case
def test_image_downloader_send_cap() -> None:
    """合法 PNG 但超过发送上限应拒图。"""
    big_png = _PNG_1PX + b"\x00" * (600 * 1024)
    client = _FakeStreamClient(_FakeStreamResponse(content=big_png))
    downloader = _make_downloader(
        client,
        max_download_bytes=1024 * 1024,
        max_send_bytes=512 * 1024,
    )
    outcome = asyncio.run(downloader.download("https://example.com/big.png"))
    assert not outcome.ok
    assert "发送上限" in outcome.error, f"原因异常: {outcome.error}"


@case
def test_image_downloader_ok() -> None:
    """正常 PNG 应下载成功。"""
    client = _FakeStreamClient(_FakeStreamResponse(content=_PNG_1PX))
    downloader = _make_downloader(client)
    outcome = asyncio.run(downloader.download("https://93.184.216.34/ok.png"))
    assert outcome.ok, f"应成功: {outcome.error}"
    assert outcome.data == _PNG_1PX
    # 新实现手动跟随重定向：无重定向时 final_url 即请求 URL 本身
    assert outcome.final_url == "https://93.184.216.34/ok.png"


# ------------------------------------------------------------------ 运行器


def main() -> int:
    """逐个跑测试并汇总。"""
    passed = 0
    failed = 0
    for fn in _TESTS:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
        else:
            passed += 1

    print("-" * 60)
    print(f"单测结果：PASS {passed} / FAIL {failed} / 合计 {passed + failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
