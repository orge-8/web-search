"""联网探测脚本：在目标机器（开发机或真机）上实测各搜索引擎与抓取链路是否可用。

运行: python tests/network_probe.py [关键词]

它会逐个尝试引擎、报告耗时与解析出的前几条标题，并实测一次网页抓取。
用于回答一个本地单测回答不了的问题：**这台机器的网络能不能搜到东西。**

所有外呼形态与插件运行时一致（同样的 UA、超时、代理策略），
所以结论可以直接外推到插件在真机上的表现。
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import time

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

import httpx  # noqa: E402

from core.engines import (  # noqa: E402
    BingEngine,
    DuckDuckGoEngine,
    EngineError,
    SearxngEngine,
    TavilyEngine,
)
from core.fetcher import Fetcher, is_safe_url  # noqa: E402

DEFAULT_KEYWORD = "MaiBot QQ机器人"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _build_client(timeout: float, proxy: str = "") -> httpx.AsyncClient:
    """构造与插件一致的 httpx 客户端。"""
    kwargs = {
        "timeout": httpx.Timeout(timeout, connect=min(10.0, timeout)),
        "follow_redirects": True,
        "headers": {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        "trust_env": True,
    }
    if proxy:
        try:
            return httpx.AsyncClient(proxy=proxy, **kwargs)
        except TypeError:
            return httpx.AsyncClient(proxies=proxy, **kwargs)
    return httpx.AsyncClient(**kwargs)


async def _probe_engine(engine, keyword: str) -> bool:
    """探测单个引擎。"""
    started = time.monotonic()
    try:
        results = await engine.search(keyword, 3)
    except EngineError as exc:
        elapsed = (time.monotonic() - started) * 1000
        print(f"  [失败] {engine.name}：{exc}（{elapsed:.0f} ms）")
        return False
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.monotonic() - started) * 1000
        print(f"  [异常] {engine.name}：{type(exc).__name__}: {exc}（{elapsed:.0f} ms）")
        return False

    elapsed = (time.monotonic() - started) * 1000
    print(f"  [成功] {engine.name}：{len(results)} 条 / {elapsed:.0f} ms")
    for item in results[:3]:
        print(f"        · {(item.title or '(无标题)')[:50]}")
        print(f"          {item.url[:80]}")
    return True


async def _probe_fetch() -> None:
    """探测网页抓取链路。"""
    print("\n[抓取链路]")
    probe_urls = ["https://example.com/", "https://www.iana.org/help/example-domains"]
    fetcher = Fetcher(timeout=20.0, user_agent=UA)
    try:
        for url in probe_urls:
            allowed, reason = is_safe_url(url)
            if not allowed:
                print(f"  [跳过] {url}：{reason}")
                continue
            outcome = await fetcher.fetch(url)
            if outcome.ok:
                print(
                    f"  [成功] {url} → {len(outcome.text)} 字符，标题 {outcome.title!r}"
                )
                preview = outcome.text[:120].replace("\n", " ")
                print(f"          预览: {preview}…")
            else:
                print(f"  [失败] {url}：{outcome.error}")
    finally:
        await fetcher.aclose()


async def main() -> int:
    """入口。"""
    keyword = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KEYWORD
    proxy = os.environ.get("WEBSEARCH_PROXY", "")

    print("=" * 68)
    print("联网搜索插件 · 链路探测")
    print("=" * 68)
    print(f"关键词: {keyword}")
    print(f"代理: {proxy or '(跟随环境变量 ' + (os.environ.get('https_proxy') or '未设置') + ')'}")
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if os.environ.get(var):
            print(f"  环境变量 {var}={os.environ[var]}")

    client = _build_client(20.0, proxy)
    engines = [
        DuckDuckGoEngine(client, language="zh-CN"),
        BingEngine(client, language="zh-CN"),
    ]
    searxng_url = os.environ.get("SEARXNG_BASE_URL", "")
    if searxng_url:
        engines.append(SearxngEngine(client, base_url=searxng_url, language="zh-CN"))
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        engines.append(TavilyEngine(client, api_key=tavily_key, language="zh-CN"))

    print("\n[搜索引擎]")
    working: list[str] = []
    try:
        for engine in engines:
            if await _probe_engine(engine, keyword):
                working.append(engine.name)
    finally:
        await client.aclose()

    await _probe_fetch()

    print("\n" + "=" * 68)
    if working:
        print(f"结论：可用引擎 = {', '.join(working)}")
        print(f"建议把 search.order 配置为：{working}")
        return 0
    print("结论：所有引擎都不通。")
    print("处置建议：")
    print("  1) 若本机需要代理才能出网，在插件配置 search.proxy 里填代理地址；")
    print("  2) 配置 search.order 追加 'tavily' 并填写 engines.tavily_api_key；")
    print("  3) 有自建 SearXNG 时填 engines.searxng_base_url 并加入 order。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
