"""冒烟测试：不启动 MaiBot、不联网，用 FakeHost 跑完插件生命周期与三条主链路。

运行: python tests/smoke_test.py

联网部分（搜索引擎、网页抓取）用 stub 替换，因此这个测试是**确定性**的：
它验证的是插件自身的编排逻辑（配置读取、降级处理、prompt 组装、结果渲染、
资源释放），而不是外部网站的可用性——后者属于真机验证的范畴。
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent


class _StubEngine:
    """固定返回两条搜索结果的假引擎。"""

    name = "stub"
    calls = 0

    def __init__(self, results):
        self._results = results

    async def search(self, query: str, limit: int):
        type(self).calls += 1
        return list(self._results)[:limit]


class _StubChain:
    """包装 stub 引擎，接口与 EngineChain 一致。"""

    def __init__(self, results):
        self.engines = [_StubEngine(results)]
        self.names = ["stub"]
        self.max_attempts = 1
        self.failures: list[tuple[str, str]] = []

    async def search(self, query: str, limit: int):
        results = await self.engines[0].search(query, limit)
        return results, list(self.failures)


class _StubFetcher:
    """固定返回正文的假抓取器。"""

    def __init__(self, text: str = "这是抓取到的正文。" * 20):
        self._text = text
        self.closed = False
        self.calls: list[str] = []

    async def fetch(self, url: str):
        self.calls.append(url)
        return SimpleNamespace(
            ok=True,
            url=url,
            final_url=url,
            status=200,
            content_type="text/html",
            title="假标题",
            text=self._text,
            error="",
            from_cache=False,
        )

    async def aclose(self) -> None:
        self.closed = True


_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd400000000"
    "49454e44ae426082"
)


class _StubImageEngine:
    """固定返回图片候选的假图片引擎。"""

    name = "baidu_image"

    def __init__(self, results):
        self._results = results

    async def search(self, query: str, limit: int):
        return list(self._results)[:limit]


class _StubImageDownloader:
    """按预设脚本返回下载结果的假图片下载器。"""

    def __init__(self, outcomes):
        # 每个 URL 依次弹出结果；耗尽后复用最后一个
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    async def download(self, url: str):
        self.calls.append(url)
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        return outcome

    async def aclose(self) -> None:
        pass


def _img_ok(data: bytes = _PNG_1PX):
    return SimpleNamespace(
        ok=True, data=data, content_type="image/png",
        final_url="https://img.example.com/final.png", size=len(data), error="",
    )


def _img_fail(reason: str):
    return SimpleNamespace(
        ok=False, data=b"", content_type="", final_url="", size=0, error=reason,
    )


def main() -> int:
    try:
        import maibot_sdk  # noqa: F401
    except Exception:
        print("SKIP: 未安装 maibot-plugin-sdk，跳过冒烟测试（这不代表通过）")
        return 0

    from fakehost import (
        FakeHost,
        bind_context,
        build_context,
        get_default_config,
        load_plugin_module,
    )

    module = load_plugin_module(PLUGIN_DIR)
    plugin = module.create_plugin()
    host = FakeHost(plugin_id="org.orge-8.web-search")
    ctx = build_context("org.orge-8.web-search", rpc_call=host.rpc_call)
    bind_context(plugin, ctx, get_default_config(getattr(type(plugin), "config_model", None)))

    SearchResult = module.SearchResult

    async def run() -> None:
        # ---------- 生命周期 ----------
        await plugin.on_load()
        assert plugin.config.plugin.enabled is True, "默认配置未生效"
        assert plugin._chain is not None, "on_load 未建立引擎链"
        assert plugin.config.plugin.config_version, "config_version 缺失"

        # ---------- SSRF 防护（走真实校验，不产生网络请求）----------
        blocked = await plugin.tool_fetch_page(url="http://127.0.0.1:8001/")
        assert "失败" in blocked["content"], f"内网地址未被拦截: {blocked}"
        assert "拒绝" in blocked["content"], f"拦截原因不明确: {blocked}"

        blocked_meta = await plugin.tool_fetch_page(url="http://169.254.169.254/latest/meta-data/")
        assert "失败" in blocked_meta["content"], f"云元数据地址未被拦截: {blocked_meta}"

        blocked_scheme = await plugin.tool_fetch_page(url="file:///etc/passwd")
        assert "失败" in blocked_scheme["content"], f"非法协议未被拦截: {blocked_scheme}"

        # ---------- 状态命令 ----------
        ok, resp, intercept = await plugin.cmd_websearch(
            matched_groups={"action": "status"}, stream_id="fake-stream"
        )
        assert ok is True and intercept is True, f"命令返回异常: {(ok, resp, intercept)}"
        assert "send.text" in [cap for cap, _ in host.calls], "命令未显式发送消息"
        assert "行为自检" in resp, "状态输出缺少行为自检标记（真机排查依赖它）"
        assert "引擎链" in resp, "状态输出缺少引擎链信息"
        assert "可用模型任务名" in resp, "状态输出缺少模型任务名列表"

        # ---------- 完整搜索链路（stub 引擎 + stub 抓取器）----------
        plugin._chain = _StubChain(
            [
                SearchResult(
                    title="结果一",
                    url="https://example.com/one",
                    snippet="摘要一",
                    engine="stub",
                ),
                SearchResult(
                    title="结果二",
                    url="https://example.com/two",
                    snippet="摘要二",
                    engine="stub",
                ),
            ]
        )
        stub_fetcher = _StubFetcher()
        plugin._fetcher = stub_fetcher

        searched = await plugin.tool_web_search(query="麦麦是什么", fetch_pages=True)
        assert isinstance(searched, dict), f"工具返回类型异常: {type(searched)}"
        content = searched.get("content") or ""
        assert content, "搜索返回空内容"
        assert "来源" in content, f"搜索结果缺少来源清单: {content}"
        assert "example.com/one" in content, f"来源链接缺失: {content}"
        assert stub_fetcher.calls, "开启 fetch_pages 后未抓取正文"
        assert "llm.generate" in [cap for cap, _ in host.calls], "未调用 LLM 总结"

        # ---------- 参数容错：空关键词 ----------
        empty = await plugin.tool_web_search(query="   ")
        assert "失败" in empty["content"], f"空关键词未返回可读错误: {empty}"

        # ---------- 快搜模式：不抓正文 ----------
        before = len(stub_fetcher.calls)
        await plugin.tool_web_search(query="快搜", fetch_pages=False)
        assert len(stub_fetcher.calls) == before, "fetch_pages=False 时仍在抓取正文"

        # ---------- fetch_page 正常路径 ----------
        page = await plugin.tool_fetch_page(url="https://example.com/article")
        assert page.get("content"), "fetch_page 返回空内容"
        assert "原文总长" in page["content"], f"fetch_page 缺少长度标注: {page['content'][:80]}"

        # ---------- 分页读取 ----------
        paged = await plugin.tool_fetch_page(
            url="https://example.com/article", start_char=0, end_char=20
        )
        assert paged.get("content"), "分页读取返回空内容"

        # ---------- 引擎链路实测命令（同样走 stub，不联网）----------
        ok_test, resp_test, _ = await plugin.cmd_websearch(
            matched_groups={"action": "test", "arg": "麦麦"}, stream_id="fake-stream"
        )
        assert ok_test is True, "test 命令返回异常"
        assert "实测" in resp_test, f"test 输出异常: {resp_test[:80]}"
        assert "全部可用" in resp_test, f"test 未给出配置结论: {resp_test[-150:]}"

        # ---------- 清缓存命令 ----------
        ok_clear, resp_clear, _ = await plugin.cmd_websearch(
            matched_groups={"action": "clear"}, stream_id="fake-stream"
        )
        assert ok_clear is True, "清缓存命令返回异常"
        assert "清空" in resp_clear or "未启用" in resp_clear, f"清缓存响应异常: {resp_clear}"

        # ---------- 图片搜索：正常路径（stub 引擎 + stub 下载器）----------
        ImageResult = module.ImageResult
        plugin._image_engine = _StubImageEngine(
            [
                ImageResult(
                    title="候选一", thumb_url="https://img/thumb1.jpg",
                    hover_url="https://img/hover1.jpg",
                    source_url="https://source.example.com/p1",
                    width=800, height=600, engine="baidu_image",
                ),
                ImageResult(
                    title="候选二", thumb_url="https://img/thumb2.jpg",
                    source_url="https://source.example.com/p2",
                    engine="baidu_image",
                ),
            ]
        )
        # 两张图字节不同（内容 hash 去重要求真实场景下图片互异）
        stub_downloader = _StubImageDownloader(
            [_img_ok(data=_PNG_1PX + b"\x01"), _img_ok(data=_PNG_1PX + b"\x02")]
        )
        plugin._image_downloader = stub_downloader
        host.calls.clear()

        sent = await plugin.tool_search_image(query="麦麦", stream_id="fake-stream")
        content = sent.get("content") or ""
        assert "已向当前聊天发送" in content, f"返回缺少发送确认: {content[:120]}"
        assert "候选一" in content, "返回缺少图片描述"
        assert "来源页" in content, "返回缺少来源页"

        image_calls = host.calls_of("send.image")
        assert image_calls, "未调用 send.image"
        for kw in image_calls:
            b64 = kw.get("image_base64") or ""
            assert isinstance(b64, str) and b64, "send.image 参数应为非空字符串"
            assert not b64.startswith("data:"), "send.image 收到带 data: 前缀的 base64（实测会失败）"
        assert len(image_calls) == 2, f"应发送 2 张，实际 {len(image_calls)}"

        # ---------- count 截断：上限 clamp ----------
        host.calls.clear()
        plugin._image_engine = _StubImageEngine(
            [ImageResult(title=f"图{i}", thumb_url=f"https://img/{i}.jpg",
                         source_url="https://s.example.com", engine="baidu_image")
             for i in range(10)]
        )
        # 每次下载返回不同字节（内容 hash 去重要求图片互异；0x30+ 段避开前场景）
        plugin._image_downloader = _StubImageDownloader(
            [_img_ok(data=_PNG_1PX + bytes([0x30 + i])) for i in range(12)]
        )
        clamped = await plugin.tool_search_image(query="刷屏", count=99, stream_id="fake-stream")
        assert len(host.calls_of("send.image")) <= plugin.config.image.max_count, (
            "count 超过 max_count 未被截断"
        )
        assert clamped.get("content"), "count 截断场景返回空内容"

        # ---------- 下载全失败：降级为来源链接 ----------
        host.calls.clear()
        plugin._image_engine = _StubImageEngine(
            [ImageResult(title="死图", thumb_url="https://img/dead.jpg",
                         source_url="https://source.example.com/dead", engine="baidu_image")]
        )
        plugin._image_downloader = _StubImageDownloader([_img_fail("HTTP 404")])
        all_dead = await plugin.tool_search_image(query="不存在", stream_id="fake-stream")
        dead_content = all_dead.get("content") or ""
        assert "下载或发送失败" in dead_content, f"全失败场景缺少降级说明: {dead_content}"
        assert not host.calls_of("send.image"), "全失败时不应发送图片"

        # ---------- 无 stream_id：跳过发送 ----------
        host.calls.clear()
        plugin._image_engine = _StubImageEngine(
            [ImageResult(title="无上下文图", thumb_url="https://img/t.jpg",
                         source_url="https://s.example.com", engine="baidu_image")]
        )
        plugin._image_downloader = _StubImageDownloader([_img_ok()])
        skipped = await plugin.tool_search_image(query="测试")
        assert "未实际发送" in (skipped.get("content") or ""), (
            f"无 stream_id 场景缺少说明: {skipped}"
        )
        assert not host.calls_of("send.image"), "无 stream_id 时不应发送图片"

        # ---------- 跨请求去重：同会话再搜同词，第二次必须避开已发 ----------
        host.calls.clear()
        # 清空历史让本场景自洽（前面场景已向同一 stream 写入指纹）
        plugin._image_sent_history.clear()
        # 引擎每次返回同一批候选（模拟百度对同词返回相同 top 结果）
        plugin._image_engine = _StubImageEngine(
            [
                ImageResult(title="候选一", thumb_url="https://img/thumb1.jpg",
                            hover_url="https://img/hover1.jpg",
                            source_url="https://source.example.com/p1",
                            width=800, height=600, engine="baidu_image"),
                ImageResult(title="候选二", thumb_url="https://img/thumb2.jpg",
                            source_url="https://source.example.com/p2",
                            width=800, height=600, engine="baidu_image"),
                ImageResult(title="候选三", thumb_url="https://img/thumb3.jpg",
                            source_url="https://source.example.com/p3",
                            width=800, height=600, engine="baidu_image"),
            ]
        )
        # 每次下载都返回不同内容（用字节计数器生成不同字节串）
        download_counter = {"n": 0}

        class _CountingDownloader(_StubImageDownloader):
            async def download(self, url):
                download_counter["n"] += 1
                return _img_ok(data=_PNG_1PX + bytes([0x60 + download_counter["n"]]))

        plugin._image_downloader = _CountingDownloader([])
        host.calls.clear()
        first = await plugin.tool_search_image(query="麦麦", stream_id="fake-stream")
        first_urls = [kw.get("image_base64") for kw in host.calls_of("send.image")]
        assert len(first_urls) == 2, f"第一次应发 2 张，实际 {len(first_urls)}"

        # 第二次：同 query 同 stream。候选一二三里，一二已发，应只发候选三
        host.calls.clear()
        second = await plugin.tool_search_image(query="麦麦", stream_id="fake-stream")
        second_content = second.get("content") or ""
        second_urls = [kw.get("image_base64") for kw in host.calls_of("send.image")]
        assert "候选三" in second_content, f"第二次应发新候选三: {second_content[:150]}"
        assert "候选一" not in second_content and "候选二" not in second_content, (
            "第二次重复发送了已发过的候选"
        )
        assert len(second_urls) == 1, f"第二次应只发 1 张新图，实际 {len(second_urls)}"
        assert not set(first_urls) & set(second_urls), "两次发送内容重复"
        assert "避开" in second_content, f"缺少去重提示: {second_content[-150:]}"

        # ---------- user_request 覆盖 count：用户说一张，LLM 传 3 也只发 1 张 ----------
        # 真机教训（第九轮日志）：用户「发一张乐正绫的图片」，Planner 传 count=3 发了 3 张
        host.calls.clear()
        plugin._image_sent_history.clear()
        plugin._image_engine = _StubImageEngine(
            [
                ImageResult(title=f"绫{i}", thumb_url=f"https://img/ling{i}.jpg",
                            source_url="https://s.example.com", width=800,
                            height=600, engine="baidu_image")
                for i in range(6)
            ]
        )
        ling_counter = {"n": 0}

        class _LingDownloader(_StubImageDownloader):
            async def download(self, url):
                ling_counter["n"] += 1
                return _img_ok(data=_PNG_1PX + bytes([0x80 + ling_counter["n"]]))

        plugin._image_downloader = _LingDownloader([])
        one = await plugin.tool_search_image(
            query="乐正绫", count=3,
            user_request="发一张乐正绫的图片",
            stream_id="fake-stream",
        )
        one_urls = [kw.get("image_base64") for kw in host.calls_of("send.image")]
        assert len(one_urls) == 1, (
            f"用户说一张应只发 1 张，实际 {len(one_urls)} 张"
        )
        assert "已向当前聊天发送 1 张" in (one.get("content") or ""), (
            f"返回文案张数异常: {one.get('content', '')[:100]}"
        )

        # ---------- 空 query ----------
        empty_img = await plugin.tool_search_image(query="  ")
        assert "失败" in (empty_img.get("content") or ""), "空 query 未返回可读错误"

        # ---------- 禁用开关 ----------
        plugin.config.image.enabled = False
        disabled = await plugin.tool_search_image(query="麦麦")
        assert "未启用" in (disabled.get("content") or ""), "禁用后工具未提示"
        plugin.config.image.enabled = True

        # ---------- 卸载 ----------
        await plugin.on_unload()
        assert plugin._chain is None, "on_unload 未释放引擎链"
        assert plugin._fetcher is None, "on_unload 未释放抓取器"
        assert plugin._search_client is None, "on_unload 未释放 httpx 客户端"
        assert plugin._image_engine is None, "on_unload 未释放图片引擎"
        assert plugin._image_downloader is None, "on_unload 未释放图片下载器"

    asyncio.run(run())
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
