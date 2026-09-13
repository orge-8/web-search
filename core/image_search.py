"""百度图片搜索（acjson JSON 端点）。

与网页搜索引擎（engines.py）刻意分离：

* 返回结构是 :class:`ImageResult`（thumb/hover/尺寸/来源页），与
  ``SearchResult`` 不同源，混入 EngineChain 会污染相关性检测路径；
* 图片搜索当前只有百度一个引擎，没有降级链可言；
* 协议形状与 ``SearchEngine`` 保持一致（``name`` + ``async search(query, limit)``），
  未来接 Bing 图片等引擎时可以平移。

acjson 端点返回的 ``data`` 数组每项常见字段：

* ``thumbURL`` —— 缩略图直链（首选下载源，小且稳）
* ``hoverURL`` —— 悬浮预览图直链（回退下载源）
* ``objURL`` —— 原始图地址，可能 URL 编码或 base64 混淆，best-effort 解码留元数据
* ``fromPageTitle`` —— 所在页面标题（含 HTML 实体/标签，需清洗）
* ``fromURL`` —— 来源页地址（给 LLM 引用）
* ``width`` / ``height`` —— 原图尺寸（可能为 0 表示未知）
"""

import asyncio
import base64
import binascii
import html
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import httpx

from .engines import _BAIDU_HEADERS, EngineError

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")

# acjson 在真实浏览器里是 XHR 请求（搜索页 JS 发起），指纹与文档导航不同：
# 带 Referer + X-Requested-With + JSON Accept。用 document 导航头打 XHR 端点
# 属于指纹错位，会提高被反爬盯上的概率（开发机实测连续请求即被 antiFlag:1 拦截）。
_ACJSON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://image.baidu.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


# 百度 fromURL/objURL 的替换加密解码表（多来源交叉验证的公开算法）：
# 1) 多字符标记替换（区分大小写，必须在 lower 之前做，否则 AzdH3F 变形失配）
# 2) 单字符替换表：in_table -> out_table（str.maketrans 一次建表）
# ⚠️ 不能对任意 URL 盲用——0-9a-v 字符会被错误翻译；必须先识别加密特征
#    （ippr 前缀或 AzdH3F 标记）再动手。
_BAIDU_STR_TABLE = {
    "ippr": "http",  # 协议头
    "_z2C$q": ":",
    "_z&e3B": ".",
    "AzdH3F": "/",
}
_BAIDU_CHAR_TABLE = str.maketrans(
    "0123456789abcdefghijklmnopqrstuvw",
    "7dgjmoru140852vsnkheb963wtqplifca",
)


def _looks_encrypted_url(raw: str) -> bool:
    """判断是否为百度替换加密串（ippr 前缀或任一标记存在）。"""
    return raw.startswith("ippr") or any(
        token in raw for token in ("AzdH3F", "_z2C$q", "_z&e3B")
    )


def _decode_baidu_url(raw: str) -> str:
    """解码百度替换加密 URL。解不出 http 开头的结果就返回空串。

    顺序关键（对齐原始 PHP/Java 算法）：
    1. 先做多字符标记替换（区分大小写，lower 之前，否则 AzdH3F 变形失配），
       此时协议头 ippr 已还原成 http；
    2. lower 后跳过前 4 个字符（http，**绝不能参与单字符翻译**，否则变成 kiit）；
    3. 剩余部分做单字符 translate，拼回协议头；
    4. best-effort unquote（fromURL 偶混 %E9 这类中文编码）。
    """
    raw = (raw or "").strip()
    if not raw or not _looks_encrypted_url(raw):
        return ""
    for token, repl in _BAIDU_STR_TABLE.items():
        raw = raw.replace(token, repl)
    lowered = raw.lower()
    if not lowered.startswith("http"):
        return ""
    decoded = "http" + lowered[4:].translate(_BAIDU_CHAR_TABLE)
    try:
        decoded = unquote(decoded)
    except Exception:  # pragma: no cover - unquote 几乎不抛，兜底而已
        pass
    return decoded if decoded.startswith(("http://", "https://")) else ""


def _clean_text(text: str) -> str:
    """HTML 实体解码 + 去标签 + 压缩空白。"""
    decoded = html.unescape(text or "")
    stripped = _TAG_RE.sub(" ", decoded)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


@dataclass(slots=True)
class ImageResult:
    """单条图片搜索结果。"""

    title: str = ""        # fromPageTitle 清洗后
    thumb_url: str = ""    # thumbURL（首选下载源）
    hover_url: str = ""    # hoverURL（回退下载源）
    image_url: str = ""    # objURL best-effort 解码结果（仅元数据，不下载）
    source_url: str = ""   # fromURL（来源页，给 LLM 引用）
    width: int = 0
    height: int = 0
    engine: str = ""

    def to_prompt_line(self, index: int) -> str:
        """渲染给 LLM 的单行描述。"""
        text = f"[{index}] {self.title or '(无描述)'}"
        if self.source_url:
            text += f"（来源页：{self.source_url}）"
        return text


def _decode_objurl(raw: str) -> str:
    """objURL best-effort 解码。

    百度返回的 objURL 可能是明文 http、URL 编码串、base64 混淆串或
    替换加密串（ippr_z2C$q...），按顺序尝试；只有解出 http 开头的结果
    才有效，否则返回空串（调用方只当元数据用）。
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    if _looks_encrypted_url(raw):
        decoded = _decode_baidu_url(raw)
        if decoded:
            return decoded
    candidate = unquote(raw)
    if candidate.startswith(("http://", "https://")):
        return candidate
    try:
        padding = "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(raw + padding).decode("utf-8", "replace")
        if decoded.startswith(("http://", "https://")):
            return decoded
    except (ValueError, binascii.Error, UnicodeDecodeError):
        pass
    return ""


def _as_int(value: Any) -> int:
    """宽松转 int（百度 JSON 里尺寸偶尔是字符串）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_acjson(payload: dict[str, Any]) -> list[ImageResult]:
    """解析 acjson 载荷为 ImageResult 列表（纯函数，便于离线单测）。

    - 跳过空 dict（百度返回末尾常带空项）
    - 无 thumbURL 的项直接跳过（没有可用下载源）
    - 按 thumb_url 去重
    """
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    results: list[ImageResult] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict) or not item:
            continue
        thumb = str(item.get("thumbURL") or "").strip()
        if not thumb:
            continue
        if thumb in seen:
            continue
        seen.add(thumb)
        from_url = str(item.get("fromURL") or "").strip()
        # fromURL 是百度替换加密串（ippr_z2C$q...）；解不出明文就保留原值，
        # 万一未来百度改回明文也不受影响（_looks_encrypted_url 判定不命中）
        decoded_from = _decode_baidu_url(from_url)
        results.append(
            ImageResult(
                title=_clean_text(str(item.get("fromPageTitle") or "")),
                thumb_url=thumb,
                hover_url=str(item.get("hoverURL") or "").strip(),
                image_url=_decode_objurl(str(item.get("objURL") or "")),
                source_url=decoded_from or from_url,
                width=_as_int(item.get("width")),
                height=_as_int(item.get("height")),
                engine="baidu_image",
            )
        )
    return results


class BaiduImageEngine:
    """百度图片搜索（acjson JSON 端点）。

    与 ``BaiduEngine`` 同源的风控对策：首次搜索前访问
    ``https://image.baidu.com/`` 取 Cookie（BAIDUID 对 image 域生效，
    与 www.baidu.com 不同域需单独预热），请求带完整浏览器头。

    百度反爬踩坑（真机 + 开发机双重实测）：acjson 端点对同 IP 连续
    请求的风控响应是 **HTTP 200 + application/json + ``{"antiFlag":1,
    "message":"Forbid spider access"}``**——没有 ``data`` 字段，绕过了
    HTTP 状态码和 JSON 解析两道防线，只能靠 payload 内容识别。命中后
    引擎进入冷却期，冷却期内直接快速失败不再打百度（避免加重风控），
    冷却结束自动恢复。
    """

    name = "baidu_image"
    _ENDPOINT = "https://image.baidu.com/search/acjson"
    _HOME = "https://image.baidu.com/"
    _COOLDOWN_SECONDS = 90.0

    def __init__(self, client: httpx.AsyncClient, *, language: str = "zh-CN") -> None:
        self._client = client
        self.language = language or "zh-CN"
        self._warmed = False
        self._warm_lock = asyncio.Lock()
        self._forbid_until = 0.0  # time.monotonic() 之前禁止再请求

    async def _ensure_warm(self) -> None:
        """首次搜索前先访问图片站首页取 Cookie。"""
        if self._warmed:
            return
        async with self._warm_lock:
            if self._warmed:
                return
            try:
                await self._client.get(self._HOME, headers=_BAIDU_HEADERS)
            except httpx.HTTPError:
                pass  # 预热失败不阻断，让真正的搜索自己去试
            self._warmed = True

    async def search(self, query: str, limit: int) -> list[ImageResult]:
        """执行图片搜索。失败时抛 EngineError（中文原因）。

        参数踩坑：``tn=baiduimage`` 是网页版参数，acjson 对它返回 67KB 的
        「页面不存在」HTML；JSON 输出必须用 ``tn=resultjson_com`` + ``ipn=rj``。
        """
        await self._ensure_warm()
        remaining = self._forbid_until - time.monotonic()
        if remaining > 0:
            raise EngineError(
                f"刚触发百度反爬拦截，约 {int(remaining / 10) * 10 + 10} 秒后自动恢复，请稍后再试"
            )
        try:
            response = await self._client.get(
                self._ENDPOINT,
                params={
                    "tn": "resultjson_com",
                    "ipn": "rj",
                    "word": query,
                    "queryWord": query,
                    "pn": 0,
                    "rn": max(limit, 20),
                    "ie": "utf-8",
                    "lm": -1,  # 全时间段
                },
                headers=_ACJSON_HEADERS,
            )
        except httpx.TimeoutException as exc:
            raise EngineError("请求超时") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"网络错误：{exc}") from exc

        if response.status_code >= 400:
            raise EngineError(f"返回 HTTP {response.status_code}")

        body = response.text
        if "安全验证" in body or "wappass.baidu.com" in body:
            raise EngineError("触发百度安全验证（请求过频），稍后会自动恢复")

        try:
            payload = response.json()
        except ValueError as exc:
            raise EngineError("返回结构异常（非 JSON）") from exc
        if not isinstance(payload, dict):
            raise EngineError("返回结构异常（非对象）")

        if payload.get("antiFlag") or "Forbid spider access" in str(payload.get("message") or ""):
            self._forbid_until = time.monotonic() + self._COOLDOWN_SECONDS
            raise EngineError(
                "触发百度反爬拦截（请求过频），约 90 秒后自动恢复，请稍后再试"
            )

        results = parse_acjson(payload)
        if not results:
            raise EngineError("未解析出任何图片结果")
        return results[:limit]
