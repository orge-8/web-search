"""网页抓取：内容类型识别、体积与超时限制、SSRF 防护、结果缓存。

安全要点（这段代码是插件里唯一会按"用户/LLM 给的字符串"主动外呼的地方）：

* 只允许 http / https；
* 解析主机名得到的所有 IP 都必须落在公网，否则拒绝——包括 DNS 解析后的
  二次校验，避免"域名看着正常、解析到 127.0.0.1"；
* 跟随重定向后**再次**校验最终地址，避免 302 跳进内网；
* 限制响应体积与读取超时，避免被超大文件或慢速连接拖死 Runner 子进程。
"""

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .extract import extract_main_text, format_json_response

# 明确列出的受限网段。不依赖 ipaddress 的 is_private 单一判断，
# 因为它对 100.64/10（运营商级 NAT）与部分保留段的归类随版本变化。
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
    ipaddress.ip_network("2001:db8::/32"),
]

# 云元数据服务主机名，一律拒绝（SSRF 的经典目标）
_BLOCKED_HOSTS = {
    "localhost",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
}

_ALLOWED_SCHEMES = {"http", "https"}

_TEXTUAL_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
)

_CHARSET_RE = re.compile(r"charset\s*=\s*['\"]?([\w\-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*['"]?\s*([\w\-]+)""", re.IGNORECASE
)


@dataclass(slots=True)
class FetchOutcome:
    """一次抓取的结果。"""

    ok: bool = False
    url: str = ""
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    title: str = ""
    text: str = ""
    error: str = ""
    from_cache: bool = False


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """判断 IP 是否落在受限网段。"""
    for network in _BLOCKED_NETWORKS:
        if ip.version != network.version:
            continue
        if ip in network:
            return True
    # 双保险：loopback / link_local / unspecified 在任何版本都该拒
    return bool(ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast)


def _resolve_host_ips(host: str) -> list[str]:
    """解析主机名到 IP 列表（阻塞，调用方自行丢线程池）。"""
    infos = socket.getaddrinfo(host, None)
    ips: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and sockaddr[0]:
            ips.append(str(sockaddr[0]))
    return ips


def is_safe_url(url: str, allow_private: bool = False) -> tuple[bool, str]:
    """校验 URL 是否允许抓取。

    Args:
        url: 待校验地址。
        allow_private: 是否放行内网地址（默认放行即关闭防护）。

    Returns:
        tuple[bool, str]: (是否安全, 原因)。不安全时第二个元素为中文原因。
    """
    raw = (url or "").strip()
    if not raw:
        return False, "地址为空"

    try:
        parsed = urlparse(raw)
    except Exception:
        return False, "地址格式无法解析"

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return False, f"仅支持 http/https，收到：{scheme or '(空)'}"

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False, "地址缺少主机名"

    if allow_private:
        return True, ""

    if host in _BLOCKED_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        return False, f"拒绝访问本地/内网主机：{host}"

    # 主机名本身就是 IP 字面量
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        if _ip_is_blocked(literal):
            return False, f"拒绝访问内网/保留地址：{host}"
        return True, ""

    # 域名：解析后逐个校验，任一落在受限网段即拒绝
    try:
        ips = _resolve_host_ips(host)
    except socket.gaierror:
        return False, f"域名解析失败：{host}"
    except Exception as exc:  # pragma: no cover - 环境相关
        return False, f"域名解析异常：{exc}"

    if not ips:
        return False, f"域名无解析结果：{host}"

    for text_ip in ips:
        try:
            parsed_ip = ipaddress.ip_address(text_ip)
        except ValueError:
            continue
        if _ip_is_blocked(parsed_ip):
            return False, f"域名 {host} 解析到内网/保留地址 {text_ip}"

    return True, ""


def _extract_charset(content_type: str, body: bytes) -> str:
    """从响应头或 HTML meta 中猜出字符集。"""
    match = _CHARSET_RE.search(content_type or "")
    if match:
        return match.group(1)

    head = body[:4096]
    meta = _META_CHARSET_RE.search(head)
    if meta:
        try:
            return meta.group(1).decode("ascii", errors="ignore")
        except Exception:
            pass
    return ""


def _decode_body(body: bytes, content_type: str) -> str:
    """按猜测字符集解码响应体，失败时逐级兜底。"""
    candidates: list[str] = []
    guessed = _extract_charset(content_type, body)
    if guessed:
        candidates.append(guessed)
    # 中文站点常见编码，按命中概率排序
    candidates.extend(["utf-8", "gb18030", "big5", "latin-1"])

    seen: set[str] = set()
    for encoding in candidates:
        key = encoding.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


async def _check_url_allowed(
    url: str, allow_private: bool
) -> tuple[bool, str]:
    """异步版 is_safe_url（getaddrinfo 是阻塞的，丢线程池）。"""
    return await asyncio.to_thread(is_safe_url, url, allow_private)


def _is_redirect(status: int) -> bool:
    """301/302/303/307/308 视为重定向。"""
    return status in (301, 302, 303, 307, 308)


def _is_textual(content_type: str) -> bool:
    """判断内容类型是否为可读文本。"""
    ctype = (content_type or "").split(";")[0].strip().lower()
    if not ctype:
        # 服务器没给 Content-Type 时按文本尝试，交给解码环节兜底
        return True
    return any(ctype.startswith(prefix) for prefix in _TEXTUAL_CONTENT_TYPES)


class Fetcher:
    """带缓存的 HTML 抓取器。

    一个实例持有一个 ``httpx.AsyncClient``（复用连接池），插件卸载时必须
    调用 :meth:`aclose`。
    """

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        user_agent: str = "Mozilla/5.0 (compatible; MaiBot-WebSearch/0.1)",
        max_bytes: int = 5 * 1024 * 1024,
        allow_private_networks: bool = False,
        proxy: str = "",
        cache: Any | None = None,
    ) -> None:
        """初始化抓取器。

        Args:
            timeout: 单次抓取超时（秒）。
            user_agent: 请求 UA。
            max_bytes: 响应体体积上限（字节），超出即截断放弃。
            allow_private_networks: 是否关闭 SSRF 防护（默认关闭）。
            proxy: 代理地址，空字符串表示直连。
            cache: 可选的 TTLCache 实例。
        """
        self.timeout = max(1.0, float(timeout))
        self.user_agent = user_agent
        self.max_bytes = max(1024, int(max_bytes))
        self.allow_private_networks = bool(allow_private_networks)
        self.proxy = (proxy or "").strip()
        self.cache = cache
        self._client: httpx.AsyncClient | None = None

    def _build_client(self) -> httpx.AsyncClient:
        """构造 httpx 客户端（兼容 0.27 与 0.28 的代理参数差异）。

        follow_redirects=False：重定向由 :meth:`_do_fetch` 手动逐跳跟随，
        每一跳都重新过 SSRF 校验（见该方法的 docstring）。
        """
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
            "follow_redirects": False,
            "max_redirects": 5,
            "headers": {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            "trust_env": True,
        }
        if not self.proxy:
            return httpx.AsyncClient(**kwargs)
        try:
            return httpx.AsyncClient(proxy=self.proxy, **kwargs)
        except TypeError:  # httpx < 0.28 只认 proxies
            return httpx.AsyncClient(proxies=self.proxy, **kwargs)

    def _ensure_client(self) -> httpx.AsyncClient:
        """惰性创建客户端，避免插件加载时就占用连接资源。"""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def aclose(self) -> None:
        """关闭底层客户端，释放连接池。"""
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass

    async def fetch(self, url: str) -> FetchOutcome:
        """抓取单个 URL。

        Args:
            url: 目标地址。

        Returns:
            FetchOutcome: 抓取结果。失败时 ``ok=False`` 且 ``error`` 为中文原因。
        """
        target = (url or "").strip()
        if not target:
            return FetchOutcome(ok=False, url=target, error="地址为空")

        cache_key = f"fetch:{target}"
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if isinstance(cached, FetchOutcome):
                return FetchOutcome(
                    ok=cached.ok,
                    url=cached.url,
                    final_url=cached.final_url,
                    status=cached.status,
                    content_type=cached.content_type,
                    title=cached.title,
                    text=cached.text,
                    error=cached.error,
                    from_cache=True,
                )

        allowed, reason = await asyncio.to_thread(
            is_safe_url, target, self.allow_private_networks
        )
        if not allowed:
            return FetchOutcome(ok=False, url=target, error=reason)

        outcome = await self._do_fetch(target)

        if outcome.ok and self.cache is not None:
            self.cache.set(cache_key, outcome)
        return outcome

    async def _do_fetch(self, target: str) -> FetchOutcome:
        """执行实际请求并解析响应。

        重定向**手动逐跳跟随**（follow_redirects=False）：每一跳都重新做
        SSRF 校验。httpx 跨主机重定向不剥离敏感头，且「中间跳进内网再跳回
        公网」能绕过只校验 final_url 的方案，所以必须每跳验证。
        """
        client = self._ensure_client()
        current_url = target
        history_urls: list[str] = []
        try:
            for _hop in range(6):  # 与原 max_redirects=5 对齐：最多跟 5 跳
                async with client.stream("GET", current_url) as response:
                    status = response.status_code
                    content_type = response.headers.get("content-type", "")
                    location = response.headers.get("location", "")

                    if _is_redirect(status) and location.strip():
                        # 校验跳转目标（先于真正发起新请求）
                        next_url = urljoin(current_url, location.strip())
                        if next_url in history_urls or next_url == current_url:
                            return FetchOutcome(
                                ok=False, url=target, final_url=next_url,
                                status=status, content_type=content_type,
                                error="重定向出现循环",
                            )
                        allowed, reason = await _check_url_allowed(
                            next_url, self.allow_private_networks
                        )
                        if not allowed:
                            return FetchOutcome(
                                ok=False, url=target, final_url=next_url,
                                status=status, content_type=content_type,
                                error=f"重定向目标被拒绝：{reason}",
                            )
                        history_urls.append(current_url)
                        current_url = next_url
                        continue

                    if status >= 400:
                        return FetchOutcome(
                            ok=False,
                            url=target,
                            final_url=current_url,
                            status=status,
                            content_type=content_type,
                            error=f"HTTP {status}",
                        )

                    if not _is_textual(content_type):
                        return FetchOutcome(
                            ok=False,
                            url=target,
                            final_url=current_url,
                            status=status,
                            content_type=content_type,
                            error=f"不支持的内容类型：{content_type.split(';')[0] or '未知'}",
                        )

                    chunks: list[bytes] = []
                    size = 0
                    truncated = False
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > self.max_bytes:
                            remain = self.max_bytes - (size - len(chunk))
                            if remain > 0:
                                chunks.append(chunk[:remain])
                            truncated = True
                            break
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    final_url = current_url
                    break
            else:
                return FetchOutcome(
                    ok=False, url=target, final_url=current_url,
                    error="重定向次数过多",
                )

        except httpx.TimeoutException:
            return FetchOutcome(ok=False, url=target, error=f"请求超时（{self.timeout:.0f}s）")
        except httpx.TooManyRedirects:
            return FetchOutcome(ok=False, url=target, error="重定向次数过多")
        except httpx.HTTPError as exc:
            return FetchOutcome(ok=False, url=target, error=f"网络错误：{exc}")
        except Exception as exc:  # pragma: no cover - 兜底，绝不让异常冒到工具层
            return FetchOutcome(ok=False, url=target, error=f"抓取异常：{exc}")

        text_body = _decode_body(body, content_type)
        normalized_type = (content_type or "").split(";")[0].strip().lower()

        if normalized_type == "application/json":
            title, content = "", format_json_response(text_body, 0)
        else:
            title, content = extract_main_text(text_body, 0)

        if truncated:
            content = content + "\n…(内容过长，已截断)"

        return FetchOutcome(
            ok=True,
            url=target,
            final_url=final_url,
            status=status,
            content_type=content_type,
            title=title,
            text=content,
        )
