"""带 SSRF 防护与体积上限的图片下载器。

与 ``fetcher.py`` 的 :class:`Fetcher` 平行、职责单一：只下图片，不做文本
解码/正文提取。安全校验**复用** ``fetcher.is_safe_url``（含重定向后二次校验），
不复制安全代码；但 ``Fetcher._do_fetch`` 只放行文本 content-type，图片走它
会被拒，所以这里单独实现，并增加图片特有的防护：

* 魔数校验（magic bytes）—— 防 HTML 错误页伪装成图片
* 双层体积上限 —— 下载硬截断 + 发送上限（base64 过 msgpack RPC 膨胀 1.33 倍，
  4MB 原图传输约 5.3MB，超出直接拒图换下一张，不做压缩以保持零新依赖）
"""

import asyncio
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from .fetcher import _check_url_allowed, is_safe_url

# 前缀魔数：JPEG / PNG / GIF / WebP
_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG", "png"),
    (b"GIF8", "gif"),
    (b"RIFF", "webp"),
)

# URL 与 Referer：百度图片的 thumbURL 对 Referer 敏感，带上可显著降低 403
_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://image.baidu.com/",
}


@dataclass(slots=True)
class ImageDownload:
    """一次图片下载的结果。"""

    ok: bool = False
    data: bytes = b""
    content_type: str = ""
    final_url: str = ""
    size: int = 0
    error: str = ""  # 中文原因，失败时填写


@dataclass(slots=True)
class ImageDownloader:
    """图片下载器。一个实例持有一个 httpx.AsyncClient，卸载时须 ``aclose()``。"""

    timeout: float = 10.0
    user_agent: str = ""
    max_download_bytes: int = 8 * 1024 * 1024
    max_send_bytes: int = 4 * 1024 * 1024
    allow_private_networks: bool = False
    proxy: str = ""
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"User-Agent": self.user_agent} if self.user_agent else {}
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=False,  # 重定向手动逐跳跟随，每跳过 SSRF 校验
                max_redirects=5,
                proxy=self.proxy or None,
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        """释放底层客户端。"""
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except httpx.HTTPError:  # pragma: no cover - 关闭失败无须处理
                pass

    async def download(self, url: str) -> ImageDownload:
        """下载一张图片。失败返回 ``ok=False`` 且 ``error`` 为中文原因。

        重定向**手动逐跳跟随**（与 fetcher 同策略）：每一跳的落地地址都重新
        做 SSRF 校验，防止「302 中间跳进内网再跳回公网」绕过只校验 final_url
        的方案。
        """
        target = (url or "").strip()
        if not target:
            return ImageDownload(error="地址为空")

        allowed, reason = await asyncio.to_thread(
            is_safe_url, target, self.allow_private_networks
        )
        if not allowed:
            return ImageDownload(error=reason)

        client = self._ensure_client()
        current_url = target
        try:
            for _hop in range(6):  # 最多跟 5 跳重定向
                # 首跳带百度 Referer（对 thumbURL 的 403 有缓解）；重定向跳到
                # 第三方域后不再带，避免向无关站点泄漏来源偏好
                hop_headers = _HEADERS if _hop == 0 else {
                    k: v for k, v in _HEADERS.items() if k != "Referer"
                }
                async with client.stream(
                    "GET", current_url, headers=hop_headers
                ) as response:
                    status = response.status_code
                    content_type = response.headers.get("content-type", "")
                    location = response.headers.get("location", "")

                    if status in (301, 302, 303, 307, 308) and location.strip():
                        next_url = urljoin(current_url, location.strip())
                        allowed, reason = await _check_url_allowed(
                            next_url, self.allow_private_networks
                        )
                        if not allowed:
                            return ImageDownload(
                                final_url=next_url,
                                error=f"重定向目标被拒绝：{reason}",
                            )
                        current_url = next_url
                        continue

                    final_url = current_url

                    if status >= 400:
                        return ImageDownload(
                            final_url=final_url,
                            content_type=content_type,
                            error=f"HTTP {status}",
                        )

                    if not (content_type or "").split(";")[0].strip().lower().startswith("image/"):
                        return ImageDownload(
                            final_url=final_url,
                            content_type=content_type,
                            error=f"非图片内容：{(content_type.split(';')[0] or '未知').strip()}",
                        )

                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > self.max_download_bytes:
                            return ImageDownload(
                                final_url=final_url,
                                content_type=content_type,
                                size=size,
                                error=(
                                    f"图片超过下载上限"
                                    f"（{self.max_download_bytes / 1024 / 1024:.0f}MB）"
                                ),
                            )
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    break
            else:
                return ImageDownload(final_url=current_url, error="重定向次数过多")

        except httpx.TimeoutException:
            return ImageDownload(error=f"请求超时（{self.timeout:.0f}s）")
        except httpx.TooManyRedirects:
            return ImageDownload(error="重定向次数过多")
        except httpx.HTTPError as exc:
            return ImageDownload(error=f"网络错误：{exc}")
        except Exception as exc:  # pragma: no cover - 兜底，绝不让异常冒到工具层
            return ImageDownload(error=f"下载异常：{exc}")

        # 魔数校验：防 HTML 错误页伪装成图片（content-type 可能说谎）
        if not any(data.startswith(magic) for magic, _kind in _IMAGE_MAGIC):
            return ImageDownload(
                data=data,
                content_type=content_type,
                final_url=final_url,
                size=len(data),
                error="内容不是有效图片",
            )

        if len(data) > self.max_send_bytes:
            return ImageDownload(
                data=data,
                content_type=content_type,
                final_url=final_url,
                size=len(data),
                error=(
                    f"图片超过发送上限（{self.max_send_bytes / 1024 / 1024:.0f}MB），已跳过"
                ),
            )

        return ImageDownload(
            ok=True,
            data=data,
            content_type=content_type,
            final_url=final_url,
            size=len(data),
        )
