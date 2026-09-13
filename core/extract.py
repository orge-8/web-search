"""HTML 正文提取。

不引入 trafilatura / readability-lxml 这类重依赖（它们需要 lxml，在真机
Windows 环境下编译安装存在失败风险），改用 BeautifulSoup + 启发式规则
自行提取。策略：

   1. 摘掉脚本、样式、导航、页脚、表单等结构性噪声；
   2. 在若干常见"正文容器"候选里选出纯文本最长的那个；
   3. 只保留块级元素的文本，逐行清洗、去重、去短行；
   4. 若上述策略拿到的内容过少（例如 SPA 页面），退回全页文本兜底。

宁可少提取，也不要提取出一堆导航链接——后者会让 LLM 的总结质量崩塌。
"""

import json
import re
from typing import Any

from bs4 import BeautifulSoup

# 刻意使用标准库解析器：lxml / html5lib 容错更好，但会引入需要编译的依赖，
# 与"插件在真机上必须能装上"这一优先级相比不划算。
_SOUP_PARSER = "html.parser"

# 结构性噪声：这些标签的内容一律丢弃
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "template",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "footer",
    "header",
    "aside",
    "dialog",
)

# 噪声类名/ID 关键字（小写匹配）
_NOISE_PATTERN = re.compile(
    r"(^|[-_ ])("
    r"nav|navbar|menu|sidebar|side-bar|footer|header|breadcrumb|comment|"
    r"advert|ads?|banner|promo|share|social|cookie|popup|modal|toolbar|"
    r"pagination|related|recommend|tag-list|meta|masthead"
    r")($|[-_ ])",
    re.IGNORECASE,
)

# 正文容器候选，按优先级从高到低尝试
_CONTENT_SELECTORS = (
    "article",
    "main",
    "[role='main']",
    "#article",
    "#content",
    "#main",
    ".article-content",
    ".article-body",
    ".post-content",
    ".post-body",
    ".entry-content",
    ".markdown-body",
    ".rich_media_content",
    ".content",
)

_BLOCK_TAGS = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "pre",
    "td",
    "th",
    "dd",
    "dt",
    "figcaption",
    "summary",
)

_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u3000]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# 低于该长度视为"没提取到正文"，触发兜底策略
_MIN_GOOD_LENGTH = 200


def _clean_line(line: str) -> str:
    """压缩单行内空白。"""
    return _WHITESPACE_RE.sub(" ", line).strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    """原地移除噪声标签与噪声容器。"""
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        if not getattr(tag, "attrs", None):
            continue
        marker = " ".join(
            str(tag.get(attr) or "") for attr in ("class", "id", "role", "aria-label")
        )
        if marker and _NOISE_PATTERN.search(marker):
            tag.decompose()


def _collect_block_text(node: Any) -> list[str]:
    """收集节点内所有块级元素的文本行。"""
    lines: list[str] = []
    for element in node.find_all(_BLOCK_TAGS):
        text = _clean_line(element.get_text(" ", strip=True))
        if text:
            lines.append(text)
    return lines


def _dedupe_and_filter(lines: list[str]) -> list[str]:
    """去掉空行、过短行与完全重复行。

    重复行通常是导航 / 版权声明的残留，保留会干扰 LLM 判断。
    但连续出现的重复行（例如表格里同值单元格）只按"全局重复"处理，
    因此这里对同一文本只保留首次出现。
    """
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        if len(line) < 2:
            continue
        # 极短且不含中文/字母的行多为分隔符或图标残留
        if len(line) <= 3 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(line)
    return output


def _extract_title(soup: BeautifulSoup) -> str:
    """按优先级提取页面标题。"""
    for selector, attr in (
        ("meta[property='og:title']", "content"),
        ("meta[name='twitter:title']", "content"),
        ("meta[name='title']", "content"),
    ):
        node = soup.select_one(selector)
        if node and node.get(attr):
            title = _clean_line(str(node.get(attr)))
            if title:
                return title

    if soup.title and soup.title.string:
        title = _clean_line(str(soup.title.string))
        if title:
            # 常见形式 "文章标题 - 站点名"，只保留主标题
            return re.split(r"\s+[|\-–—]\s+", title)[0].strip() or title

    heading = soup.find("h1")
    if heading:
        return _clean_line(heading.get_text(" ", strip=True))
    return ""


def _pick_content_node(soup: BeautifulSoup) -> Any:
    """选出纯文本最长的正文容器。"""
    best_node = None
    best_len = 0
    for selector in _CONTENT_SELECTORS:
        for node in soup.select(selector):
            text_len = len(node.get_text(" ", strip=True))
            if text_len > best_len:
                best_node, best_len = node, text_len
    if best_node is None or best_len < _MIN_GOOD_LENGTH:
        return soup.body or soup
    return best_node


def html_to_text(html: str, max_chars: int = 0) -> str:
    """把 HTML 直接转成清洗后的纯文本（不做正文容器挑选）。

    Args:
        html: HTML 源码。
        max_chars: 截断上限，0 表示不截断。

    Returns:
        str: 清洗后的纯文本；解析失败返回空字符串。
    """
    if not html or not html.strip():
        return ""
    try:
        soup = BeautifulSoup(html, _SOUP_PARSER)
    except Exception:
        return ""

    _strip_noise(soup)
    node = soup.body or soup
    lines = _dedupe_and_filter(_collect_block_text(node))
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def extract_main_text(html: str, max_chars: int = 4000) -> tuple[str, str]:
    """从 HTML 中提取 (标题, 正文)。

    Args:
        html: HTML 源码。
        max_chars: 正文截断上限，0 表示不截断。

    Returns:
        tuple[str, str]: 标题与正文；解析失败时返回 ("", "")。
    """
    if not html or not html.strip():
        return "", ""
    try:
        soup = BeautifulSoup(html, _SOUP_PARSER)
    except Exception:
        return "", ""

    title = _extract_title(soup)
    _strip_noise(soup)
    node = _pick_content_node(soup)
    lines = _dedupe_and_filter(_collect_block_text(node))

    text = "\n".join(lines)
    if len(text) < _MIN_GOOD_LENGTH:
        # 兜底：容器挑选可能过于激进（例如整页就是一个 div）
        fallback_lines = _dedupe_and_filter(_collect_block_text(soup.body or soup))
        fallback_text = "\n".join(fallback_lines)
        if len(fallback_text) > len(text):
            text = fallback_text

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    text = _BLANKLINES_RE.sub("\n\n", text).strip()
    return title, text


def format_json_response(body: str, max_chars: int = 4000) -> str:
    """把 JSON 响应体格式化为紧凑可读文本。

    Args:
        body: 原始响应文本。
        max_chars: 截断上限。

    Returns:
        str: 格式化结果；非法 JSON 时返回原文（截断）。
    """
    try:
        data = json.loads(body)
    except Exception:
        return body[:max_chars] if max_chars else body

    try:
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        pretty = body
    if max_chars and len(pretty) > max_chars:
        pretty = pretty[:max_chars]
    return pretty
