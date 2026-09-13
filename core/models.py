"""搜索与抓取的数据结构。"""

from dataclasses import dataclass


@dataclass(slots=True)
class SearchResult:
    """单条搜索结果。

    一条结果会经历三个阶段，字段随之填充：
      搜索阶段 → title / url / snippet / engine / score
      抓取阶段 → content（正文）或 fetch_error（失败原因）
      总结阶段 → 由 plugin.py 拼接为 prompt，不改动本对象
    """

    title: str = ""
    url: str = ""
    snippet: str = ""
    engine: str = ""
    content: str = ""
    fetch_error: str = ""
    score: float = 0.0

    @property
    def has_content(self) -> bool:
        """是否已抓到可用正文。"""
        return bool(self.content.strip())

    def to_prompt_block(self, index: int, max_chars: int = 4000) -> str:
        """渲染成给 LLM 阅读的材料块。

        Args:
            index: 从 1 开始的序号，用于让 LLM 按编号引用来源。
            max_chars: 单条材料正文上限。

        Returns:
            str: 形如 ``[1] 标题\\nURL: ...\\n正文摘要`` 的文本块。
        """
        body = self.content.strip() or self.snippet.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "…"
        if not body:
            body = "(未能获取正文，仅有标题)"

        lines = [f"[{index}] {self.title or '(无标题)'}"]
        if self.url:
            lines.append(f"URL: {self.url}")
        lines.append(body)
        return "\n".join(lines)

    def to_source_line(self, index: int) -> str:
        """渲染成来源清单中的一行。"""
        title = self.title or "(无标题)"
        return f"[{index}] {title} — {self.url}" if self.url else f"[{index}] {title}"
