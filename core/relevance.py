"""搜索结果相关性检测。

**为什么需要这个模块**（来自真机日志的教训）：

当搜索引擎把长查询拆散匹配时，它会返回一批"结构正常但内容完全无关"的结果。
真机实测：查询「中V最近的活动」拿到的是汉字"中"的字典释义，查询「无线电」拿到的是
一堆无关站点。插件如果照单全收，LLM 会拿着这些材料以为"搜索成功了但没找到"，
于是换个说法再搜——日志里连续空转了 9 轮、累计 200+ 秒工具执行时间。

因此这里提供一个**廉价**的相关性判据，用于两件事：

1. 引擎链据此决定"这批结果不算数，继续降级到下一个引擎"；
2. 回传给 LLM 时明确标注"结果可能不相关"，让它尽快改关键词而不是反复重试。

判断手段刻意做得简单：把查询切成若干关键片段，看结果里是否出现其中任何一个。
不引入分词库（中文分词依赖体积大且需编译），对"完全无关"这个目标场景足够用。
"""

import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .models import SearchResult

# 连续汉字串，或长度 >= 2 的英文/数字串
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}")

# 汉语虚词，用作切分点："最近的活动" → ["最近", "活动"]
_PARTICLES = "的了和与及或在是有对为地得着过吗呢吧啊之其"

# 长度低于此值的片段不参与判断（单字匹配噪声太大）
_MIN_TERM_LEN = 2

# 命中率低于该值即认为"疑似不相关"。刻意取得宽松——目的是识别"完全无关"，
# 而不是精确评估相关度；宁可放过，不可误杀。
LOW_RELEVANCE_THRESHOLD = 0.34


def extract_terms(query: str, max_terms: int = 8) -> list[str]:
    """从查询串中提取用于相关性判断的关键片段。

    Args:
        query: 原始查询。
        max_terms: 最多保留的片段数。

    Returns:
        list[str]: 去重后的关键片段；全部过短时返回空列表。
    """
    raw = (query or "").strip()
    if not raw:
        return []

    terms: list[str] = []
    seen: set[str] = set()

    for chunk in _TOKEN_RE.findall(raw):
        # 先按虚词把长串切开，避免整句变成一个永远匹配不上的长片段
        pieces = chunk
        for particle in _PARTICLES:
            pieces = pieces.replace(particle, " ")
        for piece in pieces.split():
            piece = piece.strip()
            if len(piece) < _MIN_TERM_LEN:
                continue
            key = piece.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(piece)
            if len(terms) >= max_terms:
                return terms

    return terms


def relevance_ratio(
    query: str,
    results: Sequence["SearchResult"],
    sample_chars: int = 400,
) -> float:
    """计算结果中"与查询有重合"的比例。

    Args:
        query: 原始查询。
        results: 搜索结果列表。
        sample_chars: 参与匹配的正文取样长度（避免为长正文做全量扫描）。

    Returns:
        float: 命中比例 0.0~1.0；当查询无可切分片段或结果为空时返回 -1.0
        （表示"无法判断"，调用方应当放行而不是拦截）。
    """
    terms = extract_terms(query)
    if not terms or not results:
        return -1.0

    hits = 0
    for item in results:
        haystack = " ".join(
            (
                item.title or "",
                item.snippet or "",
                (item.content or "")[:sample_chars],
            )
        ).lower()
        if any(term.lower() in haystack for term in terms):
            hits += 1
    return hits / len(results)


def is_low_relevance(ratio: float) -> bool:
    """判断命中率是否达到"疑似不相关"的程度。

    Args:
        ratio: :func:`relevance_ratio` 的返回值。

    Returns:
        bool: True 表示疑似不相关；-1（无法判断）一律返回 False。
    """
    return 0.0 <= ratio < LOW_RELEVANCE_THRESHOLD


def describe(ratio: float) -> str:
    """把命中率转成给用户/LLM 看的中文描述。"""
    if ratio < 0:
        return "无法评估相关性"
    return f"结果与查询词的重合度 {ratio:.0%}"
