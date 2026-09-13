"""带 TTL 与容量上限的内存缓存。

用于缓存搜索结果与已抓取的网页正文，避免麦麦追问同一话题时重复外呼
（既省时间，也降低被搜索引擎判定为异常流量的风险）。
"""

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """线程安全的最小 TTL + LRU 缓存。

    命中后会刷新该条目的 LRU 位置，但**不**延长 TTL——过期即失效，
    语义保持"从写入时刻起 N 秒内有效"，便于解释与排查。
    """

    def __init__(self, max_entries: int = 128, ttl_seconds: float = 1800.0) -> None:
        """初始化缓存。

        Args:
            max_entries: 条目上限，超出时淘汰最久未使用者。
            ttl_seconds: 条目存活秒数。
        """
        self._max_entries = max(1, int(max_entries))
        self._ttl = max(1.0, float(ttl_seconds))
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        """读取缓存。

        Args:
            key: 缓存键。

        Returns:
            Any | None: 命中返回值；未命中或已过期返回 None。
        """
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._data[key]
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """写入缓存。

        Args:
            key: 缓存键。
            value: 缓存值。
            ttl_seconds: 覆盖默认 TTL。
        """
        ttl = self._ttl if ttl_seconds is None else max(1.0, float(ttl_seconds))
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    def clear(self) -> int:
        """清空缓存。

        Returns:
            int: 被清除的条目数。
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def purge_expired(self) -> int:
        """主动清理已过期条目。

        Returns:
            int: 被清理的条目数。
        """
        now = time.monotonic()
        removed = 0
        with self._lock:
            for key in [k for k, (exp, _) in self._data.items() if exp <= now]:
                del self._data[key]
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        """返回缓存统计，供诊断命令展示。

        Returns:
            dict[str, int]: 条目数、命中、未命中、淘汰数。
        """
        with self._lock:
            return {
                "entries": len(self._data),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
