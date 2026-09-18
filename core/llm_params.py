"""LLM 调用参数构造：对齐 MaiBot 1.2.5 / SDK 2.8.1 的「任务名 vs 模型名」语义拆分。

背景（真机 2026-09-17，web-search v0.3.5）：::

    09-17 22:13:33 [plugin.org.orge-8.web-search] LLM 返回失败（任务名 utils）：未找到名为 'utils' 的模型

插件把**模型任务名**填进了 ``model=`` 槽位。Host ≤1.2.4 时 ``model`` 就是任务名槽位，
两者共用命名空间，所以一直能跑；Host ≥1.2.5 拆出 ``task_name`` 之后，``model`` 的语义
变成「具体模型名」，Host 于是拿 ``utils`` 去模型表里查 → 报「未找到名为 'utils' 的模型」。

为什么能确定名字是插件传出去的：插件只回显 Host 返回的 ``error``，错误文本里带上
``utils`` 就说明是本次调用的出参。且 ``utils`` 是 SDK 默认任务名，正回答「它本来是个
任务名，却被当模型名解析」这一点。

修复策略（**用户配置不需要改**）：``summarize.task`` 填的任务名继续照填，只是改送
``task_name`` 槽位；另开一个 ``model`` 配置项给「具体模型名」用。

本模块是纯逻辑（不碰 ``self.ctx``），签名探测所需的 ``generate`` 可调用体由
``plugin.py`` 注入，因此全部路径都能脱机单测。
"""

from __future__ import annotations

import inspect
import re
from importlib.metadata import version
from typing import Any, Sequence

#: 内置任务名白名单：用于识别「任务名被误填进模型名槽位」。
#: 只用它做**左侧短路**判断，命中后还要与 Host 实际任务清单求交才动手（见 plugin 侧）。
KNOWN_TASK_NAMES: frozenset[str] = frozenset(
    {"utils", "replyer", "planner", "embedding", "memory", "topic", "vlm"}
)

#: ``task_name`` 参数是 SDK 2.8.1 引入的。在此之前 ``generate`` 只有 ``model``，
#: 且 ``model`` 按任务名解释。
_TASK_NAME_MIN_SDK: tuple[int, ...] = (2, 8, 1)

_SDK_PACKAGE = "maibot-plugin-sdk"

#: 匹配 Host 的「未找到名为 'xxx' 的模型」。兼容中英文引号。
_REJECTED_NAME_RE = re.compile(r"未找到名为\s*[\"'“”']([^\"'“”']+)[\"'“”']\s*的模型")


def rejected_model_name(reason: str) -> str:
    """从 Host 报错文本里取出被拒的名字，取不到返回空串。

    ``"未找到名为 'utils' 的模型"`` → ``"utils"``。有了它才能判断「这次被拒的
    到底是我传的哪个参数」，进而只停用那一个。
    """
    match = _REJECTED_NAME_RE.search(reason or "")
    return match.group(1).strip() if match else ""


def _parse_version(raw: str) -> tuple[int, ...]:
    """把 ``"2.8.1"`` / ``"2.8.1.dev3"`` 解析成可比较的数字元组，失败返回空元组。"""
    parts: list[int] = []
    for chunk in (raw or "").strip().split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def sdk_version_supports_task_name() -> bool:
    """按已安装 SDK 版本判断是否支持 ``task_name``。

    读不到版本时返回 ``True``：当前主线（Host 1.2.5 / SDK 2.8.1+）是新语义，
    新语义下传 ``task_name`` 才正确，猜新比猜旧安全。
    """
    try:
        installed = version(_SDK_PACKAGE)
    except (PackageNotFoundError, Exception):  # noqa: BLE001 - 任何取版本失败都降级
        return True
    parsed = _parse_version(installed)
    if not parsed:
        return True
    return parsed >= _TASK_NAME_MIN_SDK


def generate_supports_task_name(generate: Any) -> bool:
    """探测 ``ctx.llm.generate`` 是否接受 ``task_name`` 关键字。

    优先看签名：显式声明了 ``task_name`` 即新版；完全没有 ``**kwargs`` 且没有
    ``task_name``，就是确凿的旧版。签名被 ``**kwargs`` 吞掉（包装器常见）时
    无法判断，退回按 SDK 版本判断。
    """
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):  # 内建/被装饰遮蔽等，探测不到
        return sdk_version_supports_task_name()

    if "task_name" in params:
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return sdk_version_supports_task_name()
    return False


def needs_task_list_lookup(model: str) -> bool:
    """纠偏是否需要拉取 Host 任务清单。

    **左侧短路**：``model`` 为空（默认配置）或填的是真模型名时都不命中，
    于是不会产生任何额外 RPC——只有「看起来像任务名」的值才值得去查一次。
    """
    value = (model or "").strip()
    return bool(value) and value in KNOWN_TASK_NAMES


def build_llm_kwargs(
    *,
    task: str,
    model: str,
    supports_task_name: bool,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_ms: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """构造 ``ctx.llm.generate`` 的关键字参数。

    Args:
        task: 模型**任务名**（Host ``[model_task_config]`` 的键，如 ``replyer``）。
        model: **具体模型名**，留空表示用任务默认模型。
        supports_task_name: SDK 是否支持新语义（由 :func:`generate_supports_task_name` 给出）。
        temperature: 采样温度，``None`` 表示不显式传。
        max_tokens: 输出上限，``None`` 表示不显式传。
        timeout_ms: RPC 层超时（毫秒），``None`` 表示不显式传。

    Returns:
        ``(kwargs, notices)``：直接展开给 ``generate`` 的参数字典，以及需要
        告警给用户的信息（措辞为简体中文）。
    """
    notices: list[str] = []
    clean_task = (task or "").strip()
    clean_model = (model or "").strip()

    kwargs: dict[str, Any] = {}

    if supports_task_name:
        # 1.2.5+：任务名与模型名各走各的槽位，这才是修复的核心。
        if clean_task:
            kwargs["task_name"] = clean_task
        if clean_model:
            kwargs["model"] = clean_model
    else:
        # ≤2.8.0：``model`` 本身按任务名解释，没有「具体模型名」这个概念。
        if clean_task:
            kwargs["model"] = clean_task
        if clean_model:
            notices.append(
                "当前 Host/SDK 不支持单独指定具体模型名（需 SDK ≥2.8.1），"
                f"已忽略配置项 summarize.model = 「{clean_model}」"
            )

    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout_ms is not None:
        kwargs["timeout_ms"] = timeout_ms

    return kwargs, notices


def describe_kwargs(kwargs: dict[str, Any]) -> str:
    """把**实际传出**的参数渲染成一行日志。

    配置值与传出值之间可能隔着纠偏/默认值注入，只读配置永远对不上号，
    所以日志必须以这里为准。
    """
    if not kwargs:
        return "(无显式参数)"
    keys = ("task_name", "model", "temperature", "max_tokens", "timeout_ms")
    parts = [f"{key}={kwargs[key]!r}" for key in keys if key in kwargs]
    extra = [f"{k}={v!r}" for k, v in kwargs.items() if k not in keys]
    return " ".join(parts + extra)


def explain_missing_model(name: str, available_tasks: Sequence[str] | None = None) -> str:
    """把「未找到名为 X 的模型」翻译成可执行的修复建议。

    Args:
        name: Host 报错里回显的名字。
        available_tasks: Host 的可用任务名清单，取不到时传 ``None``。
    """
    if not name:
        return "Host 未返回具体原因，请查看 MaiBot 主程序日志的 [模型服务] 行。"

    lines = [
        f"模型解析失败：「{name}」。",
        "若它是任务名（如 utils / replyer），请填在 summarize.task；"
        "summarize.model 只接受具体模型名，留空即可。",
    ]
    if available_tasks:
        lines.append("Host 当前可用任务名：" + " / ".join(str(x) for x in available_tasks[:15]))
    else:
        lines.append(
            "未取到 Host 任务名清单：请到 WebUI 确认模型列表非空"
            "（先保存提供商，再添加模型，最后把任务指向具体模型）。"
        )
    return " ".join(lines)


__all__ = [
    "KNOWN_TASK_NAMES",
    "build_llm_kwargs",
    "describe_kwargs",
    "explain_missing_model",
    "generate_supports_task_name",
    "needs_task_list_lookup",
    "rejected_model_name",
    "sdk_version_supports_task_name",
]
