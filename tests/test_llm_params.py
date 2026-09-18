"""LLM 传参测试：钉住「任务名 vs 模型名」的槽位归属。

背景（真机 2026-09-17）：插件把任务名填进 ``model`` 槽位，在 Host 1.2.5 / SDK 2.8.1
下得到 ``未找到名为 'utils' 的模型``。本测试把这个回归钉死，分两层：

* **纯函数层**：直接测 ``core.llm_params`` 的传参构造，与已装 SDK 版本无关。
* **插件层**：给插件注入一个假的 ``ctx.llm``，断言它**实际传出的 kwargs**。
  这样新语义（SDK ≥2.8.1）与旧语义（≤2.8.0）两条路径都能在任意环境验证——
  本地 venv 若是 2.8.0，走真实 SDK 只能看到旧语义，会漏掉真机那条路径。

运行: python tests/test_llm_params.py
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# 插件根目录也加入路径：纯函数层需要 import core.llm_params。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fakehost import (  # noqa: E402
    FakeHost,
    FakePaths,
    bind_context,
    build_context,
    get_default_config,
    load_plugin_module,
)

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent

_pass = 0
_fail: list[str] = []


def check(condition: bool, label: str) -> None:
    """记录一条断言结果。"""
    global _pass
    if condition:
        _pass += 1
    else:
        _fail.append(label)
        print(f"  FAIL: {label}")


# --------------------------------------------------------------- 假 LLM 能力

class FakeLlm:
    """记录每次 generate 的**全部实参**，用来断言实际传出值。"""

    def __init__(self, available: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.available = ["utils", "replyer"] if available is None else available
        self.response: dict = {"success": True, "response": "ok"}

    async def generate(self, prompt, model="", *, task_name="utils", model_name="", **kwargs):
        record = {"prompt": prompt, "model": model, "task_name": task_name, "model_name": model_name}
        record.update(kwargs)
        self.calls.append(record)
        return dict(self.response)

    async def get_available_models(self) -> list[str]:
        return list(self.available)

    @property
    def last(self) -> dict:
        return self.calls[-1] if self.calls else {}


class _BoomLlm(FakeLlm):
    """get_available_models 抛异常：模拟 Host RPC 挂掉。"""

    async def get_available_models(self) -> list[str]:
        raise RuntimeError("RPC 挂了")


def _make_plugin(llm) -> object:
    """造一个插件实例，ctx 里只装本次测试需要的能力。"""
    module = load_plugin_module(PLUGIN_DIR)
    plugin = module.create_plugin()
    ctx = SimpleNamespace(
        plugin_id="org.orge-8.web-search",
        logger=logging.getLogger("test.web-search"),
        paths=FakePaths(),
        llm=llm,
    )
    bind_context(plugin, ctx, get_default_config(getattr(type(plugin), "config_model", None)))
    return plugin


# ------------------------------------------------------------ 纯函数层

def test_pure_functions() -> None:
    print("[1] core.llm_params 传参构造")
    from core.llm_params import (
        KNOWN_TASK_NAMES,
        build_llm_kwargs,
        describe_kwargs,
        explain_missing_model,
        generate_supports_task_name,
        needs_task_list_lookup,
        rejected_model_name,
        sdk_version_supports_task_name,
    )

    # 核心回归：任务名必须进 task_name，绝不能占 model 槽位
    kwargs, notices = build_llm_kwargs(task="utils", model="", supports_task_name=True)
    check(kwargs.get("task_name") == "utils", "新语义：任务名进 task_name")
    check("model" not in kwargs, "新语义：model 槽位不被任务名占用（真机 bug 根因）")
    check(not notices, "新语义：正常配置不产生告警")

    # 任务名 + 具体模型名各归其位
    kwargs, _ = build_llm_kwargs(task="replyer", model="gpt-4o-mini", supports_task_name=True)
    check(kwargs.get("task_name") == "replyer" and kwargs.get("model") == "gpt-4o-mini",
          "新语义：任务名与具体模型名分别落到各自槽位")

    # 旧 SDK（≤2.8.0）：model 本身就是任务名槽位，回落才是对的
    kwargs, notices = build_llm_kwargs(task="utils", model="", supports_task_name=False)
    check(kwargs.get("model") == "utils" and "task_name" not in kwargs,
          "旧语义：任务名回落进 model（那代 SDK 的唯一槽位）")
    check(not notices, "旧语义：纯任务名不产生告警")

    # 旧 SDK 无法表达具体模型名 → 忽略并告警
    kwargs, notices = build_llm_kwargs(task="replyer", model="gpt-4o", supports_task_name=False)
    check(kwargs.get("model") == "replyer", "旧语义：model 字段不会覆盖任务名")
    check(bool(notices) and "忽略" in notices[0], "旧语义：无法指定具体模型名时给出告警")

    # 空配置：不显式传，交给 Host 默认
    kwargs, notices = build_llm_kwargs(task="", model="", supports_task_name=True)
    check("task_name" not in kwargs and "model" not in kwargs, "空配置：不显式传出任务/模型参数")
    check(not notices, "空配置：静默不告警")

    # 基础参数透传
    kwargs, _ = build_llm_kwargs(
        task="replyer", model="", supports_task_name=True,
        temperature=0.3, max_tokens=1200, timeout_ms=60000,
    )
    check(kwargs.get("temperature") == 0.3 and kwargs.get("max_tokens") == 1200
          and kwargs.get("timeout_ms") == 60000, "基础参数（temperature/max_tokens/timeout_ms）透传")

    # 左侧短路：只有「看起来像任务名」才值得查一次 RPC
    check(needs_task_list_lookup("") is False, "短路：空值不查 Host 任务清单")
    check(needs_task_list_lookup("  ") is False, "短路：空白不查 Host 任务清单")
    check(needs_task_list_lookup("gpt-4o-mini") is False, "短路：真模型名不查 Host 任务清单")
    check(needs_task_list_lookup("utils") is True, "短路：疑似任务名需要查清单确认")

    # 被拒名字解析
    check(rejected_model_name("未找到名为 'utils' 的模型") == "utils", "解析：单引号")
    check(rejected_model_name('未找到名为 "planner" 的模型') == "planner", "解析：双引号")
    check(rejected_model_name("未找到名为 “replyer” 的模型") == "replyer", "解析：中文引号")
    check(rejected_model_name("接口超时") == "", "解析：无关错误返回空串")
    check(rejected_model_name("") == "", "解析：空输入返回空串")

    # 日志渲染
    check(describe_kwargs({}) == "(无显式参数)", "日志：空参渲染")
    rendered = describe_kwargs({"task_name": "utils", "model": "", "temperature": 0.3})
    check("task_name='utils'" in rendered and "model=''" in rendered, "日志：渲染实际传出值")

    # 修复建议
    check("WebUI" in explain_missing_model("utils", None), "建议：取不到清单时指向 WebUI 模型列表")
    check("replyer" in explain_missing_model("planner", ["utils", "replyer"]),
          "建议：有清单时列出可用任务名")

    # 签名探测
    def bare_gen(prompt, model=""):
        ...

    def new_gen(prompt, model="", *, task_name="utils", model_name="", **kw):
        ...

    def wrapped_gen(prompt, model="", **kw):
        ...

    check(generate_supports_task_name(new_gen) is True, "探测：签名含 task_name → 新语义")
    check(generate_supports_task_name(bare_gen) is False, "探测：无 task_name 也无 **kwargs → 旧语义")
    check(generate_supports_task_name(wrapped_gen) is sdk_version_supports_task_name(),
          "探测：被 **kwargs 遮蔽时退回按 SDK 版本判断")
    check(isinstance(sdk_version_supports_task_name(), bool), "探测：SDK 版本判断返回布尔")
    check("utils" in KNOWN_TASK_NAMES, "白名单：含内置任务名 utils")


# ------------------------------------------------------------ 插件层

async def run_plugin_layer() -> None:
    from core.llm_params import generate_supports_task_name

    print("[2] 插件实际传出的参数")

    # ---- 真机回归：config.toml 里 summarize.task = "utils"，SDK ≥2.8.1 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True  # 真机 Host 1.2.5 / SDK 2.8.1
    plugin.config.summarize.task = "utils"  # 复刻真机配置，一字不改
    plugin.config.summarize.model = ""
    await plugin._call_llm("测试")
    check(llm.last.get("task_name") == "utils",
          "真机回归：任务名 utils 走 task_name 槽位")
    check(llm.last.get("model") == "",
          "真机回归：model 槽位留空（修复前这里是 'utils'，即报错根因）")

    # ---- 任务名 + 具体模型名 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "gpt-4o-mini"
    await plugin._call_llm("测试")
    check(llm.last.get("task_name") == "replyer" and llm.last.get("model") == "gpt-4o-mini",
          "分槽：任务名与具体模型名各走各的参数")

    # ---- 旧 SDK：任务名回落进 model ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = False  # SDK 2.8.0
    plugin.config.summarize.task = "utils"
    plugin.config.summarize.model = ""
    await plugin._call_llm("测试")
    check(llm.last.get("model") == "utils", "旧 SDK：任务名回落进 model 槽位")

    # ---- 误填纠偏：把任务名写进了 model 字段 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "utils"  # 错填任务名
    await plugin._call_llm("测试")
    check(llm.last.get("model") == "", "纠偏：误填的任务名没有传出去")
    check(llm.last.get("task_name") == "replyer", "纠偏：显式任务名不被误填字段覆盖")
    check(plugin._llm_ignore_model is True, "纠偏：置位停用标志")

    # ---- 纠偏不误伤真模型名 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "gpt-4o-mini"  # 真模型名，不在白名单
    await plugin._call_llm("测试")
    check(llm.last.get("model") == "gpt-4o-mini", "纠偏不误伤：真模型名原样传出")
    check(plugin._llm_ignore_model is False, "纠偏不误伤：未置位停用标志")

    # ---- 清单取不到时不纠偏（无法确认就不动手）----
    llm = _BoomLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "utils"
    await plugin._call_llm("测试")
    check(plugin._llm_ignore_model is False, "清单不可用时：不纠偏（无法确认就别动）")
    check(llm.last.get("model") == "utils", "清单不可用时：配置值原样传出")

    # ---- 软失败自愈：被拒的是 model ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "gpt-4o"  # 不在任务名白名单 → 会真的传出去
    llm.response = {"success": False, "error": "未找到名为 'gpt-4o' 的模型"}
    await plugin._call_llm("测试")
    check(plugin._llm_ignore_model is True, "软失败：被拒的 model 参数被停用")
    check(plugin._last_error.startswith("LLM 返回失败"), "软失败：错误被记为 _last_error")

    llm.calls.clear()
    llm.response = {"success": True, "response": "ok"}
    await plugin._call_llm("测试")
    check(llm.last.get("model") != "gpt-4o", "软失败：后续调用不再传被拒参数")

    # ---- 软失败自愈：被拒的是 task_name ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "planner"
    llm.response = {"success": False, "error": "未找到名为 'planner' 的模型"}
    await plugin._call_llm("测试")
    check(plugin._llm_ignore_task is True, "软失败：被拒的 task_name 被停用")

    llm.calls.clear()
    llm.response = {"success": True, "response": "ok"}
    await plugin._call_llm("测试")
    check(llm.last.get("task_name") != "planner", "软失败：后续调用不再传被拒任务名")

    # ---- 无关错误不误伤参数 ----
    llm = FakeLlm()
    llm.response = {"success": False, "error": "上游网关超时"}
    plugin = _make_plugin(llm)
    plugin._llm_supports_task_name = True
    plugin.config.summarize.task = "replyer"
    plugin.config.summarize.model = "gpt-4o"
    await plugin._call_llm("测试")
    check(plugin._llm_ignore_model is False and plugin._llm_ignore_task is False,
          "无关错误（超时）不误伤任何参数")

    # ---- 热重载复位自愈状态 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    plugin._llm_ignore_model = True
    plugin._llm_ignore_task = True
    plugin._llm_notice_reported = True
    plugin._llm_task_names_cache = (0.0, ["utils"])
    plugin._llm_supports_task_name = True
    await plugin.on_config_update("self", {}, "0.3.6")
    check(not plugin._llm_ignore_model and not plugin._llm_ignore_task,
          "热重载：停用标志被复位")
    check(plugin._llm_notice_reported is False, "热重载：告警去重位被复位")
    check(plugin._llm_task_names_cache is None, "热重载：任务清单缓存被清空")
    check(plugin._llm_supports_task_name is None, "热重载：签名探测结果被清空")

    # ---- 任务清单缓存与超时降级 ----
    llm = FakeLlm()
    plugin = _make_plugin(llm)
    first = await plugin._available_task_names()
    second = await plugin._available_task_names()
    check(first == second == ["utils", "replyer"], "清单：正常返回 Host 任务名")

    llm = _BoomLlm()
    plugin = _make_plugin(llm)
    check(await plugin._available_task_names() == [], "清单：RPC 失败时降级为空列表不抛异常")

    # ---- 真实 SDK 集成（用真 PluginContext + FakeHost）----
    print("[3] 真实 SDK 集成（payload 层）")
    module = load_plugin_module(PLUGIN_DIR)
    plugin = module.create_plugin()
    host = FakeHost(plugin_id="org.orge-8.web-search")
    ctx = build_context(host.plugin_id, rpc_call=host.rpc_call)
    bind_context(plugin, ctx, get_default_config(getattr(type(plugin), "config_model", None)))

    supports = generate_supports_task_name(ctx.llm.generate)
    print(f"      本地 SDK 探测：supports_task_name = {supports}")
    plugin._llm_supports_task_name = supports
    plugin.config.summarize.task = "utils"
    plugin.config.summarize.model = ""
    await plugin._call_llm("测试")
    args = host.calls_of("llm.generate")[-1]
    if supports:
        check(args.get("task_name") == "utils", "真 SDK：任务名进 task_name")
        check(args.get("model") == "", "真 SDK：model 槽位留空")
    else:
        check(args.get("model") == "utils", "真 SDK（旧版）：任务名回落进 model")


def main() -> int:
    from importlib.metadata import version

    try:
        sdk_ver = version("maibot-plugin-sdk")
    except Exception:
        sdk_ver = "(未安装)"
    print(f"maibot-plugin-sdk = {sdk_ver}")
    print(f"插件目录 = {PLUGIN_DIR}")

    test_pure_functions()
    asyncio.run(run_plugin_layer())

    print("-" * 62)
    if _fail:
        print(f"FAILED: {len(_fail)} / {_pass + len(_fail)}")
        for item in _fail:
            print("  -", item)
        return 1
    print(f"test_llm_params: OK ({_pass} 项断言通过)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
