# 联网搜索 (web-search)

给麦麦一个联网能力：把问题交给搜索引擎，抓回结果页正文，读完后给出**带来源引用**的答案。

参考了社区两个插件（[maibot-fetch-url-plugin](https://github.com/yufei-pan/maibot-fetch-url-plugin)、
[google_search_plugin](https://github.com/XXXxx7258/google_search_plugin)）的做法，
但做了两处关键取舍：**依赖压到最少**、**引擎默认顺序按国内网络实测调整**。

## 功能特性

- **多引擎降级搜索**：baidu / bing / duckduckgo / searxng / tavily，前一个失败自动换下一个，最终把失败原因一并告诉你
- **相关性检测**：搜索引擎把长查询拆散成单字匹配时（查「洛天依 2026 演唱会」返回汉字「洛」的字典），自动判定结果不可信——能降级就降级，降无可降就在回传内容里明确警告 LLM 并给出改查建议，而不是让它拿着垃圾材料空转
- **失败冷却**：挂掉的引擎在冷却期内被跳过，不会每次搜索都先去等它一次连接超时
- **正文抓取**：并发抓取前 N 条结果的正文，启发式剥离导航/广告/页脚；百度跳转链接在抓取时自动解开为真实地址
- **LLM 总结**：把材料整理成 500 字以内的中文答案，关键事实带 `[编号]` 来源标注
- **单链接抓取**：`fetch_page` 工具可抓指定 URL，长页面支持按字符区间分段读取
- **图片搜索直发**（v0.3.0）：`search_image` 工具搜索图片后直接把图发进聊天，LLM 拿到描述+来源清单做口头介绍；下载失败自动换下一张，带 SSRF 校验、魔数校验与双层体积上限
- **跨请求图片去重**（v0.3.3）：同一会话 30 分钟内发过的图（URL 指纹 + 内容 hash 双层）自动避开，「再来几张」不再重复发同一张
- **用户数量词兜底**（v0.3.4）：用户说「发一张」时代码直接按 1 张执行，LLM 乱传的 count 不生效（LLM 传参不可信原则）
- **结果缓存**：搜索与抓取结果内存缓存（默认 30 分钟），追问同一话题不再重复外呼
- **SSRF 防护**：默认拒绝内网、环回、云元数据地址；重定向**手动逐跳跟随、每一跳重新校验**（v0.3.5，可拦「中间跳进内网再跳回公网」的绕过路径）
- **诊断命令**：`/websearch status | test | clear`（别名「联网搜索」；v0.2.1 起不再劫持「搜一下」开头的自然语言消息）

## 工作流程

```text
用户提问
   ↓
web_search(query)
   ↓
① 引擎链搜索         baidu → bing → duckduckgo → …（失败或结果疑似无关即降级，带冷却）
   ↓
② 相关性判定         结果与查询词重合度过低 → 换引擎；全部不可信 → 取最好的一份并标记
   ↓
③ 并发抓取正文       前 3 条结果，逐条超时/体积/类型限制；解掉跳转链接
   ↓
④ 组装材料          按字符预算裁剪，标注编号与 URL
   ↓
⑤ LLM 阅读总结      注入当前日期，要求标注来源、不编造
   ↓
⑥ 返回答案 + 来源清单（低相关时附改查建议）
```

## 安装

1. 把整个 `web-search/` 目录放进 MaiBot 的 `plugins/` 下：

   ```bash
   cd <MaiBot 根目录>/plugins
   # 复制目录（或 git clone）
   ```

2. **完整重启 MaiBot**（manifest 校验发生在插件加载前，热重载不生效）。
   `httpx`、`beautifulsoup4` 会按 `_manifest.json` 的 `dependencies` 自动安装。

3. 在 WebUI（`http://127.0.0.1:8001`）插件管理中确认插件已加载并启用。

## 启用与验证

重启后先跑诊断命令确认链路：

```text
/websearch status          # 看引擎链、能力、缓存、模型任务名是否正常
/websearch test MaiBot     # 用真实网络逐个引擎实测，确认哪个通
```

`test` 会逐个报告「成功 N 条 / 耗时」或「失败原因」，这是排查"搜不到"最快的入口。

## 配置

配置文件由 Runner 首次启动时生成在插件目录下的 `config.toml`，也可在 WebUI 中编辑。

### `[plugin]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用插件 |
| `debug` | `false` | 输出调试日志（含每次搜索的引擎尝试过程） |
| `config_version` | `0.1.0` | 配置版本，与插件版本同步（隐藏项，勿手改） |

### `[search]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `order` | `["baidu", "bing", "duckduckgo"]` | 引擎优先级。**默认 baidu 在前**：实测 Bing 对「含数字 + 空格」的中文长查询会退化成单字匹配，百度则正常 |
| `max_results` | `6` | 每次搜索保留的结果条数 |
| `max_attempts` | `3` | 最多尝试几个引擎 |
| `engine_cooldown` | `300.0` | 引擎失败后的冷却秒数，0 表示不启用 |
| `timeout` | `15.0` | 单引擎请求超时（秒） |
| `language` | `zh-CN` | 语言偏好 |
| `proxy` | `""` | 代理地址，如 `http://127.0.0.1:7890`；留空跟随系统环境变量 |
| `user_agent` | Chrome UA | 请求 UA，默认已伪装常见浏览器 |

### `[fetch]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否抓取正文（关闭则只用搜索摘要，最快） |
| `max_pages` | `3` | 最多抓取前几条结果的正文 |
| `timeout` | `15.0` | 单页抓取超时（秒） |
| `max_bytes_mb` | `5.0` | 单页响应体积上限 |
| `per_page_chars` | `4000` | 单页正文送入 LLM 的字符上限 |
| `total_chars` | `12000` | 所有页面正文合计上限（控制上下文开销） |
| `concurrency` | `3` | 并发抓取数，过高易被站点限流 |
| `allow_private_networks` | `false` | **危险开关**：允许抓内网地址，仅内网调试时开启 |

### `[summarize]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否 LLM 总结（关闭则直接返回材料片段） |
| `task` | `replyer` | **模型任务名**，不是模型 ID。可选值用 `/websearch status` 查看 |
| `temperature` | `0.3` | 总结温度 |
| `max_tokens` | `1200` | 输出上限 |
| `rpc_timeout_ms` | `120000` | LLM 调用的 RPC 超时（毫秒）。Host 默认 30~60 秒，长材料会被截断，故显式放宽 |
| `max_output_chars` | `2000` | 总结结果回传字符上限 |

> ⚠️ `task` 不能留空。留空时 Host 会用 `plugin.<插件ID>` 作为任务名，
> 该任务未配置会回退到 embedding 模型并持续报 400。

### `[cache]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否缓存搜索结果与已抓取正文 |
| `ttl_seconds` | `1800` | 缓存有效期（秒） |
| `max_entries` | `128` | 条目上限（LRU 淘汰） |

### `[engines]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `searxng_base_url` | `""` | 自建 SearXNG 地址，如 `https://searx.example.com`（需实例开启 JSON 输出） |
| `tavily_api_key` | `""` | Tavily API Key。**敏感信息，勿提交到版本库** |

### `[image]`

| 配置项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用图片搜索工具 |
| `default_count` | `2` | 每次默认发送几张 |
| `max_count` | `4` | 单次发送张数上限（LLM 传参被截断到此值） |
| `candidate_pool` | `12` | 每次搜索抓取的候选图数量（给死链留余量） |
| `min_width` | `200` | 剔除宽度过小的图；尺寸未知（0）的放行 |
| `download_timeout` | `10.0` | 单张下载超时（秒） |
| `max_download_bytes_mb` | `8.0` | 单张下载体积上限，超出放弃该图换下一张 |
| `max_send_bytes_mb` | `4.0` | 单张发送体积上限（base64 过 RPC 膨胀 1.33 倍） |
| `allow_private_networks` | `false` | **危险开关**：允许下载内网图片地址 |
| `safe_search` | `true` | 预留位：百度接口无公开成人过滤参数，内容审核依赖 LLM 层措辞 |

## 工具用法（麦麦视角）

### `web_search`

```text
web_search(query, max_results=0, fetch_pages=true, focus="")
```

- `query`：自然语言关键词，例如「2026 年诺贝尔物理学奖得主」
- `max_results`：保留条数，0 表示用插件配置默认值
- `fetch_pages`：是否抓正文后再总结。只要快速标题列表时传 `false`
- `focus`：希望材料重点覆盖的方面，例如「价格和发布时间」

### `fetch_page`

```text
fetch_page(url, start_char=0, end_char=-1)
```

- 抓取指定页面并总结
- 长页面分页读取：先 `start_char=0, end_char=5000`，下一页 `start_char=5000`

### `search_image`

```text
search_image(query, count=0)
```

- 搜索图片并**直接发送到当前聊天**（百度图片引擎）
- `count`：发送张数，0=默认 2 张，上限 4
- 返回内容为已发送图片的描述+来源页清单，LLM 据此口头介绍，无需再发链接
- 降级链：缩略图 → 悬浮图 → 换下一张候选；全部失败时返回来源页链接兜底
- 已知边界：`fetch_page` 抓不了 JS 渲染站，图片站同理不受影响；百度图片风控与网页搜索独立（单独预热 Cookie）

## 诊断命令

| 命令 | 作用 |
|---|---|
| `/websearch` 或 `/websearch status` | 插件状态：版本、行为自检标记、引擎链、缓存统计、**可用模型任务名** |
| `/websearch test <关键词>` | 用真实网络逐个引擎实测，报告成功条数与耗时、失败原因 |
| `/websearch clear` | 清空搜索与抓取缓存 |

`status` 里会打印行为自检标记（`build=... / 提取器=...`）。这是为了应对一个真实陷阱：
**禁用/启用插件会重新执行 `plugin.py`，但依赖模块命中 `sys.modules` 缓存**——
新代码"看起来部署了"却不生效。看这个标记即可确认跑的是不是新版本。

## 权限与能力声明

`_manifest.json` 声明的能力（与源码里实际用到的 `ctx.*` 调用一一对应）：

| 能力 | 用途 |
|---|---|
| `llm.generate` | 阅读材料后生成总结 |
| `llm.get_available_models` | 诊断命令列出可用模型任务名 |
| `send.text` | 诊断命令回显结果 |
| `send.image` | `search_image` 发送图片到聊天（v0.3.0 新增） |

本插件**不需要** Napcat 适配器，也不调用任何适配器 API；不申请任何数据库或文件系统能力。

依赖：`httpx >= 0.27`、`beautifulsoup4 >= 4.12`。

## 故障排查

| 症状 | 原因与处置 |
|---|---|
| 搜索一直失败，提示所有引擎不可用 | 先跑 `/websearch test 测试`。国内网络下 duckduckgo 失败属正常；需要代理时填 `search.proxy` |
| 提示"触发百度安全验证" | 百度频率风控，通常几分钟内自动恢复。插件会自动降级到下一个引擎并进入冷却；频繁出现说明请求过密，可加大 `engine_cooldown` 或降低 `fetch.concurrency` |
| 返回结果与问题明显无关 | 搜索引擎把长查询拆散了（Bing 对含数字的中文多词查询尤其明显）。回传内容会附「检索质量提示」引导 LLM 改用短实体词重搜；也可把 `search.order` 保持 baidu 在前 |
| 报 `CERTIFICATE_VERIFY_FAILED` | 运行机器存在 TLS 中间人代理（如 `proxy-root-ca.cer`）。填 `search.proxy` 指向代理，或把该根证书导入系统信任链 |
| 搜索成功但没有正文 | 目标站点反爬或需登录。可降低 `fetch.max_pages`、在 `search.order` 追加 tavily（自带正文），或接受只用摘要 |
| 正文抓到一堆导航文字 | 站点结构特殊，启发式提取失效。可关闭 `fetch.enabled` 退化为摘要模式，或提 issue 附上 URL |
| 总结报 400 或刷 embedding 相关错误 | `summarize.task` 配错或留空。跑 `/websearch status` 看"可用模型任务名"，改成列表中的值 |
| 总结超时 | 加大 `summarize.rpc_timeout_ms`，或减小 `fetch.max_pages` / `fetch.total_chars` 降低材料量 |
| 每次搜索都慢十几秒 | 某个引擎连接超时。检查 `engine_cooldown` 是否被设为 0，或把慢引擎从 `order` 里移除 |
| 图片搜不到 / 提示图片搜索失败 | 百度图片风控或关键词过冷。换常见关键词重试；`/websearch status` 确认"图片搜索：开" |
| 图片全部下载或发送失败 | 缩略图死链或宿主 send.image 异常。工具会返回来源页链接兜底；频繁出现可加大 `image.candidate_pool` |
| 修改配置后不生效 | manifest 改动需**完整重启** MaiBot；纯配置项改动热重载即可（插件会重建引擎链与抓取器） |

## 开发与测试

```bash
# 交付门禁（静态自检 + 冒烟 + 单测）
../../../gate.sh web-search

# 或分别运行
python check_plugin.py --plugin .
python tests/smoke_test.py          # 生命周期 + 三条主链路（离线）
python tests/test_pipeline.py       # 42 项离线单测
python -m pytest -q tests           # 同上（pytest 风格）

# 联网探测（在目标机器上跑，确认引擎与抓取可用）
python tests/network_probe.py "关键词"
SEARXNG_BASE_URL=https://... TAVILY_API_KEY=... python tests/network_probe.py
```

测试设计说明：`smoke_test.py` 与 `test_pipeline.py` **完全离线且确定性**——
外部引擎与抓取器被 stub 替换，因此不会因为网络抖动误报。网络可用性交给
`network_probe.py` 单独验证，这样"代码坏了"和"网不通"能被区分开。

## 设计取舍

1. **只用 2 个依赖**（httpx + beautifulsoup4）。同类插件常用 trafilatura /
   readability-lxml / lxml 做正文提取，效果略好，但它们需要编译，在真机 Windows 上
   安装失败的概率不低——而依赖装不上等于整个插件不可用。这里用
   BeautifulSoup + 启发式规则自己实现，实测提取质量够用。
2. **不做 Google 引擎**。国内网络下 Google 直连不可达，加它只会让降级链多一次超时。
3. **baidu 首位是实测换来的教训**。真机日志显示，LLM 用「洛天依 2026年 活动 演唱会」
   这类自然问法搜索时，Bing 把查询拆散成单字匹配（返回汉字「洛」的字典、洛谷、
   LOL 英雄），LLM 拿着垃圾材料连续重试了 9 轮、空转 200+ 秒。横向实测确认：
   同一查询百度返回的正是 2026 巡回演唱会的各站信息。故 baidu 置于首位，
   并新增相关性检测兜底——即使引擎给了结果，也要先确认结果对不对题。
4. **工具默认进 deferred 池**。SDK 2.8.0 没有 `core_tool` 字段，插件侧无法强制常显；
   实测 tool_search 能正常发现本工具（真机日志：`已找到 1 个 deferred tools:
   web_search（本次新发现）`）。

## License

MIT
