# MTranServer 历史遗留、兼容性设计与协议规范

本文档系统性记录 MTranServer-Rust 中的向后兼容设计、参数别名、协议仿真规范以及保留的历史遗留逻辑。为防止后续迭代中将必要的兼容性设计误判为“死代码”或“冗余逻辑”，特此设立本规范。

---

## 目录

1. [第三方翻译协议兼容规范（强制保留）](#1-第三方翻译协议兼容规范强制保留)
   - [1.1 协议入参静默转发与“auto”强转机制](#11-协议入参静默转发与auto强转机制)
   - [1.2 Google 翻译兼容接口](#12-google-翻译兼容接口)
   - [1.3 DeepL / DeepLX 兼容接口](#13-deepl--deeplx-兼容接口)
2. [配置别名与向后兼容设计](#2-配置别名与向后兼容设计)
   - [2.1 标记解析格式别名 (parse_markup_bool)](#21-标记解析格式别名-parse_markup_bool)
   - [2.2 语言代码别名与规范化 (ALIASES)](#22-语言代码别名与规范化-aliases)
   - [2.3 多源认证凭证回退机制 (Multi-Scheme Auth)](#23-多源认证凭证回退机制-multi-scheme-auth)
   - [2.4 CLI 与环境变量容错解析](#24-cli-与环境变量容错解析)
3. [模型系统与推理解码兼容规范](#3-模型系统与推理解码兼容规范)
   - [3.1 no_repeat_ngram_size 生成防重复机制](#31-no_repeat_ngram_size-生成防重复机制)
   - [3.2 语种检测 Tier 3 问候词快速拦截 (greetings)](#32-语种检测-tier-3-问候词快速拦截-greetings)
   - [3.3 单一词表与拆分词表双模支持](#33-单一词表与拆分词表双模支持)
   - [3.4 遗留模型版本推断机制 (infer_installed_version)](#34-遗留模型版本推断机制-infer_installed_version)
4. [低收益与评估期遗留项目归档](#4-低收益与评估期遗留项目归档)
   - [4.1 探针别名 /_heartbeat__](#41-探针别名-_heartbeat__)
   - [4.2 错误体系收敛说明](#42-错误体系收敛说明)
   - [4.3 基础通用库的人类友好 API (SentencePiece / Cache)](#43-基础通用库的人类友好-api-sentencepiece--cache)

---

## 1. 第三方翻译协议兼容规范（强制保留）

为确保生态中基于 Google Translate、DeepL 和 DeepLX 的现成客户端（如各类浏览器划词扩展、沉浸式翻译、第三方插件等）能够零配置直接指向本服务端，系统实现了高保真仿真层。

### 1.1 协议入参静默转发与“auto”强转机制

在以下第三方协议结构体中，源语言参数必须完整保留：

- `GoogleRequest.source` (`POST /google/language/translate/v2`)
- `GoogleSingleQuery.sl` (`GET /google/translate_a/single`)
- `DeeplRequest.source_lang` (`POST /deepl`)
- `DeeplxRequest.source_lang` (`POST /deeplx`)

**设计约束**：
- **反序列化保真**：客户端在发起请求时通常必传上述字段。若在 Rust 结构体中移除这些字段或标记为非可选导致解析失败，会破坏既有客户端。
- **服务端统一自动检测**：服务端运行时统一将源语言参数路由至 `"auto"`，由内置的高性能语言检测管道（Tier 0 专用字符/脚本 -> Tier 1 问候词与特征词 -> Tier 2 whatlang 统计模型）决定真实源语言。这一设计解决了外部客户端传入错误语种代码导致翻译失败的问题。

### 1.2 Google 翻译兼容接口

1. **`POST /google/language/translate/v2`**：
   - 兼容 Google Cloud Translation API v2。
   - 响应格式输出 `{ "data": { "translations": [{ "translatedText": "...", "detectedSourceLanguage": "..." }] } }`。
2. **`GET /google/translate_a/single`**：
   - 兼容 Google Web / Chrome 扩展划词接口。
   - 响应为 8 元组嵌套无结构 JSON 数组：`[[["译文", "原文", null, null, 1]], null, "zh-CN", null, null, null, null, []]`。
   - 必须维持 `GoogleSingleResponse` 的特定元组结构与 `Option<()>` 填充，禁止重构成普通对象。

### 1.3 DeepL / DeepLX 兼容接口

1. **`POST /deepl`**：
   - 兼容 DeepL API v2。
   - BCP-47 语种代码需经由 `convert_bcp47_to_deepl` 转为 DeepL 专用的大写格式（如 `ZH`, `EN`, `NB`, `ZH-TW` 等）。
2. **`POST /deeplx`**：
   - 兼容开源 DeepLX 协议。
   - 响应包含 DeepLX 客户端校验所需的 `code: 200`、`method: "Free"`、`alternatives: []` 及纳秒时间戳 `id`。

---

## 2. 配置别名与向后兼容设计

### 2.1 标记解析格式别名 (parse_markup_bool)

在早期版本中，HTML 与 Markdown 解析曾使用格式字符串区分。当前引擎已将二者统一为无损内联语法树管道。为保持向后兼容：

- **真值映射**：`"true"`, `"1"`, `"html"`, `"md"`, `"html,md"`, `"md,html"`, `"all"`, `"yes"`, `"on"` 均映射为 `true`。
- **假值映射**：`"false"`, `"0"`, `"none"`, `"raw"`, `"off"`, `"no"`, `"disable"`, `"text"` 均映射为 `false`。
- **多态反序列化**：通过 `parse_markup_from_json_value` 兼容布尔值、数字 (`1`/`0`)、字符串及 `null`（默认为全局配置值）。

### 2.2 语言代码别名与规范化 (ALIASES)

`src/utils/lang.rs` 中的 `ALIASES` 静态映射表包含了 70+ 个条目，涵盖：
- **历史过时代码**：如 Java/Android 遗留代号（`iw` -> `he`, `in` -> `id`, `jw` -> `jv`）。
- **三位 ISO 639-2/3 代码**：如 `eng`, `cmn`, `zho`, `fra`, `deu`, `rus`, `jpn` 等。
- **常见地区及口语简称**：如 `zh-cn`, `zh-tw`, `en-us`, `chinese`, `cht`, `jp`, `kr` 等。

该映射属于前端与各方客户端接入时的关键缓冲层，不可轻易削减。

### 2.3 多源认证凭证回退机制 (Multi-Scheme Auth)

`src/utils/auth.rs` 支持按以下优先级逐级回退提取 Token：
1. `Authorization: Bearer <token>`（标准 OAuth / REST）
2. `Authorization: DeepL-Auth-Key <token>`（兼容 DeepL 官方 SDK）
3. URL Query Parameter: `api_token`
4. URL Query Parameter: `token`
5. HTTP Header: `x-api-token`

### 2.4 CLI 与环境变量容错解析

- **模型保持列表**：`resolve_keep_models_pairs` 容许以逗号、空格、连字符、下划线混合分割语言对（如 `zh-en,es_en en-fr`），并支持全选关键字 `"all"`。
- **存储大小单位**：`parse_size` 支持不区分大小写及带空格的容量后缀（`b`, `k`, `kb`, `m`, `mb`, `g`, `gb`）。
- **模型自动更新频率**：`resolve_model_update` 支持纯布尔开关、整数小时（`12`）或带单位的人类可读周期（`24h`, `1d`）。

---

## 3. 模型系统与推理解码兼容规范

### 3.1 no_repeat_ngram_size 生成防重复机制

在全量 57 个语言规则配置 `lang_rules/*.toml` 的 `[generation]` 分区中，均定义了 `no_repeat_ngram_size`。

**关键机制说明**：
- 该参数被加载进 `RuleProfile.generation.no_repeat_ngram_size`，并在 `src/engine/transformer.rs:1145` 映射为 `DecodingConfig.no_repeat_ngram_size`。
- 推理核心算子 `apply_penalties_with_exemptions_fast` 依据此阈值动态屏蔽生成序列中重复出现的 token n-gram，有效阻断退化自重复死循环。
- 对于高密度的 CJK 语言（中文 `zh`、日语 `ja`、韩语 `ko`），`transformer.rs` 在内部自动保底提升至 `5`，以避免过度惩罚常用成语与固定搭配。
- **结论**：该配置是语言规则的有效核心参数，禁止从 TOML 中作为死代码移除。

### 3.2 语种检测 Tier 3 问候词快速拦截 (greetings)

- 语言检测流水线具备 Tier 3 快速拦截机制：对于简短输入，优先通过短语前缀判定语种。
- `lang_rules/en.toml` 中配置了专用常用问候词，配合 `src/engine/rules/detect.rs` 内的 `CORE_GREETINGS` 静态查找表，在不触发重型统计模型的情况下实现常数时间判定。

### 3.3 单一词表与拆分词表双模支持

Firefox Translations 项目早期模型采用单一整体词表 `vocab`，而较新版本的模型则将词表拆分为源语言词表 `srcvocab` 与目标语言词表 `trgvocab`。

`src/models/records.rs` 显式保留了双模探查逻辑：
```rust
let has_split = installed.contains_key("srcvocab") && installed.contains_key("trgvocab");
let has_unified = installed.contains_key("vocab");
```
这保证了用户无论是下载最新模型还是挂载离线旧版模型，均可正常加载运行。

### 3.4 遗留模型版本推断机制 (infer_installed_version)

对于引入 `version.json` 元数据管理机制之前就已经下载落盘的模型文件夹，`infer_installed_version` 会读取目录下的 `.bin` 权重文件和词表，计算其实际哈希值并比对远程 `records.json`，自动补全版本信息，防止旧版用户升级后引发重复重新下载。

---

## 4. 低收益与评估期遗留项目归档

### 4.1 探针别名 /_heartbeat__

- **现状**：`src/api/mod.rs` 中同时注册了 `/__heartbeat__`（标准）与 `/_heartbeat__`（少一个下划线）。
- **说明**：单下划线版本系历史拼写手误遗留，由于 HTTP 路由注册成本极低，暂保留作为向下兼容探针。若外部基础设施均已切换为 `/health` 或 `/__heartbeat__`，可在主版本迭代时废弃。

### 4.2 错误体系收敛说明

- `ApiError::Unauthorized`：由于接口鉴权已在全局中间件层拦截完毕，路由处理函数内部不再主动构建该枚举，仅在 `IntoResponse` 映射中作为兜底保留。
- `ServiceError::Timeout`：服务超时当前均由外层 Tokio/Axum 机制接管并直接返回对应 HTTP 状态码，服务层不再显式包裹此枚举。

### 4.3 基础通用库的人类友好 API (SentencePiece / Cache)

- `SentencePiece::encode` 与 `decode`：虽然推理热路径全面使用了无分配的 `encode_into` / `decode_into`，但基础 `encode` / `decode` 仍然是通用的标准 API，供单元测试、离线脚本与轻量调试场景调用。
- `cache` 状态检查：`cache::invalidate`、`contains`、`len`、`memory_usage` 供运维监控与并发审计测试使用，予以保留。
