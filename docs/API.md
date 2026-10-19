# MTranServer-Rust API

MTranServer-Rust 提供的 HTTP API 端点说明。服务默认监听 `0.0.0.0:8989`。

## 鉴权

通过 `--api-token` 参数或 `MT_API_TOKEN` 环境变量设置访问令牌, 默认值为空, 表示服务完全公开可用。

配置令牌后, 除前端资源与心跳端点外的所有 API 均需携带令牌, 令牌缺失或不匹配时返回 `401`:

- 公开端点 (无需令牌): `/version`、`/health`, 以及 Web UI 静态资源
- 受保护端点 (其余全部, 含 `/translate` `/detect` `/languages` 与全部兼容插件端点): 必须提供有效令牌

受保护端点的令牌支持四种传递方式, 按以下优先级依次尝试, 取第一个命中的值:

| 优先级 | 传递方式 | 示例 |
|---|---|---|
| 1 | `Authorization` 请求头 (自动剥离 `Bearer ` 或 `DeepL-Auth-Key ` 前缀) | `Authorization: Bearer <token>` |
| 2 | 查询参数 `api_token` | `?api_token=<token>` |
| 3 | 查询参数 `token` | `?token=<token>` |
| 4 | `x-api-token` 请求头 | `x-api-token: <token>` |

## API 端点

### 翻译 — `/translate`

**POST** `http://localhost:8989/translate`

请求体:

```json
{
  "from": "en",
  "to": "zh-Hans",
  "text": "Hello world"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `from` | string | 是 | 源语言代码,可用 `"auto"` 自动检测 |
| `to` | string | 是 | 目标语言代码 |
| `text` | string | 是 | 待翻译文本 |
| `parse` | string / bool / int | 否 | 标记解析格式: `html,md`、`html`、`md`、`false`、`0` |

> 复合文档与富文本标记（包含 YAML Frontmatter、代码围栏、表格、HTML 块级容器/注释、LaTeX 数学公式、HTML 实体及各种行内标记）由统一 Hybrid AST 流水线自动解析与保护，行内标记通过 Marian Transformer Cross-Attention 词对齐动态投影回译文语序。未传入 `parse` 参数时使用后端全局 `--parse-formats` 配置（默认 `html,md`）。
成功响应:

```json
{"result": "你好，世界", "time_cost": 50}
```

`time_cost` 为翻译推理耗时(整数毫秒)，不包括模型下载与加载时间 (若命中缓存则为 0)。

当请求的 `from` 为 `"auto"` 时, 响应额外携带检测出的源语言代码(检测不到时省略):

```json
{"result": "你好，世界", "detected_language": "en", "time_cost": 50}
```

### 批量翻译 — `/translate/batch`

**POST** `http://localhost:8989/translate/batch`

请求体:

```json
{
  "from": "en",
  "to": "zh-Hans",
  "texts": ["Hello world", "Good morning"]
}
```
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `from` | string | 是 | 源语言代码,可用 `"auto"` 自动检测 |
| `to` | string | 是 | 目标语言代码 |
| `texts` | string[] | 是 | 待翻译文本数组 |
| `parse` | string / bool / int | 否 | 按需指定或关闭富文本解析（同 `/translate`） |

成功响应:
```json
{"results": ["你好，世界", "早上好"]}
```

当请求的 `from` 为 `"auto"` 时, 响应额外携带 `detected_languages`, 与 `results` 逐项对应(检测不到的项为 `null`):

```json
{"results": ["你好，世界", "早上好"], "detected_languages": ["en", "en"]}
```

### 语言检测 — `/detect`

**POST** `http://localhost:8989/detect`

请求体:

```json
{
  "text": "Hello world",
  "minConfidence": 0.5
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | 是 | 待检测文本 |
| `minConfidence` | float | 否 | 最低置信度阈值 |

成功响应有两种形态。传入 `minConfidence` 时返回置信度:

```json
{"language": "en", "confidence": 0.95}
```

未传 `minConfidence` 时不计算置信度, 响应中的 `confidence` 字段可能省略:

```json
{"language": "en"}
```

### 语言列表 — `/languages`

**GET** `http://localhost:8989/languages`

返回支持的语言代码和可用语言对。

响应:

```json
{
  "languages": ["en", "zh-Hans", "ja", ...],
  "pairs": [
    {"from": "en", "to": "zh-Hans"},
    {"from": "zh-Hans", "to": "en"},
    ...
  ]
}
```

### 系统状态

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/version` | 返回版本号 |
| GET | `/health` | 健康检查,返回 `{"status":"ok"}` |
| GET | `/__heartbeat__` | 心跳检查,返回空 200 |

## 兼容性 API 端点

服务端兼容以下第三方翻译 API 的请求格式,可直接替换使用。

### 极简翻译 — `/kiss`

**POST** `http://localhost:8989/kiss`

请求体:

```json
{
  "from": "en",
  "to": "zh-Hans",
  "text": "Hello world"
}
```

单条响应:

```json
{"text": "你好，世界", "src": "en"}
```

支持 `texts` 数组批量:

```json
{"texts": ["Hello", "World"]}
```

批量响应:

```json
{"translations": [{"text": "你好", "src": "en"}, {"text": "世界", "src": "en"}]}
```

`texts` 为非空数组时进入批量分支并返回 `translations`; 否则读取 `text` 字段, 两者均缺失时返回 `500`。

### DeepL — `/deepl`

**POST** `http://localhost:8989/deepl`

兼容 DeepL API v2 格式:

```json
{
  "text": "Hello world",
  "source_lang": "EN",
  "target_lang": "ZH"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string \| string[] | 是 | 待翻译文本, 支持单个字符串或字符串数组 |
| `source_lang` | string | 否 | 源语言, 缺省时自动检测 |
| `target_lang` | string | 是 | 目标语言 |
| `tag_handling` | string | 否 | 保留字段以兼容 DeepL 客户端 (HTML/XML 标签现已全自动保留) |

鉴权遵循全局规则 (见顶部「鉴权」章节); DeepL 客户端发送的 `DeepL-Auth-Key` 前缀会被自动剥离。

响应:

```json
{
  "translations": [
    {"detected_source_language": "EN", "text": "你好，世界"}
  ]
}
```

### DeepLx — `/deeplx`

**POST** `http://localhost:8989/deeplx`

兼容 DeepLx 格式:

```json
{
  "text": "Hello world",
  "source_lang": "EN",
  "target_lang": "ZH"
}
```

鉴权遵循全局规则 (见顶部「鉴权」章节), 兼容 `Authorization: Bearer <token>`、裸 `Authorization` 头或查询参数 `?token=<token>`。

响应:

```json
{
  "alternatives": [],
  "code": 200,
  "data": "你好，世界",
  "id": 1234567890,
  "method": "Free",
  "source_lang": "EN",
  "target_lang": "ZH"
}
```

### 划词翻译 — `/hcfy`

**POST** `http://localhost:8989/hcfy`

兼容划词翻译扩展格式:

```json
{
  "text": "Hello world",
  "destination": ["中文(简体)"],
  "source": "英语"
}
```

语言代码支持中文名称: `"英语"`、`"中文(简体)"`、`"日语"` 等。

响应:

```json
{
  "text": "Hello world",
  "from": "英语",
  "to": "中文(简体)",
  "result": ["你好，世界"]
}
```

### Google Translate — `/google`

**POST** `http://localhost:8989/google/language/translate/v2`

兼容 Google Translate API v2 格式:

```json
{
  "q": "Hello world",
  "source": "en",
  "target": "zh-Hans",
  "format": "text"
}
```

`q` 支持字符串或字符串数组。

响应:

```json
{
  "data": {
    "translations": [
      {"translatedText": "你好，世界", "detectedSourceLanguage": "en"}
    ]
  }
}
```

**GET** `http://localhost:8989/google/translate_a/single?q=Hello&sl=en&tl=zh-Hans`

兼容旧版 Google Translate 格式:

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `q` | string | 是 | 待翻译文本 |
| `tl` | string | 是 | 目标语言 |
| `sl` | string | 否 | 源语言, 缺省时自动检测 |

响应:

```json
[[["你好，世界","Hello",null,null,1]],null,"en",null,null,null,null,[]]
```

### 浏览器插件 — `/imme`

**POST** `http://localhost:8989/imme`

兼容 Immersive Translate 扩展格式:

```json
{
  "source_lang": "en",
  "target_lang": "zh-Hans",
  "text_list": ["Hello world", "Good morning"]
}
```

鉴权遵循全局规则 (见顶部「鉴权」章节), 兼容查询参数 `?token=<token>`。

`source_lang` 为源语言(仅用于标识, 翻译时始终自动检测语言); `text_list` 中的空字符串条目直接返回空文本, 单条翻译失败时该条目返回原文。

```json
{
  "translations": [
    {"detected_source_lang": "en", "text": "你好，世界"},
    {"detected_source_lang": "en", "text": "Good morning"}
  ]
}
```

## 错误响应

发生错误时, 除下面列出的特殊情况外, 响应体均为如下 JSON 结构:

```json
{
  "error": "Internal Server Error",
  "message": "Model file not found for en -> zz"
}
```

| 状态码 | 出现时机 | 响应体 |
|---|---|---|
| `200` | 成功 | 各端点定义的正常响应 |
| `400` | 请求体缺失、JSON 无法解析或字段类型不符合要求; 批量翻译超过最大批量限制 (默认 `512` 条, 由 `--max-batch-size` 控制) 或累计 `65536` 字符; 非字符串数组元素 | 上述 JSON 结构 |
| `401` | 受保护端点鉴权失败 | 恰为 `{"error":"Unauthorized"}` |
| `404` | 请求了未注册的路径 | 纯文本 `404` (`text/plain`) |
| `413` | 请求体大小超过配置的最大限制 (默认 `512kb`, 由 `--max-request-body-size` 控制) | 纯文本错误提示 (`text/plain`) |
| `429` | 当前全局或单 IP 活跃并发请求数达到最大限制 (由 `--max-concurrency` / `--max-concurrency-per-ip` 控制) | 上述 JSON 结构 |
| `500` | 服务端处理失败: 模型缺失、引擎错误等 | 上述 JSON 结构 |

补充说明:

- `error` 为错误类别, `message` 为错误信息
- `2xx` 成功与 `3xx` 重定向响应不做任何改写
