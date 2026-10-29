# MTranServer-Rust

这是 [Mozilla Firefox Translations](https://github.com/mozilla/translations) 推理引擎的纯 Rust 实现! 参考 [MTranServer](https://github.com/xxnuo/MTranServer) 项目, 并提供了兼容的 API.

> **在线试用**  
> https://translate.ipacel.cc/

**主要特性**
- 原生 Rust 实现完整 Transformer 编解码器推理, 并进行了大量性能优化
- 支持 HTML、Markdown 等多行富文本格式直接翻译
- 通过语言规则改进翻译质量和识别准确性

## 使用

从 [Releases](https://github.com/ApliNi/MTranServer-Rust/releases) 下载对应平台的二进制文件.

或构建二进制文件:
```bash
cargo build --release
# 产物在 ./target/release/
```

直接启动即可运行:
```bash
# 首次翻译时将自动下载模型 (需要联网)
./MTranServer-Rust
```

内置 WebUI 可通过浏览器直接访问: `http://localhost:8989/`.

所有数据默认位于 `./data/` (可通过 `--data-dir` 修改):
```
data/
├── models/          # 模型文件 (按语言对子目录)
├── lang_rules/      # 语言规则覆盖
└── records.json     # Mozilla 模型记录缓存
```

## 命令行参数

| 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--host` | `MT_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `MT_PORT` | `8989` | 监听端口 (1-65535) |
| `--api-token` | `MT_API_TOKEN` | `""` | API 访问令牌, 为空时服务公开可用 |
| `--data-dir` | `MT_DATA_DIR` | `./data` | 数据目录  |
| `--worker-idle-timeout` | `MT_WORKER_IDLE_TIMEOUT` | `60` | 工作线程空闲超时 (秒) |
| `--cache-size` | `MT_CACHE_SIZE` | `2000` | 翻译结果缓存数量 |
| `--offline` | `MT_OFFLINE` | `false` | 离线模式, 不下载或更新模型 |
| `--ui` | `MT_ENABLE_UI` | `true` | 启用 Web UI |
| `--preprocess-symbols` | `MT_PREPROCESS_SYMBOLS` | `false` | 符号预处理 (驼峰/连字符/下划线拆词) |
| `--parse-formats` | `MT_PARSE_FORMATS` | `html,md` | 详见下方 [标记格式解析与投影](#标记格式解析与投影---parse-formats) |
| `--check-model-update` | `MT_CHECK_MODEL_UPDATE` | `true` | 模型更新检查, 使用时间参数 (如 `24h`、`1d`,) 定期检查 |
| `--keep-models` | `MT_KEEP_MODELS` | — | 常驻内存不卸载的模型, 特殊值 `all` 表示全部常驻 |
| `--keep-top-n` | `MT_KEEP_TOP_N` | `0` | 额外保留最常用的 N 个模型不卸载 |
| `--max-threads` | `MT_MAX_THREADS` | `0` | 推理程限制, 默认为 CPU 核心数 |
| `--max-sentence-length` | `MT_MAX_SENTENCE_LENGTH` | `512` | 单句最大字符数, 超长文本自动分段 |
| `--max-loaded-models` | `MT_MAX_LOADED_MODELS` | `0` | 最大同时加载模型数, 超限时立即卸载最不活跃的模型, `0` 不限制 |
| `--max-concurrency` | `MT_MAX_CONCURRENCY` | `0` | 最大并发请求数限制, `0` 不限制 |
| `--max-concurrency-per-ip` | `MT_MAX_CONCURRENCY_PER_IP` | `0` | 每 IP 最大并发请求数限制, `0` 不限制 |
| `--trust-proxy` | `MT_TRUST_PROXY` | `false` | 信任反向代理请求头 (`X-Real-IP`, `X-Forwarded-For`), 未开启时优先使用连接 Socket IP |
| `--max-request-body-size` | `MT_MAX_REQUEST_BODY_SIZE` | `512kb` | 最大请求体大小限制, 支持单位 `kb`、`mb` 或字节数 |
| `--max-batch-size` | `MT_MAX_BATCH_SIZE` | `512` | 批处理接口的单次处理数量限制, `0` 不限制 |
| `--log-level` | `MT_LOG_LEVEL` | `info` | 日志级别: `debug`、`info`、`warn`、`error` |
| `--log-requests` | `MT_LOG_REQUESTS` | `true` | HTTP 请求日志 |
| `--log-to-file` | `MT_LOG_TO_FILE` | `false` | 写日志文件到 `data/logs/` (按天自动切分 `mtran-YYYY-MM-DD.log`) |
| `--log-retention-days` | `MT_LOG_RETENTION_DAYS` | `7` | 日志文件保留天数, 为 `0` 不自动清理 |
| `--profile` | `MT_PROFILE` | `false` | 退出时输出引擎各区域耗时统计 |
| `--download` | — | — | 下载或校验指定模型后退出, 逗号/空格分隔, 特殊值 `all` 表示全部语言对 |
| `--languages` | — | — | 列出全部可用语言对与语言代码后退出 |
| `--bench-qps` | — | `en_zh-Hans` | 内置 QPS 基准: 对指定语言对满负荷压测 10 秒, 输出 QPS 与延迟后退出 |
| `--bench-qps-input-length` | — | `128` | `--bench-qps` 测试输入字符长度 |

布尔参数 (含环境变量) 统一接受 `true`/`false`/`1`/`0`; 布尔开关加 `=false` 或 `=0` 即可关闭, 如 `--offline=false`。

### 参数示例

```bash
# 下载或更新指定模型 (逗号或空格分隔)
./MTranServer-Rust --download en_zh-Hans zh-Hans_en

# 下载或更新全部模型
./MTranServer-Rust --download all

# 查看可用模型 (Language pairs)
./MTranServer-Rust --languages

# 启动后检查更新, 且每 24 小时再次检查一次
# 支持 `d` (天) 、`h` (小时) 后缀, 如 `24h`、`1d`; 不填单位默认为 `h` (如 `12` = 12 小时)
./MTranServer-Rust --check-model-update 24h

# 手动指定常用模型不被卸载
./MTranServer-Rust --keep-models en_zh-Hans zh-Hans_en

# 自动保持 5 个最常用模型不被卸载
./MTranServer-Rust --keep-top-n 5
```

### 标记格式解析与投影 `--parse-formats`

指定启用词对齐标记投影与结构保护的格式, 可选值:
- `html,md`: 默认
- `html`: 仅启用 HTML 标记解析与词对齐投影
- `md`: 仅启用 Markdown 标记解析与词对齐投影
- `false` 或 `0`: 关闭标记解析, 直接将原始文本透传给底层神经网络模型

词对齐标记投影机制说明：
- 语义完整性：将文档划分为块级句边界, 提取行内标记, 将干净整句送入 Marian Transformer 模型推理
- 跨语言倒装投影：基于 Transformer 最后一层交叉注意力矩阵输出的实时 Token 词对齐, 将行内标记动态投影回译文中对应的位置
- API 覆盖：客户端调用 `/translate` 时, 可通过请求体 JSON 传入 `parse` 覆盖后端 `--parse-formats` 设置

## API

完整 API 说明见 [`./docs/API.md`](./docs/API.md).

| 端点 | 方法 | 说明 |
|---|---|---|
| `/translate` | POST | 翻译文本 |
| `/translate/batch` | POST | 批量翻译 |
| `/detect` | POST | 语言检测 |
| `/languages` | GET | 支持的语言列表 |
| `/version` | GET | 版本号 |
| `/health` | GET | 健康检查 |
| `/__heartbeat__` | GET | 心跳检查 |

### 兼容 API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/deeplx` | POST | 兼容 DeepLX (推荐) |
| `/deepl` | POST | 兼容 DeepL API v2 |
| `/imme` | POST | 兼容 [沉浸式翻译](https://immersivetranslate.com/) |
| `/kiss` | POST | 兼容 [简约翻译](https://github.com/fishjar/kiss-translator) |
| `/hcfy` | POST | 兼容 [划词翻译](https://github.com/Selection-Translator/crx-selection-translate) |
| `/google/language/translate/v2` | POST | 兼容 Google Translate API v2 |
| `/google/translate_a/single` | GET | 兼容 Google translate_a/single |

> 注意: 兼容 API 始终使用 MTranServer-Rust 自动检测语言.

若开启鉴权, 可通过在 URL 中添加 `?token=MT_API_TOKEN` 进行授权, 也可通过各翻译客户端提供的鉴权方式填写密钥.

## 翻译质量差异跟踪

该文件包含一些测试用例, 并将翻译结果一并保存在该文件中, 以便后续对比翻译质量变化.

https://github.com/ApliNi/MTranServer-Rust/commits/main/test/cases/regression.json

## 性能基准

使用 `--bench-qps` 运行内置基准测试.

默认通过 `en_zh-Hans` 模型对 128 字符随机文本翻译进行 10 秒满负荷压测.

> 测试平台: Intel(R) Core(TM) Ultra X7 358H
```text
==== QPS Benchmark Result ====
pair            : en -> zh-Hans
input length    : 128 chars
workers         : 16
duration        : 10.03s
requests        : 12385 ok, 0 failed
QPS             : 1235.3
latency p50     : 12.70 ms
latency p95     : 23.96 ms
latency p99     : 26.96 ms
```

## 相关项目

- 原项目: [xxnuo/MTranServer](https://github.com/xxnuo/MTranServer), [Firefox Translations](https://github.com/mozilla/translations)
- 语言检测: [whichlang](https://github.com/quickwit-oss/whichlang), [whatlang](https://github.com/greyblake/whatlang-rs)
