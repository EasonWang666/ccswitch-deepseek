# cswitch-deepseek

让 Codex CLI / Codex 桌面客户端通过 DeepSeek 模型运行。

Codex 使用 Responses API 协议，而 DeepSeek 只提供 Chat Completions API。本项目在本地启动一个协议翻译代理，在两者之间无缝转换。

## 架构

```
Codex 客户端 ──Responses API──▶ app.js :11435 ──Chat API──▶ api.deepseek.com
                                  协议翻译
```

## 前置条件

- Node.js >= 18
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/api_keys)）

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置 API Key

编辑 `.env`：

```
api_key=sk-your-deepseek-api-key
```

### 3. 启动代理服务

```bash
npm start
```

输出：

```
========================================
  Codex DeepSeek Proxy (Node.js)
========================================
  Address:   http://127.0.0.1:11435
  Endpoint:  http://127.0.0.1:11435/v1/responses
  Upstream:  DeepSeek API
  Model:     deepseek-v4-pro
========================================
```

### 4. 配置 CCSwitch

CCSwitch 桌面应用中，API 地址填写：

```
http://127.0.0.1:11435/v1
```

CCSwitch 会引导 Codex 客户端将请求发送到本地代理。

## Codex CLI 用户

如果直接使用 Codex CLI（不通过 CCSwitch），编辑 `~/.codex/config.toml`：

```toml
[model_providers.deepseek]
base_url = "http://127.0.0.1:11435/v1"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 300000

[profiles.deepseek-v4-pro]
model_provider = "deepseek"
model_name = "deepseek-v4-pro"
context_window = 1000000
max_output_tokens = 32768

[profiles.deepseek-v4-pro.features]
tool_search = false
tool_search_always_defer_mcp_tools = false
```

使用：

```bash
codex --profile deepseek-v4-pro
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | - | DeepSeek API Key（必填） |
| `DEEPSEEK_PROXY_HOST` | `127.0.0.1` | 代理监听地址 |
| `DEEPSEEK_PROXY_PORT` | `11435` | 代理监听端口 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 默认模型 |
| `DEEPSEEK_THINKING` | `disabled` | 思考模式（`enabled` / `disabled`） |
| `DEEPSEEK_REASONING_EFFORT` | `medium` | 推理深度（`low` / `medium` / `high`） |

## 功能

- **协议翻译**：Responses API ↔ Chat Completions 双向转换
- **工具过滤**：DeepSeek 限制 128 个工具，超出时自动按域名关键词优先级裁剪
- **命名空间处理**：自动处理 MCP 工具命名空间（`gmail___search_emails` 等）
- **DSML 恢复**：修复 DeepSeek 将工具调用以纯文本格式泄露的问题
- **角色映射**：自动将 OpenAI `developer` role 映射为 `system`
- **内容格式翻译**：`input_text` / `output_text` → `text`

## License

ISC
