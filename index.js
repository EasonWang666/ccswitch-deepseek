import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

// ============================================================
// CCSwitch + DeepSeek-V4-Pro 配置
// ============================================================
// 两种模式:
//   1. CCSwitch 代理模式: CCSwitch 桌面应用启动代理后，本地转发到 DeepSeek
//   2. 直连模式: 直接调用 DeepSeek API (无需 CCSwitch)
// ============================================================

const DEEPSEEK_API_KEY = process.env.api_key ?? "";
const DEEPSEEK_BASE_URL = "https://api.deepseek.com";

// CCSwitch 本地代理默认地址 (Codex 应用使用 /v1 路径)
const CCSWITCH_PROXY_URL = "http://127.0.0.1:15721/v1";

// ============================================================
// 创建客户端
// ============================================================

/**
 * 创建直连 DeepSeek 的 OpenAI 兼容客户端
 */
function createDeepSeekClient(apiKey = DEEPSEEK_API_KEY) {
  return new OpenAI({
    baseURL: DEEPSEEK_BASE_URL,
    apiKey,
  });
}

/**
 * 创建通过 CCSwitch 代理的 OpenAI 兼容客户端
 */
function createCCSwitchClient(apiKey = DEEPSEEK_API_KEY) {
  return new OpenAI({
    baseURL: CCSWITCH_PROXY_URL,
    apiKey,
  });
}

// ============================================================
// 核心调用
// ============================================================

/**
 * 调用 DeepSeek-V4-Pro (非流式)
 *
 * @param {OpenAI} client        - OpenAI 客户端实例
 * @param {object}  opts         - 可选配置
 * @param {string}  opts.model   - 模型名, 默认 deepseek-v4-pro
 * @param {object[]} opts.messages - 消息列表
 * @param {boolean} opts.thinking - 是否启用深度思考
 * @param {string}  opts.reasoningEffort - 推理深度: low/medium/high
 */
async function chat(client, opts = {}) {
  const {
    model = "deepseek-v4-pro",
    messages = [{ role: "user", content: "Hello!" }],
    thinking = false,
    reasoningEffort = "medium",
  } = opts;

  const body = { model, messages, stream: false };

  if (thinking) {
    body.thinking = { type: "enabled" };
    body.reasoning_effort = reasoningEffort;
  }

  const completion = await client.chat.completions.create(body);
  return completion.choices[0].message;
}

/**
 * 调用 DeepSeek-V4-Pro (流式输出)
 *
 * @param {OpenAI} client          - OpenAI 客户端实例
 * @param {object}  opts           - 可选配置
 * @param {function} [opts.onToken]  - 正文 token 回调
 * @param {function} [opts.onReasoning] - 思考过程 token 回调
 */
async function chatStream(client, opts = {}) {
  const {
    model = "deepseek-v4-pro",
    messages = [{ role: "user", content: "Hello!" }],
    thinking = false,
    reasoningEffort = "medium",
    onToken = (text) => process.stdout.write(text),
    onReasoning = null,
  } = opts;

  const body = { model, messages, stream: true };

  if (thinking) {
    body.thinking = { type: "enabled" };
    body.reasoning_effort = reasoningEffort;
  }

  const stream = await client.chat.completions.create(body);

  let fullContent = "";
  let fullReasoning = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta;
    if (delta?.reasoning_content) {
      fullReasoning += delta.reasoning_content;
      onReasoning?.(delta.reasoning_content);
    }
    if (delta?.content) {
      fullContent += delta.content;
      onToken(delta.content);
    }
  }
  return { content: fullContent, reasoning: fullReasoning };
}

// ============================================================
// 高层 API: 预设场景
// ============================================================

/**
 * 判断是否为连接被拒错误
 */
function isConnectionRefused(err) {
  return (
    err?.code === "ECONNREFUSED" ||
    err?.cause?.cause?.code === "ECONNREFUSED" ||
    err?.message?.includes("Connection error")
  );
}

/**
 * Codex 风格对话 — 支持自动 fallback
 *
 * @param {string} prompt       - 用户输入
 * @param {object} opts         - 配置
 * @param {boolean} opts.useProxy - 是否通过 CCSwitch 代理 (默认 false, 直连)
 * @param {boolean} opts.stream   - 是否流式输出
 * @param {boolean} opts.thinking - 是否开启深度思考
 * @param {boolean} opts.autoFallback - 代理不通时自动降级直连 (默认 true)
 */
async function ask(prompt, opts = {}) {
  const { useProxy = false, stream = true, thinking = true, autoFallback = true } = opts;

  const messages = [{ role: "user", content: prompt }];

  const doCall = (client) =>
    stream
      ? chatStream(client, { messages, thinking }).then((r) => r.content)
      : chat(client, { messages, thinking }).then((m) => m.content);

  if (useProxy) {
    try {
      return await doCall(createCCSwitchClient());
    } catch (err) {
      if (autoFallback && isConnectionRefused(err)) {
        console.error("[CCSwitch 代理不可用, 自动降级为直连 DeepSeek]");
        return await doCall(createDeepSeekClient());
      }
      throw err;
    }
  }
  return doCall(createDeepSeekClient());
}

// ============================================================
// 入口: 命令行直接运行时自测
// ============================================================
async function main() {
  console.log("========================================");
  console.log("  CCSwitch + DeepSeek-V4-Pro 接入验证");
  console.log("========================================\n");

  // 检查 API Key
  if (!DEEPSEEK_API_KEY) {
    console.error("❌ 未找到 api_key，请检查 .env 文件");
    process.exit(1);
  }
  console.log(`✅ API Key 已加载: ${DEEPSEEK_API_KEY.slice(0, 8)}...\n`);

  // ---------- 测试1: 直连 DeepSeek (非流式) ----------
  console.log("--- 测试1: 直连 DeepSeek (非流式) ---");
  const directClient = createDeepSeekClient();
  try {
    const msg = await chat(directClient, {
      messages: [{ role: "user", content: "用一句话介绍你自己" }],
      thinking: false,
    });
    console.log(`✅ 直连成功: ${msg.content}\n`);
  } catch (err) {
    console.error(`❌ 直连失败: ${err.message}\n`);
  }

  // ---------- 测试2: 直连 DeepSeek (流式 + 思考) ----------
  console.log("--- 测试2: 直连 DeepSeek (流式 + 思考) ---");
  try {
    const result = await chatStream(directClient, {
      messages: [{ role: "user", content: "你是什么模型" }],
      thinking: true,
      onReasoning: (text) => process.stdout.write(`\n[思考] ${text}`),
    });
    console.log(`\n✅ 流式测试完成`);
    console.log(`   思考用时长度: ${result.reasoning.length} 字符`);
    console.log(`   最终回答: ${result.content}\n`);
  } catch (err) {
    console.error(`❌ 流式失败: ${err.message}\n`);
  }

  // ---------- 测试3: 通过 CCSwitch 代理 (自动 fallback 到直连) ----------
  console.log("--- 测试3: CCSwitch 代理模式 (autoFallback) ---");
  try {
    const answer = await ask("Hi! 回应 OK 即可", {
      useProxy: true,
      stream: false,
      thinking: false,
      autoFallback: true,
    });
    console.log(`✅ 调用成功: ${answer}\n`);
  } catch (err) {
    console.error(`❌ 调用失败: ${err.message}\n`);
  }

  console.log("========================================");
  console.log("  验证完成");
  console.log("========================================");
}

// 直接运行
const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, "/"));
if (isMain) {
  main();
}

export {
  createDeepSeekClient,
  createCCSwitchClient,
  chat,
  chatStream,
  ask,
  DEEPSEEK_BASE_URL,
  CCSWITCH_PROXY_URL,
};
