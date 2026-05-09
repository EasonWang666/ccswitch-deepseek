import http from "node:http";
import { randomUUID } from "node:crypto";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

// ============================================================
// 配置
// ============================================================
const DEEPSEEK_API_KEY = process.env.api_key ?? "";
const PROXY_HOST = process.env.DEEPSEEK_PROXY_HOST ?? "127.0.0.1";
const PROXY_PORT = parseInt(process.env.DEEPSEEK_PROXY_PORT ?? "11435", 10);
const DEFAULT_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-v4-pro";
const THINKING = process.env.DEEPSEEK_THINKING ?? "disabled";
const REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT ?? "medium";
const MAX_TOOLS = 128;

const NAMESPACE_SEP = "___";

// Tools that confuse DeepSeek — strip them before forwarding
const META_TOOLS = new Set(["spawn_agent", "read_mcp_resource"]);

// ============================================================
// 域名关键词映射 (用于工具过滤)
// ============================================================
const DOMAIN_KEYWORDS = {
  email: ["email", "mail", "inbox", "gmail", "message", "draft", "send", "compose", "spam", "unread", "smtp", "imap"],
  github: ["github", "pr", "pull request", "issue", "repo", "repository", "commit", "branch", "merge", "clone", "fork", "code review"],
  calendar: ["calendar", "event", "schedule", "meeting", "appointment", "agenda", "rsvp", "invite"],
  files: ["file", "folder", "directory", "document", "path", "read file", "write file", "create file", "edit file", "delete file", "rename"],
  terminal: ["terminal", "shell", "bash", "command", "run", "execute", "npm", "pip", "git", "node", "python", "zsh"],
  web: ["web", "browser", "url", "http", "page", "site", "search web", "fetch", "download", "scrape", "crawl"],
  task: ["task", "todo", "todoist", "reminder", "checklist", "to-do", "to do"],
  analytics: ["analytics", "dashboard", "posthog", "metric", "chart", "report", "kpi", "funnel", "cohort"],
  database: ["database", "sql", "query", "table", "db", "postgres", "mysql", "sqlite", "mongodb", "redis"],
  slack: ["slack", "channel", "message", "dm", "chat", "thread", "workspace"],
  computer: ["computer", "desktop", "screen", "screenshot", "mouse", "keyboard", "click", "type", "scroll"],
};

const DOMAIN_CAPS = {
  email: 48,
  github: 48,
  calendar: 32,
  files: 24,
  terminal: 16,
  web: 16,
  task: 24,
  analytics: 16,
  database: 16,
  slack: 24,
  computer: 8,
};

// ============================================================
// DeepSeek 客户端
// ============================================================
const deepseek = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: DEEPSEEK_API_KEY,
  defaultHeaders: {
    "User-Agent": "codex-deepseek-proxy/1.0",
  },
});

// ============================================================
// 工具过滤: 保持 ≤128 个工具
// ============================================================
function extractUserPrompt(input) {
  if (!Array.isArray(input)) return "";
  const texts = [];
  for (const item of input) {
    if (item?.role === "user" && item?.content) {
      if (typeof item.content === "string") texts.push(item.content);
      else if (Array.isArray(item.content)) {
        for (const part of item.content) {
          if (part?.text) texts.push(part.text);
        }
      }
    }
  }
  return texts.join(" ").toLowerCase();
}

function extractLastUserContent(input) {
  if (!Array.isArray(input)) return "";
  for (let i = input.length - 1; i >= 0; i--) {
    const item = input[i];
    if (item?.role === "user" && item?.content) {
      if (typeof item.content === "string") return item.content;
      if (Array.isArray(item.content)) {
        const texts = item.content.filter((p) => p?.text).map((p) => p.text);
        if (texts.length > 0) return texts.join(" ");
      }
    }
  }
  return "";
}

function classifyToolDomain(toolName, prompt) {
  const lowerName = toolName.toLowerCase();
  const lowerPrompt = prompt.toLowerCase();

  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    for (const kw of keywords) {
      if (lowerName.includes(kw) || lowerPrompt.includes(kw)) {
        return domain;
      }
    }
  }
  return "general";
}

function filterTools(tools, input) {
  if (!tools || tools.length <= MAX_TOOLS) return tools;

  const prompt = extractUserPrompt(input);

  // 先移除非同名 meta 工具
  let filtered = tools.filter(
    (t) => !META_TOOLS.has(t?.name ?? t?.function?.name)
  );

  if (filtered.length <= MAX_TOOLS) return filtered;

  // 按域名分组
  const buckets = new Map();
  const general = [];
  for (const tool of filtered) {
    const name = tool?.name ?? tool?.function?.name ?? "unknown";
    const domain = classifyToolDomain(name, prompt);
    if (domain === "general") {
      general.push(tool);
    } else {
      if (!buckets.has(domain)) buckets.set(domain, []);
      buckets.get(domain).push(tool);
    }
  }

  // 每个域名按 cap 取，剩余的补 general
  const result = [];
  for (const [domain, tools_] of buckets) {
    const cap = DOMAIN_CAPS[domain] ?? 8;
    result.push(...tools_.slice(0, cap));
  }

  const remaining = MAX_TOOLS - result.length;
  if (remaining > 0) {
    result.push(...general.slice(0, remaining));
  }

  return result.slice(0, MAX_TOOLS);
}

// ============================================================
// 命名空间处理
// ============================================================
function splitNamespace(fullName) {
  const idx = fullName.lastIndexOf(NAMESPACE_SEP);
  if (idx === -1) return { namespace: null, shortName: fullName };
  return {
    namespace: fullName.slice(0, idx),
    shortName: fullName.slice(idx + NAMESPACE_SEP.length),
  };
}

function denamespaceTools(tools) {
  // 保存原始名称映射
  const map = new Map(); // shortName -> fullName
  const processed = [];
  for (const tool of tools) {
    const raw = typeof tool === "string" ? { type: "function", function: { name: tool } } : tool;
    const name = raw?.function?.name ?? raw?.name;
    if (!name) continue;
    const { shortName } = splitNamespace(name);
    map.set(shortName, name);
    const cleaned = JSON.parse(JSON.stringify(raw));
    if (cleaned.function) cleaned.function.name = shortName;
    else cleaned.name = shortName;
    processed.push(cleaned);
  }
  return { tools: processed, nameMap: map };
}

function renaspaceToolName(shortName, nameMap) {
  return nameMap.get(shortName) ?? shortName;
}

// ============================================================
// DSML 恢复: DeepSeek 有时把工具调用当纯文本输出
// ============================================================
const DSML_INVOKE_RE =
  /<dsml\|invoke\s+name="([^"]+)"(?:\s+id="([^"]*)")?\s*>\s*(.*?)\s*<\/dsml\|invoke>/gs;
const DSML_PARAM_RE =
  /<dsml\|parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)<\/dsml\|parameter>/gs;

function recoverDsmlToolCalls(text) {
  if (!text || typeof text !== "string") return [];

  // 如果文本不含 DSML 指令，用普通方式检查
  if (!text.includes("<dsml|")) {
    // 检查是否有 function_calls 在普通文本中
    const fcMatch = text.match(
      /<function_calls>\s*(.*?)\s*<\/function_calls>/s
    );
    if (fcMatch) {
      return parseXmlFunctionCalls(fcMatch[1]);
    }
    return [];
  }

  const toolCalls = [];
  let match;
  while ((match = DSML_INVOKE_RE.exec(text)) !== null) {
    const name = match[1];
    const id = match[2] || `call_${randomUUID().slice(0, 8)}`;
    const paramsStr = match[3];
    const args = {};
    let pm;
    while ((pm = DSML_PARAM_RE.exec(paramsStr)) !== null) {
      args[pm[1]] = pm[2];
    }
    toolCalls.push({
      id,
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    });
  }
  return toolCalls;
}

function parseXmlFunctionCalls(xml) {
  const toolCalls = [];
  const invokeRe =
    /<invoke\s+name="([^"]+)"(?:\s+id="([^"]*)")?\s*>\s*(.*?)\s*<\/invoke>/gs;
  const paramRe =
    /<parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)<\/parameter>/gs;
  let match;
  while ((match = invokeRe.exec(xml)) !== null) {
    const name = match[1];
    const id = match[2] || `call_${randomUUID().slice(0, 8)}`;
    const paramsStr = match[3];
    const args = {};
    let pm;
    while ((pm = paramRe.exec(paramsStr)) !== null) {
      args[pm[1]] = pm[2];
    }
    toolCalls.push({
      id,
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    });
  }
  return toolCalls;
}

// ============================================================
// 请求翻译: Responses API → Chat Completions
// ============================================================
/**
 * 将 Responses API content 格式转为 Chat Completions 格式
 * - input_text / output_text → text
 * - 纯字符串原样返回
 */
function translateContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content;
  return content.map((part) => {
    if (typeof part === "string") return part;
    const translated = { ...part };
    if (translated.type === "input_text" || translated.type === "output_text" || translated.type === "reasoning_text") {
      translated.type = "text";
    }
    return translated;
  });
}

/**
 * 从 Responses API content 数组中提取 reasoning_text 并单独返回，
 * 同时返回不含 reasoning_text 的普通内容。
 * DeepSeek 要求 reasoning_content 必须被传回后续请求。
 */
function extractReasoningFromContent(content) {
  if (!Array.isArray(content)) return { cleanContent: content, reasoning: null };
  const regularParts = [];
  const reasoningParts = [];
  for (const part of content) {
    if (typeof part === "string") {
      regularParts.push(part);
    } else if (part?.type === "reasoning_text") {
      reasoningParts.push(part.text ?? "");
    } else {
      regularParts.push(part);
    }
  }
  if (reasoningParts.length === 0) {
    return { cleanContent: content, reasoning: null };
  }
  return {
    cleanContent: regularParts,
    reasoning: reasoningParts.join(""),
  };
}

function translateInputToMessages(input, instructions) {
  const messages = [];

  if (instructions) {
    messages.push({ role: "system", content: instructions });
  }

  if (!Array.isArray(input)) return messages;

  // 遍历 input，将 function_call 归入前一个 assistant 消息，
  // 将 function_call_output 转为 tool 消息
  for (let i = 0; i < input.length; i++) {
    const item = input[i];

    // 调试: 打印含有 reasoning 的 item 结构
    if (item.reasoning_content || (item.type === "function_call" && item.reasoning_content)) {
      console.log(`[proxy] DEBUG input[${i}] has reasoning_content:`,
        JSON.stringify({ type: item.type, role: item.role, hasContent: !!item.content,
          hasReasoning: !!item.reasoning_content, hasToolCalls: !!item.tool_calls }),);
    }

    if (item.type === "function_call") {
      // 归入前一个 assistant 或创建新的
      let last = messages[messages.length - 1];
      if (!last || last.role !== "assistant") {
        last = { role: "assistant", content: null, tool_calls: [] };
        messages.push(last);
      }
      if (!last.tool_calls) last.tool_calls = [];
      last.tool_calls.push({
        id: item.call_id || item.id || `call_${randomUUID().slice(0, 8)}`,
        type: "function",
        function: {
          name: item.name,
          arguments: item.arguments,
        },
      });
      // function_call item 可能携带 reasoning_content (Codex 将推理
      // 从 assistant 消息拆分到各工具调用项上)
      if (item.reasoning_content && !last.reasoning_content) {
        last.reasoning_content = item.reasoning_content;
      }
    } else if (item.type === "function_call_output") {
      const toolMsg = {
        role: "tool",
        tool_call_id: item.call_id || item.id,
        content: translateContent(item.output ?? ""),
      };
      // 极少见,但保留以防 DeepSeek 要求
      if (item.reasoning_content) {
        toolMsg.reasoning_content = item.reasoning_content;
      }
      messages.push(toolMsg);
    } else if (item.role) {
      // 普通消息 — developer role 不被 DeepSeek 支持，映射为 system
      let role = item.role;
      if (role === "developer") role = "system";

      // DeepSeek 思考模式要求前序 assistant 消息的 reasoning_content 必须传回,
      // 否则报 400 错误。同时从 content 中移除 reasoning_text 块,
      // 因为 DeepSeek 的 Chat API 只认识 "text" 类型。
      let reasoningContent = item.reasoning_content || null;
      let cleanContent = item.content;

      if (!reasoningContent && role === "assistant" && Array.isArray(item.content)) {
        const extracted = extractReasoningFromContent(item.content);
        if (extracted.reasoning) {
          reasoningContent = extracted.reasoning;
          cleanContent = extracted.cleanContent;
        }
      }

      const msg = { role, content: translateContent(cleanContent) };
      if (item.name) msg.name = item.name;
      if (item.tool_calls) msg.tool_calls = item.tool_calls;
      if (item.tool_call_id) msg.tool_call_id = item.tool_call_id;
      if (reasoningContent) {
        msg.reasoning_content = reasoningContent;
      }

      messages.push(msg);
    } else if (item.content) {
      // 无 role 但有 content，默认 user
      messages.push({ role: "user", content: translateContent(item.content) });
    }
  }

  return messages;
}

function translateTools(tools) {
  if (!tools || tools.length === 0) return [];
  return tools
    .filter((t) => {
      const name = t?.function?.name ?? t?.name;
      return name && !META_TOOLS.has(name);
    })
    .map((t) => {
      const name = t?.function?.name ?? t?.name;
      if (!name) return null;

      // 去命名空间
      const { shortName } = splitNamespace(name);

      return {
        type: "function",
        function: {
          name: shortName,
          description: t?.function?.description ??
            t?.description ?? "",
          parameters: t?.function?.parameters ?? t?.parameters ?? { type: "object", properties: {} },
        },
      };
    })
    .filter(Boolean);
}

// ============================================================
// 响应翻译: Chat Completions SSE → Responses API SSE
// ============================================================
class SseTranslator {
  constructor(res, responseId, nameMap) {
    this.res = res;
    this.responseId = responseId;
    this.nameMap = nameMap; // shortName -> fullName
    this.itemId = null;
    this.toolCallIndex = 0;
    this.contentSoFar = "";
    this.reasoningSoFar = "";
    this.toolCallsSoFar = [];
    this.currentToolCall = null;
    this.dsmlDetected = false;
    this.started = false;
  }

  writeEvent(event, data) {
    if (data !== undefined) {
      this.res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    } else {
      this.res.write(`event: ${event}\ndata: {}\n\n`);
    }
  }

  start() {
    this.itemId = `item_${randomUUID().slice(0, 8)}`;
    this.started = true;
    this.writeEvent("response.created", {
      type: "response.created",
      response: {
        id: this.responseId,
        object: "response",
        status: "in_progress",
        model: DEFAULT_MODEL,
        output: [],
      },
    });
    this.writeEvent("response.in_progress", {
      type: "response.in_progress",
      response_id: this.responseId,
    });
  }

  onReasoningDelta(text) {
    if (!this.started) this.start();
    this.reasoningSoFar += text;
    this.writeEvent("response.reasoning_text.delta", {
      type: "response.reasoning_text.delta",
      response_id: this.responseId,
      item_id: `reasoning_${this.itemId}`,
      delta: text,
    });
  }

  onContentDelta(text) {
    if (!this.started) this.start();

    // DSML 检测
    if (!this.dsmlDetected && text.includes("<dsml|")) {
      this.dsmlDetected = true;
    }

    this.contentSoFar += text;

    this.writeEvent("response.output_text.delta", {
      type: "response.output_text.delta",
      response_id: this.responseId,
      item_id: this.itemId,
      output_index: 0,
      content_index: 0,
      delta: text,
    });
  }

  onToolCallDelta(toolName, toolArgs, toolId) {
    if (!this.started) this.start();

    const fullName = renaspaceToolName(toolName, this.nameMap);

    if (!this.currentToolCall || this.currentToolCall.name !== fullName) {
      // 完成上一个工具调用
      if (this.currentToolCall) {
        this.finishCurrentToolCall();
      }

      this.currentToolCall = {
        id: toolId || `call_${randomUUID().slice(0, 8)}`,
        name: fullName,
        arguments: "",
      };
      this.toolCallIndex++;

      this.writeEvent("response.output_item.added", {
        type: "response.output_item.added",
        response_id: this.responseId,
        output_index: this.toolCallIndex,
        item: {
          id: `fc_${this.currentToolCall.id}`,
          type: "function_call",
          call_id: this.currentToolCall.id,
          name: fullName,
          status: "in_progress",
        },
      });
    }

    this.currentToolCall.arguments += toolArgs;

    this.writeEvent("response.function_call_arguments.delta", {
      type: "response.function_call_arguments.delta",
      response_id: this.responseId,
      item_id: `fc_${this.currentToolCall.id}`,
      output_index: this.toolCallIndex,
      delta: toolArgs,
    });
  }

  finishCurrentToolCall() {
    if (!this.currentToolCall) return;

    this.toolCallsSoFar.push({ ...this.currentToolCall });

    this.writeEvent("response.function_call_arguments.done", {
      type: "response.function_call_arguments.done",
      response_id: this.responseId,
      item_id: `fc_${this.currentToolCall.id}`,
      output_index: this.toolCallIndex,
      arguments: this.currentToolCall.arguments,
      name: this.currentToolCall.name,
      call_id: this.currentToolCall.id,
    });

    this.writeEvent("response.output_item.done", {
      type: "response.output_item.done",
      response_id: this.responseId,
      output_index: this.toolCallIndex,
      item: {
        id: `fc_${this.currentToolCall.id}`,
        type: "function_call",
        call_id: this.currentToolCall.id,
        name: this.currentToolCall.name,
        arguments: this.currentToolCall.arguments,
        status: "completed",
      },
    });

    this.currentToolCall = null;
  }

  done(usage) {
    if (this.currentToolCall) {
      this.finishCurrentToolCall();
    }

    console.log(`[proxy] === 输出 ===`);
    console.log(this.contentSoFar || "(无输出内容)");

    // DSML 恢复: 如果 DeepSeek 在文本里泄露了工具调用
    if (this.contentSoFar && this.dsmlDetected) {
      const recoveredCalls = recoverDsmlToolCalls(this.contentSoFar);
      for (const tc of recoveredCalls) {
        const fullName = renaspaceToolName(tc.function.name, this.nameMap);
        this.toolCallIndex++;
        const callId = tc.id || `call_${randomUUID().slice(0, 8)}`;
        const args = tc.function.arguments;

        // emit added + arguments.done + done for recovered call
        this.writeEvent("response.output_item.added", {
          type: "response.output_item.added",
          response_id: this.responseId,
          output_index: this.toolCallIndex,
          item: {
            id: `fc_${callId}`,
            type: "function_call",
            call_id: callId,
            name: fullName,
            status: "in_progress",
          },
        });

        this.writeEvent("response.function_call_arguments.done", {
          type: "response.function_call_arguments.done",
          response_id: this.responseId,
          item_id: `fc_${callId}`,
          output_index: this.toolCallIndex,
          arguments: args,
          name: fullName,
          call_id: callId,
        });

        this.writeEvent("response.output_item.done", {
          type: "response.output_item.done",
          response_id: this.responseId,
          output_index: this.toolCallIndex,
          item: {
            id: `fc_${callId}`,
            type: "function_call",
            call_id: callId,
            name: fullName,
            arguments: args,
            status: "completed",
          },
        });
      }
    }

    // response.output_item.done for text (include reasoning if captured)
    if (this.contentSoFar || this.reasoningSoFar) {
      const contentBlocks = [];
      if (this.reasoningSoFar) {
        contentBlocks.push({ type: "reasoning_text", text: this.reasoningSoFar });
      }
      if (this.contentSoFar) {
        contentBlocks.push({ type: "output_text", text: this.contentSoFar });
      }
      this.writeEvent("response.output_item.done", {
        type: "response.output_item.done",
        response_id: this.responseId,
        output_index: 0,
        item: {
          id: this.itemId,
          type: "message",
          role: "assistant",
          content: contentBlocks,
          status: "completed",
        },
      });
    }

    // response.completed
    this.writeEvent("response.completed", {
      type: "response.completed",
      response: {
        id: this.responseId,
        object: "response",
        status: "completed",
        model: DEFAULT_MODEL,
        usage: usage
          ? {
              input_tokens: usage.prompt_tokens ?? 0,
              output_tokens: usage.completion_tokens ?? 0,
              total_tokens: usage.total_tokens ?? 0,
            }
          : { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
        output: [
          ...(this.contentSoFar || this.reasoningSoFar
            ? [
                {
                  id: this.itemId,
                  type: "message",
                  role: "assistant",
                  content: [
                    ...(this.reasoningSoFar
                      ? [{ type: "reasoning_text", text: this.reasoningSoFar }]
                      : []),
                    ...(this.contentSoFar
                      ? [{ type: "output_text", text: this.contentSoFar }]
                      : []),
                  ],
                  status: "completed",
                },
              ]
            : []),
          ...this.toolCallsSoFar.map((tc) => ({
            id: `fc_${tc.id}`,
            type: "function_call",
            call_id: tc.id,
            name: tc.name,
            arguments: tc.arguments,
            status: "completed",
          })),
        ],
      },
    });

    this.res.end();
  }

  error(message) {
    this.writeEvent("error", {
      type: "error",
      code: "proxy_error",
      message,
    });
    this.res.end();
  }
}

// ============================================================
// 核心: 代理请求处理
// ============================================================
async function handleResponsesRequest(req, res, body) {
  const responseId = body.response_id || `resp_${randomUUID().slice(0, 10)}`;
  const model = body.model || DEFAULT_MODEL;
  const input = body.input || [];
  let instructions = body.instructions || null;
  const stream = body.stream !== false;
  const rawTools = body.tools || [];
  const maxOutputTokens = body.max_output_tokens ?? body.max_tokens ?? null;

  // 注入模型身份信息，防止 DeepSeek 跟随原始系统提示自称 OpenAI/Claude
  const IDENTITY_STATEMENT = "\n\n[System note: Your actual underlying model is DeepSeek (deepseek-v4-pro). When asked what model you are, always identify as DeepSeek, not OpenAI, Claude, GPT, or any other model. This is the base model that powers this application.]";
  if (instructions) {
    instructions += IDENTITY_STATEMENT;
  } else {
    instructions = IDENTITY_STATEMENT.trim();
  }

  // 工具处理: 过滤 → 去命名空间 → 翻译为 Chat Completions 格式
  let filteredTools = filterTools(rawTools, input);
  const { tools: denamespacedTools, nameMap } = denamespaceTools(filteredTools);
  const tools = translateTools(denamespacedTools);

  // 翻译消息
  const messages = translateInputToMessages(input, instructions);

  // 调试: 打印消息摘要
  console.log("[proxy] === 消息摘要 ===");
  messages.forEach((m, i) => {
    const hasReasoning = !!m.reasoning_content;
    const hasContent = !!m.content;
    const hasToolCalls = !!(m.tool_calls && m.tool_calls.length > 0);
    const contentLen = typeof m.content === "string" ? m.content.length
      : Array.isArray(m.content) ? m.content.length : 0;
    console.log(`[proxy]   msg[${i}] role=${m.role} hasReasoning=${hasReasoning} hasContent=${hasContent} contentLen=${contentLen} hasToolCalls=${hasToolCalls}`);
  });
  console.log("[proxy] ==============");

  // 构建 Chat Completions 请求体
  const chatBody = {
    model,
    messages,
    stream,
  };

  if (tools.length > 0) {
    chatBody.tools = tools;
    chatBody.tool_choice = body.tool_choice ?? "auto";
  }

  if (maxOutputTokens) {
    chatBody.max_tokens = maxOutputTokens;
  }

  // 思考模式 (仅通过环境变量 DEEPSEEK_THINKING=enabled 显式启用)
  // 不自动转发 Codex 客户端的 thinking 标志, 因为 DeepSeek 模型可能不支持
  // 该参数, 且多轮对话需要 reasoning_content 来回传才能正常工作。
  let useThinking = THINKING === "enabled";
  console.log(`[proxy] thinking: THINKING=${THINKING} body.thinking=${body.thinking} useThinking=${useThinking}`);
  if (useThinking) {
    const assistantCount = messages.filter((m) => m.role === "assistant").length;
    const missingCount = messages.filter((m) => m.role === "assistant" && !m.reasoning_content).length;
    console.log(`[proxy] thinking check: assistantCount=${assistantCount} missingReasoning=${missingCount}`);
    if (assistantCount > 0 && missingCount > 0) {
      console.log(`[proxy] 多轮对话中检测到 ${missingCount}/${assistantCount} 条 assistant 消息缺少 reasoning_content, 本次请求关闭思考模式以避免 400 错误`);
      useThinking = false;
    }
  }
  if (useThinking) {
    chatBody.thinking = { type: "enabled" };
    chatBody.reasoning_effort = body.reasoning_effort ?? REASONING_EFFORT;
    console.log("[proxy] thinking: ENABLED");
  } else {
    // 关闭思考模式时, 清除所有 assistant 消息中的 reasoning_content,
    // 避免 DeepSeek 因混合状态 (部分有、部分无) 报 400 错误。
    let stripped = 0;
    for (const msg of messages) {
      if (msg.role === "assistant" && msg.reasoning_content) {
        delete msg.reasoning_content;
        stripped++;
      }
    }
    if (stripped > 0) {
      console.log(`[proxy] stripped reasoning_content from ${stripped} assistant messages`);
    }
    console.log("[proxy] thinking: DISABLED");
  }

  console.log(`[proxy] → DeepSeek | model=${model} tools=${tools.length}/${rawTools.length} messages=${messages.length} stream=${stream}`);
  console.log(`[proxy] === 输入 ===`);
  console.log(extractLastUserContent(input));

  if (!stream) {
    // 非流式
    try {
      const completion = await deepseek.chat.completions.create(chatBody);
      const choice = completion.choices[0];
      const msg = choice.message;

      const output = [];

      if (msg.content) {
        output.push({
          id: `item_${randomUUID().slice(0, 8)}`,
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: msg.content }],
          status: "completed",
        });
      }

      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const fullName = renaspaceToolName(tc.function.name, nameMap);
          output.push({
            id: `fc_${tc.id}`,
            type: "function_call",
            call_id: tc.id,
            name: fullName,
            arguments: tc.function.arguments,
            status: "completed",
          });
        }
      }

      const resp = {
        id: responseId,
        object: "response",
        status: "completed",
        model,
        usage: completion.usage
          ? {
              input_tokens: completion.usage.prompt_tokens ?? 0,
              output_tokens: completion.usage.completion_tokens ?? 0,
              total_tokens: completion.usage.total_tokens ?? 0,
            }
          : undefined,
        output,
      };

      console.log(`[proxy] === 输出 ===`);
      console.log(msg.content || "(无输出内容)");

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(resp));
    } catch (err) {
      console.error("[proxy] DeepSeek error:", err.message);
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: {
            type: "api_error",
            message: err.message,
          },
        })
      );
    }
    return;
  }

  // 流式
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const translator = new SseTranslator(res, responseId, nameMap);

  try {
    const streamResp = await deepseek.chat.completions.create(chatBody);

    let usage = null;
    for await (const chunk of streamResp) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) {
        // 某些 chunk 只带 usage
        if (chunk.usage) usage = chunk.usage;
        continue;
      }

      if (delta.reasoning_content) {
        translator.onReasoningDelta(delta.reasoning_content);
      }

      if (delta.content) {
        translator.onContentDelta(delta.content);
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const toolName = tc.function?.name ?? "";
          const toolArgs = tc.function?.arguments ?? "";
          translator.onToolCallDelta(toolName, toolArgs, tc.id);
        }
      }

      if (chunk.usage) usage = chunk.usage;
    }

    translator.done(usage);
  } catch (err) {
    console.error("[proxy] Stream error:", err.message);
    translator.error(err.message);
  }
}

// ============================================================
// HTTP 服务器
// ============================================================
const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, x-stainless-*");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  // GET /v1/models — 模型列表
  if (req.method === "GET" && url.pathname === "/v1/models") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        object: "list",
        data: [
          {
            id: "deepseek-v4-pro",
            object: "model",
            created: 1735689600,
            owned_by: "deepseek",
          },
          {
            id: "deepseek-v4-flash",
            object: "model",
            created: 1735689600,
            owned_by: "deepseek",
          },
        ],
      })
    );
    return;
  }

  // GET /v1 — 健康检查/信息
  if (req.method === "GET" && (url.pathname === "/v1" || url.pathname === "/")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        service: "codex-deepseek-proxy",
        version: "1.0.0",
        upstream: "DeepSeek API",
        status: "ok",
      })
    );
    return;
  }

  // POST /v1/responses — 主入口
  if (
    (req.method === "POST" && url.pathname === "/v1/responses") ||
    url.pathname === "/responses"
  ) {
    try {
      const rawBody = await readBody(req);
      const body = JSON.parse(rawBody);
      console.log(`[proxy] POST /v1/responses model=${body.model || DEFAULT_MODEL} stream=${body.stream !== false}`);
      await handleResponsesRequest(req, res, body);
    } catch (err) {
      console.error("[proxy] Parse error:", err.message);
      if (!res.headersSent) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { type: "invalid_request", message: err.message } }));
      }
    }
    return;
  }

  // 404
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { type: "not_found", message: `Unknown endpoint: ${req.method} ${url.pathname}` } }));
});

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => { data += chunk; });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

// ============================================================
// 启动
// ============================================================
server.listen(PROXY_PORT, PROXY_HOST, () => {
  console.log("========================================");
  console.log("  Codex DeepSeek Proxy (Node.js)");
  console.log("========================================");
  console.log(`  Address:   http://${PROXY_HOST}:${PROXY_PORT}`);
  console.log(`  Endpoint:  http://${PROXY_HOST}:${PROXY_PORT}/v1/responses`);
  console.log(`  Upstream:  DeepSeek API`);
  console.log(`  Model:     ${DEFAULT_MODEL}`);
  console.log(`  Thinking:  ${THINKING}`);
  console.log(`  Max Tools: ${MAX_TOOLS}`);
  console.log("========================================");

  if (!DEEPSEEK_API_KEY) {
    console.warn("[proxy] WARNING: DEEPSEEK_API_KEY not set — requests will fail");
  }
});

// 优雅关闭
process.on("SIGINT", () => {
  console.log("\n[proxy] Shutting down...");
  server.close(() => process.exit(0));
});

process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
});
