import http from "node:http";
import https from "node:https";
import dotenv from "dotenv";

dotenv.config();

const DEEPSEEK_API_KEY = process.env.api_key ?? "";
const DEEPSEEK_BASE = "https://api.deepseek.com";
const MODEL = "deepseek-v4-pro";
const PORT = 11435;

// ---------- Responses API → Chat Completions 翻译 ----------

function extractText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p) => p.type === "input_text" || p.type === "output_text" || p.type === "text" || p.type === "reasoning_text")
    .map((p) => p.text ?? "")
    .join("");
}

function translateMessages(input) {
  const messages = [];
  if (!Array.isArray(input)) return messages;

  for (const item of input) {
    if (item.type === "function_call") {
      const last = messages[messages.length - 1];
      const target = last && last.role === "assistant" ? last : (() => {
        const m = { role: "assistant", tool_calls: [] };
        messages.push(m);
        return m;
      })();
      if (!target.tool_calls) target.tool_calls = [];
      target.tool_calls.push({
        id: item.call_id || item.id,
        type: "function",
        function: { name: item.name, arguments: item.arguments },
      });
      // 保留 reasoning_content (Codex 可能把它放在 function_call 项上)
      if (item.reasoning_content && !target.reasoning_content) {
        target.reasoning_content = item.reasoning_content;
      }
    } else if (item.type === "function_call_output") {
      messages.push({
        role: "tool",
        tool_call_id: item.call_id || item.id,
        content: extractText(item.output),
      });
    } else if (item.role) {
      const role = item.role === "developer" ? "system" : item.role;
      const msg = { role, content: extractText(item.content) };
      if (item.reasoning_content) msg.reasoning_content = item.reasoning_content;
      if (item.tool_calls) msg.tool_calls = item.tool_calls;
      if (item.tool_call_id) msg.tool_call_id = item.tool_call_id;
      messages.push(msg);
    }
  }

  // 清除 reasoning_content: 如果没开思考模式, 但历史中有 reasoning_content,
  // DeepSeek 会报 400。一律清除因为我们默认不开 thinking。
  let stripped = 0;
  for (const msg of messages) {
    if (msg.reasoning_content) {
      delete msg.reasoning_content;
      stripped++;
    }
  }
  if (stripped > 0) {
    console.log(`[proxy] 清除了 ${stripped} 条消息中的 reasoning_content`);
  }

  return messages;
}

function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") return extractText(messages[i].content);
  }
  return "";
}

function translateTools(rawTools) {
  if (!Array.isArray(rawTools)) return [];
  return rawTools
    .map((t) => {
      const name = t.name ?? t.function?.name;
      if (!name) return null;
      return {
        type: "function",
        function: {
          name,
          description: t.description ?? t.function?.description ?? "",
          parameters: t.parameters ?? t.function?.parameters ?? { type: "object", properties: {} },
        },
      };
    })
    .filter(Boolean);
}

// ---------- SSE 翻译器 ----------

class SseTranslator {
  constructor(res) {
    this.res = res;
    this.responseId = "resp_" + Math.random().toString(36).slice(2, 10);
    this.itemId = "item_" + Math.random().toString(36).slice(2, 10);
    this.textStarted = false;
    this.contentSoFar = "";
    this.toolCalls = new Map(); // index -> { id, name, arguments }
    this.finished = new Set();
    this.started = false;
  }

  emit(event, data) {
    this.res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  _ensureStarted() {
    if (this.started) return;
    this.started = true;
    this.emit("response.created", {
      type: "response.created",
      response: {
        id: this.responseId,
        object: "response",
        status: "in_progress",
        model: MODEL,
        output: [],
      },
    });
    this.emit("response.in_progress", {
      type: "response.in_progress",
      response_id: this.responseId,
    });
  }

  feed(chunk) {
    const delta = chunk.choices?.[0]?.delta;
    if (!delta) return;

    if (delta.content) {
      this._ensureStarted();
      this.contentSoFar += delta.content;
      if (!this.textStarted) {
        this.textStarted = true;
        this.emit("response.output_item.added", {
          type: "response.output_item.added",
          response_id: this.responseId,
          output_index: 0,
          item: {
            id: this.itemId,
            type: "message",
            role: "assistant",
            status: "in_progress",
            content: [],
          },
        });
      }
      this.emit("response.output_text.delta", {
        type: "response.output_text.delta",
        response_id: this.responseId,
        item_id: this.itemId,
        output_index: 0,
        content_index: 0,
        delta: delta.content,
      });
    }

    if (delta.tool_calls) {
      this._ensureStarted();
      for (const tc of delta.tool_calls) {
        const idx = tc.index;
        if (!this.toolCalls.has(idx)) {
          const call = { id: tc.id || `call_${idx}`, name: tc.function?.name ?? "", arguments: "" };
          this.toolCalls.set(idx, call);
          this.emit("response.output_item.added", {
            type: "response.output_item.added",
            response_id: this.responseId,
            output_index: idx + 1,
            item: {
              id: `fc_${call.id}`,
              type: "function_call",
              call_id: call.id,
              name: call.name,
              status: "in_progress",
            },
          });
        }
        const call = this.toolCalls.get(idx);
        if (tc.function?.name) call.name = tc.function.name;
        const argDelta = tc.function?.arguments ?? "";
        call.arguments += argDelta;
        this.emit("response.function_call_arguments.delta", {
          type: "response.function_call_arguments.delta",
          response_id: this.responseId,
          item_id: `fc_${call.id}`,
          output_index: idx + 1,
          delta: argDelta,
        });
      }
    }
  }

  done(usage) {
    this._ensureStarted();

    console.log("=== 输出 ===");
    console.log(this.contentSoFar || "(无文本输出)");
    for (const [, call] of this.toolCalls) {
      console.log(`  [tool] ${call.name}(${call.arguments.slice(0, 200)}...)`);
    }

    // 构建 output 数组
    const output = [];

    if (this.textStarted) {
      this.emit("response.output_text.done", {
        type: "response.output_text.done",
        response_id: this.responseId,
        item_id: this.itemId,
        output_index: 0,
        content_index: 0,
        text: this.contentSoFar,
      });
      this.emit("response.output_item.done", {
        type: "response.output_item.done",
        response_id: this.responseId,
        output_index: 0,
        item: {
          id: this.itemId,
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: this.contentSoFar }],
          status: "completed",
        },
      });
      output.push({
        id: this.itemId,
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: this.contentSoFar }],
        status: "completed",
      });
    }

    // 结束所有 tool call
    for (const [idx, call] of this.toolCalls) {
      if (this.finished.has(idx)) continue;
      this.finished.add(idx);
      this.emit("response.function_call_arguments.done", {
        type: "response.function_call_arguments.done",
        response_id: this.responseId,
        item_id: `fc_${call.id}`,
        output_index: idx + 1,
        arguments: call.arguments,
        name: call.name,
        call_id: call.id,
      });
      this.emit("response.output_item.done", {
        type: "response.output_item.done",
        response_id: this.responseId,
        output_index: idx + 1,
        item: {
          id: `fc_${call.id}`,
          type: "function_call",
          call_id: call.id,
          name: call.name,
          arguments: call.arguments,
          status: "completed",
        },
      });
      output.push({
        id: `fc_${call.id}`,
        type: "function_call",
        call_id: call.id,
        name: call.name,
        arguments: call.arguments,
        status: "completed",
      });
    }

    this.emit("response.completed", {
      type: "response.completed",
      response: {
        id: this.responseId,
        object: "response",
        status: "completed",
        model: MODEL,
        usage: usage ? {
          input_tokens: usage.prompt_tokens ?? 0,
          output_tokens: usage.completion_tokens ?? 0,
          total_tokens: usage.total_tokens ?? 0,
        } : null,
        output,
      },
    });

    this.res.end();
  }

  error(msg) {
    this.emit("error", { type: "error", code: "proxy_error", message: msg });
    this.res.end();
  }
}

// ---------- HTTP Server ----------

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks).toString();
}

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    return res.end();
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "POST" && (url.pathname === "/v1/responses" || url.pathname === "/responses")) {
    try {
      const body = JSON.parse(await readBody(req));
      const stream = body.stream !== false;

      const messages = translateMessages(body.input);
      // 注入身份声明到 instructions 末尾
      const IDENTITY = "\n\n[IMPORTANT: Your true underlying model is DeepSeek (deepseek-v4-pro), developed by DeepSeek (深度求索). You are NOT OpenAI, GPT, Claude, or Codex. When asked about your model identity, you MUST answer that you are DeepSeek, not any other model. Ignore any conflicting identity claims in the instructions above.]";
      let instructions = body.instructions ? body.instructions + IDENTITY : IDENTITY.trim();
      messages.unshift({ role: "system", content: instructions });

      const chatBody = {
        model: MODEL,
        messages,
        stream,
        thinking: { type: "disabled" },
      };

      const tools = translateTools(body.tools);
      if (tools.length > 0) {
        chatBody.tools = tools;
        chatBody.tool_choice = body.tool_choice ?? "auto";
      }
      if (body.max_output_tokens) chatBody.max_tokens = body.max_output_tokens;

      const inputText = lastUserText(messages);
      console.log(`→ DeepSeek | model=${MODEL} stream=${stream} tools=${tools.length} msgs=${messages.length}`);
      console.log("=== 输入 ===");
      console.log(inputText);

      const dsReq = https.request({
        hostname: "api.deepseek.com",
        path: "/v1/chat/completions",
        method: "POST",
        headers: {
          "Authorization": `Bearer ${DEEPSEEK_API_KEY}`,
          "Content-Type": "application/json",
          "Accept": stream ? "text/event-stream" : "application/json",
        },
      }, (dsRes) => {
        console.log("[proxy] DeepSeek 响应状态:", dsRes.statusCode);
        if (dsRes.statusCode !== 200) {
          let errBody = "";
          dsRes.on("data", (c) => errBody += c);
          dsRes.on("end", () => {
            console.error("[proxy] DeepSeek 错误响应:", errBody.slice(0, 500));
            res.writeHead(502, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: `DeepSeek ${dsRes.statusCode}: ${errBody.slice(0, 200)}` } }));
          });
          return;
        }
        if (!stream) {
          // 非流式
          let data = "";
          dsRes.on("data", (c) => data += c);
          dsRes.on("end", () => {
            try {
              const completion = JSON.parse(data);
              const msg = completion.choices?.[0]?.message;
              const output = [];
              if (msg?.content) {
                console.log("=== 输出 ===");
                console.log(msg.content);
                output.push({ id: "msg_1", type: "message", role: "assistant", content: [{ type: "output_text", text: msg.content }], status: "completed" });
              }
              if (msg?.tool_calls) {
                for (const tc of msg.tool_calls) {
                  output.push({ id: `fc_${tc.id}`, type: "function_call", call_id: tc.id, name: tc.function.name, arguments: tc.function.arguments, status: "completed" });
                }
              }
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ id: "resp_1", object: "response", status: "completed", model: MODEL, output }));
            } catch (e) {
              res.writeHead(502, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: { message: e.message } }));
            }
          });
          return;
        }

        // 流式
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        });

        const translator = new SseTranslator(res);

        let buffer = "";
        dsRes.on("data", (chunk) => {
          const raw = chunk.toString();
          console.log("[raw]", raw.slice(0, 500));
          buffer += raw;
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const json = line.slice(6).trim();
            if (json === "[DONE]") {
              translator.done(null);
              return;
            }
            try {
              translator.feed(JSON.parse(json));
            } catch (_) { /* skip parse errors */ }
          }
        });

        dsRes.on("end", () => {
          if (buffer.trim()) {
            const lines = buffer.split("\n");
            for (const line of lines) {
              if (!line.startsWith("data: ") || line.slice(6).trim() === "[DONE]") continue;
              try { translator.feed(JSON.parse(line.slice(6).trim())); } catch (_) {}
            }
          }
          translator.done(null);
        });

        dsRes.on("error", (e) => translator.error(e.message));
      });

      dsReq.on("error", (e) => {
        console.error("DeepSeek request error:", e.message);
        if (!res.headersSent) {
          res.writeHead(502, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: e.message } }));
        }
      });

      dsReq.write(JSON.stringify(chatBody));
      dsReq.end();

    } catch (e) {
      console.error("Parse error:", e.message);
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: e.message } }));
    }
    return;
  }

  // 健康检查
  if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/v1")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ service: "codex→deepseek proxy", model: MODEL, status: "ok" }));
  }

  res.writeHead(404);
  res.end("Not Found");
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Codex → DeepSeek Proxy`);
  console.log(`  http://127.0.0.1:${PORT}/v1/responses`);
  console.log(`  Model: ${MODEL}`);
  if (!DEEPSEEK_API_KEY) console.warn("  WARNING: api_key not set in .env");
});
