const API_BASE = "http://localhost:8000";

const chatWindow = document.getElementById("chat-window");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const nodePipeline = document.getElementById("node-pipeline");
const llmStream = document.getElementById("llm-stream");

const NODE_LABELS = {
  router: "路由决策",
  analyze_query: "搜索意图分析",
  retrieve: "知识库检索",
  web_search: "智谱搜索",
  grade: "质量评估",
  generate: "生成回答",
  reflect: "自检",
};

let history = [];
let sending = false;
let visitedNodes = [];
let activeNode = null;
let currentLlmNode = null;

function renderMarkdown(text) {
  if (typeof marked !== "undefined" && marked.parse) {
    return marked.parse(text);
  }
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const roleLabel = role === "user" ? "你" : "助手";
  const rendered =
    role === "assistant" ? renderMarkdown(content) : escapeHtml(content);
  div.innerHTML = `<div class="role">${roleLabel}</div><div class="msg-body">${rendered}</div>`;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function renderPipeline() {
  if (!nodePipeline) return;
  let html = "";
  for (let i = 0; i < visitedNodes.length; i++) {
    const node = visitedNodes[i];
    const label = NODE_LABELS[node] || node;
    const isLast = i === visitedNodes.length - 1;
    const isActive = isLast && activeNode !== null;
    const cls = isActive ? "active" : "done";
    const icon = isActive ? "&#9654; " : "&#10003; ";
    html += `<span class="node-step ${cls}">${icon}${label}</span>`;
    if (i < visitedNodes.length - 1) {
      html += `<span class="node-arrow">&#8594;</span>`;
    }
  }
  nodePipeline.innerHTML = html;
}

function onNodeEvent(node) {
  visitedNodes.push(node);
  activeNode = node;
  renderPipeline();
}

function resetPipeline() {
  visitedNodes = [];
  activeNode = null;
  currentLlmNode = null;
  if (nodePipeline) {
    nodePipeline.innerHTML = "";
  }
  if (llmStream) {
    llmStream.innerHTML = "";
  }
}

async function sendMessage(text) {
  if (sending) return;
  sending = true;
  form.querySelector(".send-btn").disabled = true;

  resetPipeline();
  appendMessage("user", text);
  history.push({ role: "user", content: text });

  try {
    const res = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        history,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let assistantDiv = null;
    let assistantContent = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let evt;
        try {
          evt = JSON.parse(trimmed);
        } catch (e) {
          console.error("无法解析流事件:", trimmed, e);
          continue;
        }

        if (evt.type === "node") {
          onNodeEvent(evt.node);
        } else if (evt.type === "llm_token") {
          const node = evt.node || "";
          const delta = evt.delta || "";
          if (!llmStream) continue;
          if (node && node !== currentLlmNode) {
            if (currentLlmNode !== null) {
              llmStream.appendChild(document.createTextNode("\n"));
            }
            currentLlmNode = node;
            const label = NODE_LABELS[node] || node;
            const tag = document.createElement("span");
            tag.className = "llm-node-tag";
            tag.textContent = `[${label}]`;
            llmStream.appendChild(tag);
            llmStream.appendChild(document.createTextNode("\n"));
          }
          if (delta) {
            llmStream.appendChild(document.createTextNode(delta));
            llmStream.scrollTop = llmStream.scrollHeight;
          }
        } else if (evt.type === "done") {
          const fullMessage = evt.message || "";
          if (fullMessage) {
            if (!assistantDiv) {
              assistantDiv = appendMessage("assistant", "");
            }
            const bodyEl = assistantDiv.querySelector(".msg-body");
            if (bodyEl) {
              bodyEl.innerHTML = renderMarkdown(fullMessage);
            }
            chatWindow.scrollTop = chatWindow.scrollHeight;
          }
          const messages = evt.messages || history;
          history = messages;
          activeNode = null;
          currentLlmNode = null;
          renderPipeline();
        }
      }
    }
  } catch (err) {
    console.error(err);
    appendMessage("assistant", "请求失败，请确认后端已在 8000 端口启动。");
  } finally {
    sending = false;
    form.querySelector(".send-btn").disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendMessage(text);
});
