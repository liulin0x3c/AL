const API_BASE = "http://localhost:8000";

const chatWindow = document.getElementById("chat-window");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");

let history = [];
let sending = false;

function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const roleLabel = role === "user" ? "你" : "助手";
  div.innerHTML = `<div class="role">${roleLabel}</div>${content
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>")}`;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(text) {
  if (sending) return;
  sending = true;
  form.querySelector(".send-btn").disabled = true;

  appendMessage("user", text);
  history.push({ role: "user", content: text });

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
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

    const data = await res.json();
    const messages = data.messages || [];
    const last = messages[messages.length - 1];
    if (last && last.role === "assistant") {
      appendMessage("assistant", last.content);
    } else {
      appendMessage("assistant", "后端返回了意外格式，请检查日志。");
    }

    history = messages;
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

