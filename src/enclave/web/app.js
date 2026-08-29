const form = document.querySelector("#query-form");
const input = document.querySelector("#query");
const send = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const intro = document.querySelector("#intro");
const statusEl = document.querySelector("#status");
const statusText = document.querySelector("#status-text");
const loadingTemplate = document.querySelector("#loading-template");
const library = document.querySelector("#library");
const libraryOpen = document.querySelector("#library-open");
const libraryClose = document.querySelector("#library-close");
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#document-file");
const uploadButton = document.querySelector("#upload-button");
const uploadMessage = document.querySelector("#upload-message");
const documentList = document.querySelector("#document-list");
const documentCount = document.querySelector("#document-count");
let pollTimer;
const historyPanel = document.querySelector("#history-panel");
const historyOpen = document.querySelector("#history-open");
const historyClose = document.querySelector("#history-close");
const historyList = document.querySelector("#history-list");
const newChat = document.querySelector("#new-chat");
const loginPanel = document.querySelector("#login-panel");
const loginForm = document.querySelector("#login-form");
const loginButton = document.querySelector("#login-button");
const loginError = document.querySelector("#login-error");
const accountButton = document.querySelector("#account-button");
let currentConversationId = null;

function showLogin() {
  accountButton.hidden = true;
  if (!loginPanel.open) loginPanel.showModal();
  document.querySelector("#login-username").focus();
}

function enterApp(user) {
  if (loginPanel.open) loginPanel.close();
  accountButton.textContent = `${user.username} · Sign out`;
  accountButton.hidden = false;
  loadDocuments();
  loadHistory();
}

async function bootstrap() {
  try {
    const response = await fetch("/v1/auth/me");
    if (!response.ok) return showLogin();
    enterApp(await response.json());
  } catch {
    showLogin();
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginButton.disabled = true;
  loginError.textContent = "";
  const fields = new FormData(loginForm);
  try {
    const response = await fetch("/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: fields.get("username"), password: fields.get("password") }),
    });
    if (!response.ok) throw new Error("Incorrect username or password.");
    loginForm.reset();
    enterApp(await response.json());
  } catch (error) {
    loginError.textContent = error.message;
  } finally {
    loginButton.disabled = false;
  }
});

accountButton.addEventListener("click", async () => {
  await fetch("/v1/auth/logout", { method: "POST" });
  currentConversationId = null;
  conversation.replaceChildren();
  documentList.replaceChildren();
  historyList.replaceChildren();
  showLogin();
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health/ready");
    if (!response.ok) throw new Error("not ready");
    const health = await response.json();
    statusEl.className = "status ready";
    statusText.textContent = health.models === "warm" ? "Local · Models warm" : "Local · Ready";
  } catch {
    statusEl.className = "status error";
    statusText.textContent = "Service unavailable";
  }
}

function addUserTurn(question) {
  conversation.append(element("article", "turn user-turn", question));
}

function addSources(card, data) {
  if (!data.evidence?.length) return;
  const section = element("section", "evidence");
  section.append(element("h3", "evidence-title", `Sources · ${data.evidence.length}`));
  data.evidence.forEach((source, index) => {
    const details = element("details", "source");
    const summary = element("summary");
    summary.append(element("span", "source-id", `E${index + 1}`));
    summary.append(element("span", "source-heading", source.heading_path || "Untitled section"));
    const score = source.rerank_score ?? source.fusion_score;
    summary.append(element("span", "source-score", `score ${Number(score).toFixed(3)}`));
    details.append(summary, element("p", "source-content", source.content));
    section.append(details);
  });
  card.append(section);
}

function addMetrics(card, data, question) {
  const row = element("div", "meta-row");
  row.append(element("span", `metric ${data.verified ? "verified" : ""}`, data.verified ? "✓ Verified" : "Review needed"));
  row.append(element("span", "metric", data.reranked ? "Reranked" : "Rerank skipped"));
  if (data.contextualized) row.append(element("span", "metric verified", "Context used"));
  row.append(element("span", "metric", `Search ${formatMs(data.timings?.retrieval_ms)}`));
  row.append(element("span", "metric", `Answer ${formatMs(data.timings?.generation_ms)}`));
  row.append(element("span", "metric", `Total ${formatMs(data.timings?.total_ms)}`));

  if (data.evidence?.[0]) {
    const controls = element("div", "feedback");
    controls.setAttribute("aria-label", "Rate this result");
    [[true, "Useful", "↑"], [false, "Not useful", "↓"]].forEach(([relevant, label, icon]) => {
      const button = element("button", "", icon);
      button.type = "button";
      button.title = label;
      button.setAttribute("aria-label", label);
      button.addEventListener("click", async () => {
        try {
          const response = await fetch("/v1/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question, chunk_id: data.evidence[0].chunk_id, shown_rank: 1, relevant, stage: data.reranked ? "rerank" : "retrieval" }),
          });
          if (!response.ok) throw new Error("feedback failed");
          controls.querySelectorAll("button").forEach((item) => item.classList.remove("selected"));
          button.classList.add("selected");
        } catch {
          button.title = "Could not save feedback";
        }
      });
      controls.append(button);
    });
    row.append(controls);
  }
  card.append(row);
}

function addAnswer(data, question) {
  const card = element("article", "turn assistant-turn");
  const label = element("div", "turn-label");
  label.append(element("span", "mini-mark", "E"), document.createTextNode("Enclave"));
  card.append(label, element("div", "answer", data.answer));
  if (!data.verified) card.append(element("div", "warning", "This answer did not pass every evidence check. Review the cited source before relying on it."));
  addSources(card, data);
  addMetrics(card, data, question);
  conversation.append(card);
}

function addError(message) {
  const card = element("article", "turn assistant-turn error-card");
  card.append(element("div", "turn-label", "Enclave"), element("div", "answer", message));
  conversation.append(card);
}

async function ask(question) {
  intro.classList.add("compact");
  addUserTurn(question);
  const loading = loadingTemplate.content.firstElementChild.cloneNode(true);
  conversation.append(loading);
  send.disabled = true;
  input.disabled = true;
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });

  try {
    const response = await fetch("/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, top_k: 5, conversation_id: currentConversationId }),
    });
    if (!response.ok) throw new Error(`The local service returned ${response.status}.`);
    const data = await response.json();
    currentConversationId = data.conversation_id;
    loading.remove();
    addAnswer(data, question);
    loadHistory();
  } catch (error) {
    loading.remove();
    addError(error.message || "The local service could not answer right now.");
  } finally {
    send.disabled = false;
    input.disabled = false;
    input.focus();
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

checkHealth();

function renderDocuments(jobs) {
  documentList.replaceChildren();
  documentCount.textContent = jobs.filter((job) => job.status === "complete").length;
  if (!jobs.length) {
    documentList.append(element("p", "document-empty", "No uploaded documents yet."));
    return;
  }
  jobs.forEach((job) => {
    const row = element("article", "document-row");
    const info = element("div");
    info.append(element("p", "document-name", job.filename));
    const meta = element("div", `document-meta ${job.status === "failed" ? "job-failed" : ""}`);
    const statusLabel = job.status === "complete" ? `${job.chunks_written} chunks` : `${job.status} · ${job.progress}%`;
    meta.append(element("span", "", statusLabel));
    if (job.error) meta.append(element("span", "", job.error));
    info.append(meta);
    if (!["complete", "failed"].includes(job.status)) {
      const track = element("div", "progress-track");
      const bar = element("div", "progress-bar");
      bar.style.width = `${job.progress}%`;
      track.append(bar);
      info.append(track);
    }
    row.append(info);
    if (["complete", "failed"].includes(job.status)) {
      const remove = element("button", "delete-document", "Delete");
      remove.type = "button";
      remove.addEventListener("click", async () => {
        if (!window.confirm(`Delete ${job.filename} and its indexed evidence?`)) return;
        const response = await fetch(`/v1/documents/${job.job_id}`, { method: "DELETE" });
        if (response.ok) loadDocuments();
      });
      row.append(remove);
    }
    documentList.append(row);
  });
}

async function loadDocuments() {
  try {
    const response = await fetch("/v1/documents");
    if (!response.ok) throw new Error("Could not load documents");
    const jobs = await response.json();
    renderDocuments(jobs);
    const importing = jobs.some((job) => !["complete", "failed"].includes(job.status));
    window.clearTimeout(pollTimer);
    if (importing) pollTimer = window.setTimeout(loadDocuments, 1200);
  } catch {
    uploadMessage.textContent = "Could not load the local knowledge base.";
  }
}

libraryOpen.addEventListener("click", () => {
  library.showModal();
  loadDocuments();
});
libraryClose.addEventListener("click", () => library.close());
library.addEventListener("click", (event) => {
  if (event.target === library) library.close();
});

["dragenter", "dragover"].forEach((name) => uploadForm.addEventListener(name, (event) => {
  event.preventDefault();
  uploadForm.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => uploadForm.addEventListener(name, (event) => {
  event.preventDefault();
  uploadForm.classList.remove("dragging");
}));
uploadForm.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) fileInput.files = event.dataTransfer.files;
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadMessage.textContent = `Selected: ${fileInput.files[0].name}`;
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  uploadButton.disabled = true;
  uploadMessage.textContent = "Uploading securely to the local service…";
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/v1/documents", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Upload failed");
    uploadMessage.textContent = "Uploaded. Parsing and embedding in the background.";
    fileInput.value = "";
    loadDocuments();
  } catch (error) {
    uploadMessage.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
});

bootstrap();

function resetConversation() {
  currentConversationId = null;
  conversation.replaceChildren();
  intro.classList.remove("compact");
  historyPanel.close();
  input.focus();
}

function renderHistory(items) {
  historyList.replaceChildren();
  if (!items.length) {
    historyList.append(element("p", "document-empty", "No saved conversations yet."));
    return;
  }
  items.forEach((item) => {
    const row = element("article", "history-row");
    const select = element("button", "history-select");
    select.type = "button";
    select.append(element("span", "history-title", item.title));
    select.append(element("span", "history-preview", item.preview || `${item.message_count} messages`));
    select.addEventListener("click", () => openConversation(item.conversation_id));
    const remove = element("button", "history-delete", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Delete ${item.title}`);
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete conversation “${item.title}”?`)) return;
      const response = await fetch(`/v1/conversations/${item.conversation_id}`, { method: "DELETE" });
      if (response.ok) {
        if (currentConversationId === item.conversation_id) resetConversation();
        loadHistory();
      }
    });
    row.append(select, remove);
    historyList.append(row);
  });
}

async function loadHistory() {
  try {
    const response = await fetch("/v1/conversations");
    if (!response.ok) throw new Error();
    renderHistory(await response.json());
  } catch {
    historyList.replaceChildren(element("p", "document-empty", "Could not load local history."));
  }
}

async function openConversation(identifier) {
  const response = await fetch(`/v1/conversations/${identifier}`);
  if (!response.ok) return;
  const saved = await response.json();
  currentConversationId = identifier;
  conversation.replaceChildren();
  intro.classList.add("compact");
  for (let index = 0; index < saved.messages.length; index += 1) {
    const message = saved.messages[index];
    if (message.role === "user") {
      addUserTurn(message.content);
    } else {
      const previous = saved.messages[index - 1];
      addAnswer({ ...message.metadata, answer: message.content }, previous?.content || "Saved question");
    }
  }
  historyPanel.close();
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

historyOpen.addEventListener("click", () => {
  historyPanel.showModal();
  loadHistory();
});
historyClose.addEventListener("click", () => historyPanel.close());
historyPanel.addEventListener("click", (event) => {
  if (event.target === historyPanel) historyPanel.close();
});
newChat.addEventListener("click", resetConversation);
loadHistory();
