const form = document.querySelector("#query-form");
const input = document.querySelector("#query");
const send = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const intro = document.querySelector("#intro");
const statusEl = document.querySelector("#status");
const statusText = document.querySelector("#status-text");
const loadingTemplate = document.querySelector("#loading-template");

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
      body: JSON.stringify({ query: question, top_k: 5 }),
    });
    if (!response.ok) throw new Error(`The local service returned ${response.status}.`);
    const data = await response.json();
    loading.remove();
    addAnswer(data, question);
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
