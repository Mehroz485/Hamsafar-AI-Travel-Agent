"use strict";

let threadId = null;
let busy = false;

const $ = (id) => document.getElementById(id);

// Surface any uncaught error directly on the page so nothing fails silently.
window.addEventListener("error", (e) => {
  console.error("Uncaught error:", e.error || e.message);
  const el = $("error");
  if (el) {
    el.textContent = "A script error occurred: " + (e.message || "unknown error");
    el.hidden = false;
  }
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled promise rejection:", e.reason);
  const el = $("error");
  if (el) {
    el.textContent = "A script error occurred: " + (e.reason?.message || e.reason || "unknown error");
    el.hidden = false;
  }
});

/* ---------- Agent labels ---------- */

const AGENTS = {
  supervisor:        { icon: "🧭", running: "Supervisor is checking and planning your request", done: "Supervisor planned the request" },
  flight_agent:      { icon: "✈️", running: "Flight agent is searching flights", done: "Flight agent finished" },
  hotel_agent:       { icon: "🏨", running: "Hotel agent is finding hotels", done: "Hotel agent finished" },
  weather_agent:     { icon: "🌦️", running: "Weather agent is calling the weather MCP", done: "Weather agent finished" },
  budget_agent:      { icon: "💰", running: "Budget agent is checking the budget", done: "Budget agent finished" },
  itinerary_agent:   { icon: "🗓️", running: "Itinerary agent is building the draft", done: "Itinerary draft ready" },
  human_approval:    { icon: "🙋", running: "Waiting for your review", done: "Review received" },
  final_agent:       { icon: "📝", running: "Final agent is writing the final plan", done: "Final plan written" },
  guardrail_blocked: { icon: "🛡️", running: "Guardrail is blocking this request", done: "Request blocked by the guardrail" },
};

/* ---------- Helpers ---------- */

function esc(s) {
  // If the data is an object/array, format it nicely instead of returning "[object Object]"
  if (s !== null && typeof s === 'object') {
    s = JSON.stringify(s, null, 2);
  }
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderMarkdown(text) {
  const raw = String(text ?? "");
  if (window.marked && window.DOMPurify) {
    try {
      return window.DOMPurify.sanitize(window.marked.parse(raw));
    } catch (e) {
      console.error("Markdown rendering failed", e);
    }
  }
  return `<pre class="plain">${esc(raw)}</pre>`;
}

function showError(message) {
  const el = $("error");
  el.textContent = message;
  el.hidden = false;
}

function clearError() {
  $("error").hidden = true;
  $("feedback-error").hidden = true;
}

function setBusy(state) {
  busy = state;
  ["sendBtn", "approveBtn", "reviseBtn", "newTripBtn"].forEach((id) => {
    if ($(id))$(id).disabled = state;
  });
  $("sendBtn").textContent = state ? "Working…" : "Generate plan";
}

/* ---------- Live progress ---------- */

function resetProgress() {
  $("agent-progress").innerHTML = "";
  $("progress-wrap").hidden = false;
}

function updateStep(ev) {
  const list = $("agent-progress");
  const meta = AGENTS[ev.agent] || { icon: "•", running: ev.agent, done: ev.agent };

  let row = list.querySelector(`[data-agent="${ev.agent}"]`);
  if (!row) {
    row = document.createElement("li");
    row.dataset.agent = ev.agent;
    list.appendChild(row);
  }

  if (ev.type === "agent_start") {
    row.className = "step running";
    row.innerHTML = `<span class="dot"><span class="spin"></span></span><span>${meta.icon} ${esc(meta.running)}</span>`;
  } else if (ev.type === "waiting_approval") {
    row.className = "step waiting";
    row.innerHTML = `<span class="dot">⏸</span><span>${meta.icon} ${esc(meta.running)}</span>`;
  } else if (ev.type === "agent_done") {
    if (ev.error) {
      row.className = "step failed";
      row.innerHTML = `<span class="dot">✕</span><span>${meta.icon} ${esc(meta.done)} with an error</span>`;
    } else {
      row.className = "step done";
      row.innerHTML = `<span class="dot">✓</span><span>${meta.icon} ${esc(meta.done)}</span>`;
    }
  }
}

/* ---------- Streaming client ---------- */

async function streamTravel(url, body) {
  resetProgress();

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let msg = `Server error (${res.status})`;
    try {
      const j = await res.json();
      if (j && j.error) msg = j.error;
    } catch (_) { /* ignore */ }
    throw new Error(msg);
  }

  // FIX: If the API sends standard JSON instead of a stream, return it safely immediately
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  // FIX: Robust parser that splits purely on \n to prevent mid-JSON splitting crashes
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete trailing line in buffer
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;
      
      try {
        const ev = JSON.parse(trimmed.slice(5).trim());
        if (ev.type === "final") finalData = ev.data;
        else if (ev.type === "error") throw new Error(ev.error || "Agent run failed.");
        else updateStep(ev);
      } catch (e) {
        console.warn("Skipped malformed stream chunk");
      }
    }
  }

  if (!finalData) throw new Error("The agent stopped before returning a result.");
  return finalData;
}

/* ---------- Result rendering ---------- */

function renderAgentInfo(data) {
  const ALL = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"];
  const selected = data.selected_agents || [];

  const chips = ALL.map((a) => {
    const on = selected.includes(a);
    return `<span class="agent-chip ${on ? "on" : "off"}">${a.replace("_agent", "")}</span>`;
  }).join("");

  const block = (title, text) =>
    text ? `<details><summary>${esc(title)}</summary><pre>${esc(text)}</pre></details>` : "";

  $("agent-info").innerHTML = `
    <div class="info">
      <h3>How this plan was built</h3>
      <div class="agent-chips">${chips}</div>
      <p><b>Guardrail:</b> ${data.guardrail_allowed === false ? "blocked" : "allowed"}${data.guardrail_reason ? " — " + esc(data.guardrail_reason) : ""}</p>
      <p><b>Supervisor reasoning:</b> ${esc(data.supervisor_reasoning) || "—"}</p>
      <p><b>LLM calls:</b> ${data.llm_calls ?? 0}</p>
      ${block("Trip constraints", data.trip_constraints)}
      ${block("Flight results", data.flight_results)}
      ${block("Hotel results", data.hotel_results)}
      ${block("Weather results", data.weather_results)}
      ${block("Budget results", data.budget_results)}
    </div>`;
}

function showResult(data) {
  try {
    $("result").hidden = false;
    $("progress-wrap").hidden = true;

    let title = "Final plan";
    if (data.guardrail_allowed === false) title = "Request blocked";
    else if (data.requires_approval) title = "Draft itinerary";
    $("result-title").textContent = title;

    $("result-body").innerHTML = renderMarkdown(data.answer || "No response text was returned.");

    $("approval").hidden = !data.requires_approval;
    if (data.requires_approval) {
      $("approval-note").textContent =
        data.approval_request || "Review the draft. Approve it or request changes.";
      $("feedback").value = "";
    }

    renderAgentInfo(data);
  } catch (err) {
    console.error("showResult failed:", err);
    showError("Could not display the result: " + (err.message || err));
  }
}

/* ---------- Actions ---------- */

async function sendMessage() {
  if (busy) return;
  const message = $("userInput").value.trim();
  if (!message) {
    showError("Type a trip request first.");
    return;
  }

  clearError();
  $("result").hidden = true;
  setBusy(true);
  try {
    const data = await streamTravel("/api/travel/stream", { message, thread_id: threadId });
    threadId = data.thread_id || threadId;
    showResult(data);
  } catch (err) {
    console.error(err);
    showError(err.message || "Something went wrong.");
  } finally {
    setBusy(false);
  }
}

async function submitApproval(approved) {
  if (busy) return;
  const feedback = $("feedback").value.trim();

  clearError();
  if (!approved && !feedback) {
    $("feedback-error").hidden = false;
    $("feedback").focus();
    return;
  }
  if (!threadId) {
    showError("No active plan to review. Generate a plan first.");
    return;
  }

  setBusy(true);
  try {
    const data = await streamTravel("/api/travel/approve/stream", {
      thread_id: threadId,
      approved,
      feedback,
    });
    showResult(data);
  } catch (err) {
    console.error(err);
    showError(err.message || "Something went wrong.");
  } finally {
    setBusy(false);
  }
}

function newTrip() {
  if (busy) return;
  threadId = null;
  $("userInput").value = "";
  $("result").hidden = true;
  $("progress-wrap").hidden = true;
  $("agent-progress").innerHTML = "";
  clearError();
  $("userInput").focus();
}

/* ---------- Wiring ---------- */

$("sendBtn").addEventListener("click", (e) => {
  e.preventDefault();
  sendMessage();
});
$("approveBtn").addEventListener("click", (e) => { e.preventDefault(); submitApproval(true); });
$("reviseBtn").addEventListener("click", (e) => { e.preventDefault(); submitApproval(false); });
$("newTripBtn").addEventListener("click", (e) => { e.preventDefault(); newTrip(); });

$("userInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendMessage();
  }
});

$("examples").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  $("userInput").value = chip.dataset.prompt;
  $("userInput").focus();
});