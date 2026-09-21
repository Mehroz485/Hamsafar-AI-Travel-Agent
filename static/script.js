let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let isSending = false;

// Friendly names for the agent ids sent by the backend
const AGENT_LABELS = {
    flight_agent: "Flights",
    hotel_agent: "Hotels",
    weather_agent: "Weather",
    budget_agent: "Budget",
    itinerary_agent: "Itinerary"
};

function setPrompt(text) {
    const input = document.getElementById("userInput");
    input.value = text;
    autoGrow(input);
    input.focus();
}

function autoGrow(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 420) + "px";
}

function setLoading(isLoading) {
    isSending = isLoading;

    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");
    const quickPrompts = document.querySelectorAll(".quick-prompts button");
    const approvalButtons = document.querySelectorAll(".approval-actions button");

    sendBtn.disabled = isLoading;
    quickPrompts.forEach((button) => { button.disabled = isLoading; });
    approvalButtons.forEach((button) => { button.disabled = isLoading; });

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

function showError(message) {
    const errorBox = document.getElementById("errorBox");

    errorBox.setAttribute("role", "alert");
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideError() {
    const errorBox = document.getElementById("errorBox");

    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

// Show small chips for the agents the supervisor picked (e.g. Flights, Hotels).
// If the backend sends no list, the chips stay hidden.
function renderAgents(selectedAgents) {
    const agentsUsed = document.getElementById("agentsUsed");

    if (!agentsUsed) {
        return;
    }

    agentsUsed.innerHTML = "";

    if (!Array.isArray(selectedAgents) || selectedAgents.length === 0) {
        agentsUsed.classList.add("hidden");
        return;
    }

    selectedAgents.forEach((name) => {
        const chip = document.createElement("span");
        chip.className = "agent-chip";
        chip.textContent = AGENT_LABELS[name] || name;
        agentsUsed.appendChild(chip);
    });

    agentsUsed.classList.remove("hidden");
}

// Show the human review box (Approve / Request changes) under the draft plan.
function showApproval(message) {
    const approvalBox = document.getElementById("approvalBox");
    const approvalText = document.getElementById("approvalText");
    const feedbackInput = document.getElementById("feedbackInput");

    if (!approvalBox) {
        return;
    }

    approvalText.textContent = message || "Please review this draft plan.";
    feedbackInput.value = "";
    approvalBox.classList.remove("hidden");
}

// Hide the human review box (used when the plan is final or a new request starts).
function hideApproval() {
    const approvalBox = document.getElementById("approvalBox");

    if (approvalBox) {
        approvalBox.classList.add("hidden");
    }
}

// Send the user's decision (approve or request changes) to the backend,
// then show the final plan that comes back.
async function submitApproval(approved) {
    if (isSending) {
        return;
    }

    hideError();

    const feedback = document.getElementById("feedbackInput").value.trim();

    // When asking for changes, the user must say what to change.
    if (!approved && !feedback) {
        showError("Please write what should change before requesting changes.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel/approve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                thread_id: currentThreadId,
                approved: approved,
                feedback: feedback
            })
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The server sent back something unreadable. Please try again.");
        }

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        // The plan is final now, so hide the review box and show the final answer.
        hideApproval();
        showResult(data.answer, data.thread_id, data.selected_agents);

    } catch (error) {
        if (error instanceof TypeError) {
            showError("Couldn't reach the server. Check your connection and try again.");
        } else {
            showError(error.message);
        }
    } finally {
        setLoading(false);
    }
}

function showResult(answer, threadId, selectedAgents) {
    latestAnswerMarkdown = answer;

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");

    if (typeof marked !== "undefined") {
        resultBox.innerHTML = marked.parse(answer);
    } else {
        resultBox.innerText = answer;
    }

    threadInfo.textContent = `Thread ID: ${threadId}`;
    renderAgents(selectedAgents);

    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

async function sendMessage() {
    const inputEl = document.getElementById("userInput");   // CHANGE to your input's id
    const message = inputEl.value.trim();
    if (!message || isSending) return;

    isSending = true;
    try {
        const data = await streamTravel("/api/travel/stream", {
            message: message,
            thread_id: threadId,
        });

        if (!data || !data.success) {
            throw new Error((data && data.error) || "Something went wrong.");
        }

        threadId = data.thread_id;
        showResult(data);
    } catch (err) {
        console.error(err);
        alert(err.message);   // swap for your own error display if you have one
    } finally {
        isSending = false;
    }
}

    hideError();
    // A new request starts, so hide any old review box.
    hideApproval();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        input.focus();
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: currentThreadId
            })
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The server sent back something unreadable. Please try again.");
        }

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);

        // The guardrail blocked this request (not a travel question):
        // show its message instead of a travel plan.
        if (data.guardrail_allowed === false) {
            showError(data.answer || data.guardrail_reason || "Hamsafar can only help with travel-planning requests.");
            return;
        }

        showResult(data.answer, data.thread_id, data.selected_agents);

        // The graph paused: this is a DRAFT. Show Approve / Request changes buttons.
        if (data.requires_approval) {
            showApproval(data.approval_request);
        }

    } catch (error) {
        if (error instanceof TypeError) {
            showError("Couldn't reach the server. Check your connection and try again.");
        } else {
            showError(error.message);
        }
    } finally {
        setLoading(false);
    }
}

function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;

            copyBtn.textContent = "Copied";
            copyBtn.disabled = true;

            setTimeout(() => {
                copyBtn.textContent = oldText;
                copyBtn.disabled = false;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;

    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: {
            type: "jpeg",
            quality: 0.98
        },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff"
        },
        jsPDF: {
            unit: "in",
            format: "a4",
            orientation: "portrait"
        },
        pagebreak: {
            mode: ["avoid-all", "css", "legacy"]
        }
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("userInput");

    if (input) {
        input.addEventListener("input", function () {
            autoGrow(input);
        });
    }
});

document.addEventListener("keydown", function (event) {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }

    if (event.key === "Escape") {
        hideError();
    }
});


const AGENT_LABELS = {
  supervisor: "🧭 Supervisor: checking request & planning",
  flight_agent: "✈️ Flight agent: searching flights",
  hotel_agent: "🏨 Hotel agent: finding hotels",
  weather_agent: "🌦️ Weather agent: calling weather MCP",
  budget_agent: "💰 Budget agent: analysing budget",
  itinerary_agent: "🗓️ Itinerary agent: building draft",
  human_approval: "🙋 Waiting for your approval",
  final_agent: "📝 Final agent: writing final plan",
  guardrail_blocked: "🛡️ Guardrail: request blocked",
};

function progressBox() {
  let box = document.getElementById("agent-progress");
  if (!box) {
    box = document.createElement("div");
    box.id = "agent-progress";
    const anchor = document.getElementById("result") || document.body;
    anchor.insertAdjacentElement("beforebegin", box);
  }
  box.innerHTML = "";
  return box;
}

function updateStep(box, ev) {
  let row = box.querySelector(`[data-agent="${ev.agent}"]`);
  if (!row) {
    row = document.createElement("div");
    row.dataset.agent = ev.agent;
    box.appendChild(row);
  }
  const label = AGENT_LABELS[ev.agent] || ev.agent;
  if (ev.type === "agent_start") { row.className = "step running"; row.innerHTML = `<span class="spin"></span>${label}`; }
  else if (ev.type === "waiting_approval") { row.className = "step waiting"; row.innerHTML = `⏸ ${label}`; }
  else if (ev.type === "agent_done") { row.className = ev.error ? "step failed" : "step done"; row.innerHTML = `${ev.error ? "❌" : "✅"} ${label}`; }
}

async function streamTravel(url, body) {
  const box = progressBox();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const p of parts) {
      if (!p.startsWith("data: ")) continue;
      const ev = JSON.parse(p.slice(6));
      if (ev.type === "final") finalData = ev.data;
      else if (ev.type === "error") throw new Error(ev.error);
      else updateStep(box, ev);
    }
  }
  return finalData;
}