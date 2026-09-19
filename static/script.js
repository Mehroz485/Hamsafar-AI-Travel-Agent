let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let isSending = false;

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

    sendBtn.disabled = isLoading;
    quickPrompts.forEach((button) => { button.disabled = isLoading; });

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

function showResult(answer, threadId) {
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

    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

async function sendMessage() {
    if (isSending) {
        return;
    }

    hideError();

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

        showResult(data.answer, data.thread_id);

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