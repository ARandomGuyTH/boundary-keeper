const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const submitBtn = document.getElementById("submit-prompt");
const chatBox = document.getElementById("chat-box");
//var output

let isWaiting = false;

// Endpoints: change these when the backend is settled
const CHAT_URL = "/askLLM?prompt=";
const BOUNDS_URL = "/classify?text=";           // classifies the response

const BOUNDS_DISPLAY = {
    "GENERAL":   { text: "In bounds",     variant: "badge-in" },
    "GREY AREA": { text: "Borderline",    variant: "badge-border" },
    "PERSONAL":  { text: "Out of bounds", variant: "badge-out" },
};

const USE_MOCK = true;            // set to false when the backend is connected
const MOCK_VERDICT = "PERSONAL";  // try "GENERAL", "GREY AREA", or "PERSONAL" to see different behaviours

async function fetchJSON(url) {
    if (USE_MOCK) {
        await new Promise(resolve => setTimeout(resolve, 600)); // fake delay
        if (url.startsWith(CHAT_URL)) return { result: "This is a fake model reply for testing." };
        if (url.includes("type=1")) return { result: MOCK_VERDICT };
        return { result: "GENERAL" };   // prompt type
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error("Request failed: " + response.status);
    const data = await response.json();
    console.log(data);
    if (data.error) throw new Error(data.error);
    return data;
}

function addMessage(text, className) {
    const bubble = document.createElement("div");
    bubble.classList.add("chat-message", className);
    bubble.textContent = text;
    chatBox.append(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
    return bubble;
}

function setBadge(elementId, text, variant = "badge-neutral") {
    const el = document.getElementById(elementId);
    el.textContent = text;
    el.className = "badge " + variant;
}

// Each analysis has its own try/catch, so a failure never affects the chat
async function analysePrompt(text) {
    setBadge("prompt-type", "Analysing…");
    try {
        const data = await fetchJSON(BOUNDS_URL + encodeURIComponent(text) + "&type=0");
        setBadge("prompt-type", data.result);
    } catch (err) {
        console.error(err);
        setBadge("prompt-type", "Unavailable");
    }
}

async function analyseResponse(text) {
    setBadge("bounds-verdict", "Analysing…");
    try {
        const data = await fetchJSON(BOUNDS_URL + encodeURIComponent(text) + "&type=1");
        const label = String(data.result).toUpperCase().replace(/[_-]+/g, " ").trim();
        const display = BOUNDS_DISPLAY[label];

        if (!display) {
            // a label we don't recognise: treat as unverified, never as safe
            setBadge("bounds-verdict", "Not verified", "badge-border");
            return null;
}
        setBadge("bounds-verdict", display.text, display.variant);
        return label;

    } catch (err) {
        console.error(err);
        setBadge("bounds-verdict", "Not verified", "badge-border");
        return null;    // couldn't check
    }
}


function renderReply(bubble, text, verdict) {
    if (verdict !== "PERSONAL" && verdict !== null) {
        bubble.textContent = text;
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }

    bubble.classList.add("flagged");
    bubble.textContent = "";

    const note = document.createElement("p");
    note.textContent = verdict === null
        ? "This response couldn't be checked, so it has been withheld."
        : "This response was withheld because it may contain or request personal information.";

    const hidden = document.createElement("p");
    hidden.textContent = text;
    hidden.hidden = true;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "reveal-btn";
    btn.textContent = "Show anyway";
    btn.addEventListener("click", () => {
        hidden.hidden = !hidden.hidden;
        btn.textContent = hidden.hidden ? "Show anyway" : "Hide response";
    });

    bubble.append(note, btn, hidden);
}

promptForm.addEventListener("submit", async function (event) {
    event.preventDefault(); // stops page reloading

    if (isWaiting) return;

    const userInput = promptInput.value.trim();
    if (!userInput) return;

    addMessage(userInput, "sent-message");

    promptForm.reset();
    isWaiting = true;
    submitBtn.disabled = true;
    promptInput.disabled = true;

    const messageBubble = addMessage("Thinking...", "received-message");

    analysePrompt(userInput); // runs alongside the chat request, not awaited

    try {
        const output = await fetchJSON(CHAT_URL + encodeURIComponent(userInput));
        messageBubble.textContent = "Checking response…";
        const verdict = await analyseResponse(output.result);
        renderReply(messageBubble, output.result, verdict);
    } catch (err) {
        console.error(err);
        messageBubble.textContent = "Sorry, something went wrong. Please try again.";
        setBadge("bounds-verdict", "-");
    } finally {
        // Always runs, so the form can never get stuck locked
        isWaiting = false;
        submitBtn.disabled = false;
        promptInput.disabled = false;
        promptInput.focus();
    }
});
