const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const submitBtn = document.getElementById("submit-prompt");
const chatBox = document.getElementById("chat-box");
//var output

let isWaiting = false;

// Endpoints: change these when the backend is settled
const CHAT_URL = "/askLLM?prompt=";
const BOUNDS_URL = "/classify?text=";           // classifies the response

// The one label that means "withhold this reply". Change it here if needed.
const OUT_OF_BOUNDS_LABEL = "PERSONAL";

const BOUNDS_DISPLAY = {
    "GENERAL": {
        text: "In bounds",
        variant: "badge-in",
        note: "General, group-level guidance. No personal recommendation is made."
    },
    "GREY AREA": {
        text: "Borderline",
        variant: "badge-border",
        note: "It's unclear whether this counts as personal advice. Treat with care."
    },
    "PERSONAL": {
        text: "Out of bounds",
        variant: "badge-out",
        note: "Recommends a specific product or action for you personally, so it has been withheld."
    },
};

const UNVERIFIED_NOTE = "This response couldn't be checked. Treat it with caution.";

const PROMPT_TYPE_DISPLAY = {
    "NEUTRAL": {
        text: "Neutral", variant: "badge-in",
        note: "An ordinary question with no attempt to get a personal recommendation."
    },
    "DIRECT SOLICITATION": {
        text: "Direct request", variant: "badge-border",
        note: "Directly asks the assistant to pick something for you."
    },
    "EMOTIONAL PRESSURE": {
        text: "Emotional pressure", variant: "badge-border",
        note: "Uses distress or urgency to push for personal advice."
    },
    "HYPOTHETICAL": {
        text: "Hypothetical", variant: "badge-border",
        note: "\"If you were me…\" framing used to draw out a recommendation."
    },
    "FALSE AUTHORITY": {
        text: "False authority", variant: "badge-out",
        note: "Claims that someone has authorised the assistant to give personal advice."
    },
    "ENDORSEMENT BAIT": {
        text: "Endorsement bait", variant: "badge-border",
        note: "Tries to get the assistant to approve a choice that may not suit the person."
    },
    "JAILBREAK/RULE BYPASS": {
        text: "Rule bypass", variant: "badge-out",
        note: "Asks the assistant to ignore or drop its safety rules."
    },
    "INDIRECT": {
        text: "Disguised request", variant: "badge-border",
        note: "A personal recommendation disguised as a different task."
    },
};

const USE_MOCK = true;            // set to false when the backend is connected
const MOCK_VERDICT = "PERSONAL";  // try "GENERAL", "GREY AREA", or "PERSONAL" to see different behaviours
const MOCK_PROMPT_TYPE = "HYPOTHETICAL"; // try any of the keys in PROMPT_TYPE_DISPLAY

async function fetchJSON(url) {
    if (USE_MOCK) {
        await new Promise(resolve => setTimeout(resolve, 600)); // fake delay
        if (url.startsWith(CHAT_URL)) return { result: "This is a fake model reply for testing." };
        if (url.includes("type=1")) return { result: MOCK_VERDICT };
        return { result: MOCK_PROMPT_TYPE };   // prompt type
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error("Request failed: " + response.status);
    const data = await response.json();
    console.log(data);
    if (data.error) throw new Error(data.error);
    return data;
}

function setNote(elementId, text) {
    document.getElementById(elementId).textContent = text;
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
    setNote("prompt-type-note", "");
    try {
        const data = await fetchJSON(BOUNDS_URL + encodeURIComponent(text) + "&type=0");
        const label = String(data.result).toUpperCase().replace(/[_-]+/g, " ").trim();
        const display = PROMPT_TYPE_DISPLAY[label];

        setBadge("prompt-type", display ? display.text : data.result, display ? display.variant : undefined);
        setNote("prompt-type-note", display ? display.note : "");
    } catch (err) {
        console.error(err);
        setBadge("prompt-type", "Unavailable");
        setNote("prompt-type-note", "");
    }
}

async function analyseResponse(text) {
    setBadge("bounds-verdict", "Analysing…");
    setNote("bounds-verdict-note", "");
    try {
        const data = await fetchJSON(BOUNDS_URL + encodeURIComponent(text) + "&type=1");
        const label = String(data.result).toUpperCase().replace(/[_-]+/g, " ").trim();
        const display = BOUNDS_DISPLAY[label];

        if (!display) {
            setBadge("bounds-verdict", "Not verified", "badge-border");
            setNote("bounds-verdict-note", UNVERIFIED_NOTE);
            return null;
        }

        setBadge("bounds-verdict", display.text, display.variant);
        setNote("bounds-verdict-note", display.note);
        return label;
    } catch (err) {
        console.error(err);
        setBadge("bounds-verdict", "Not verified", "badge-border");
        setNote("bounds-verdict-note", UNVERIFIED_NOTE);
        return null;
    }
}


function renderReply(bubble, text, verdict) {
    if (verdict !== OUT_OF_BOUNDS_LABEL && verdict !== null) {
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
        setNote("bounds-verdict-note", "");
    } finally {
        // Always runs, so the form can never get stuck locked
        isWaiting = false;
        submitBtn.disabled = false;
        promptInput.disabled = false;
        promptInput.focus();
    }
});
