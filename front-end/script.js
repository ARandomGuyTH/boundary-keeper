const button = document.getElementById("classification");
const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const submitBtn = document.getElementById("submit-prompt");
const chatBox = document.getElementById("chat-box");
var output

let isWaiting = false;

// Endpoints: change these when the backend is settled
const CHAT_URL = "/askLLM?prompt=";
const BOUNDS_URL = "/classify?text=";           // classifies the response
const PROMPT_TYPE_URL = "/classifyPrompt?text="; // placeholder, rename to yours

// Label -> badge colour. Unknown labels fall back to grey.
const BOUNDS_VARIANTS = {
    "in bounds": "badge-in",
    "borderline": "badge-border",
    "out of bounds": "badge-out",
};

// Fetch JSON and turn every kind of failure into a thrown Error
async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Request failed: " + response.status);
    const data = await response.json();
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
        const data = await fetchJSON(PROMPT_TYPE_URL + encodeURIComponent(text));
        setBadge("prompt-type", data.result);
    } catch (err) {
        console.error(err);
        setBadge("prompt-type", "Unavailable");
    }
}

async function analyseResponse(text) {
    setBadge("bounds-verdict", "Analysing…");
    try {
        const data = await fetchJSON(BOUNDS_URL + encodeURIComponent(text));
        const label = String(data.result).toLowerCase();
        setBadge("bounds-verdict", data.result, BOUNDS_VARIANTS[label]);
    } catch (err) {
        console.error(err);
        setBadge("bounds-verdict", "Unavailable");
    }
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
        messageBubble.textContent = output.result;
        analyseResponse(output.result); // not awaited either
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

/*
var current_prompt = "Some people hedge currency risk while others accept it.";

button.addEventListener("click", async function() {
    const response = await fetch("/classify?text=" + encodeURIComponent(output.result));
    const resultsBox = document.getElementById("classification");

    if (!response.ok) {
        console.error("Request failed:", response.status);
        return;
    }

    const outputClass = await response.json();
    resultsBox.innerHTML = "This kind of response is: " + outputClass.result;
    console.log(outputClass.result);

})

promptForm.addEventListener("submit", async function (event) {
    event.preventDefault(); //stops page reloading

    if (isWaiting) return;

    const userInput = promptInput.value.trim();
    const chatBox = document.getElementById("chat-box")
    if (!userInput) return;
    let sentMessage = document.createElement('div');
    sentMessage.classList.add('chat-message')
    sentMessage.classList.add('sent-message')
    sentMessage.innerHTML = userInput
    chatBox.append(sentMessage);

    promptForm.reset();
    submitBtn.disabled = true;
    isWaiting = true;
    promptInput.value = "";
    promptInput.disabled = true;

    let messageBubble = document.createElement('div');
    messageBubble.classList.add('chat-message')
    messageBubble.classList.add('received-message')
    messageBubble.innerHTML = "Thinking..."
    chatBox.append(messageBubble);

    const response = await fetch("/askLLM?prompt=" + encodeURIComponent(userInput));

    if (!response.ok) {
        console.error("Request failed:", response.status);
        isWaiting = false;
        submitBtn.disabled = false;
        promptInput.disabled = false;
        return;
    }

    output = await response.json(); //return value of LLM 
    
    messageBubble.innerHTML = output.result
    

    console.log(output.result);
    
    isWaiting = false;
    submitBtn.disabled = false;
    promptInput.value = "";
    promptInput.disabled = false;
})
*/