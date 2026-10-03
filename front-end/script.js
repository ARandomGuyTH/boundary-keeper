const button = document.getElementById("button1");
const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input")
const submitBtn = document.getElementById("submit-prompt");

let isWaiting= false;

var current_prompt = "Some people hedge currency risk while others accept it.";

button.addEventListener("click", async function() {
    const response = await fetch("/classify?text=" + encodeURIComponent(current_prompt));

    if (!response.ok) {
        console.error("Request failed:", response.status);
        return;
    }

    const outputClass = await response.json();
    console.log(outputClass.result);
})

promptForm.addEventListener("submit", async function (event) {
    event.preventDefault(); //stops page reloading

    if (isWaiting) return;

    const userInput = promptInput.value.trim();
    if (!userInput) return;

    promptForm.reset();
    submitBtn.disabled = true;
    isWaiting = true;
    promptInput.value = "Thinking...";
    promptInput.disabled = true;

    const response = await fetch("/askLLM?prompt=" + encodeURIComponent(userInput));

    if (!response.ok) {
        console.error("Request failed:", response.status);
        isWaiting = false;
        submitBtn.disabled = false;
        promptInput.disabled = false;
        return;
    }

    const output = await response.json(); //return value of LLM 
    console.log(output.result);
    
    isWaiting = false;
    submitBtn.disabled = false;
    promptInput.value = "Enter prompt...";
    promptInput.disabled = false;
})
