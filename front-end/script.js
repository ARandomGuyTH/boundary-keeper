const button = document.getElementById("button1");
const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input")
const submitBtn = document.getElementById("submit-prompt");
var output

let isWaiting= false;

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
