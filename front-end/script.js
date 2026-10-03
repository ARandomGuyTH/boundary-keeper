const button = document.getElementById("button1");
const submitPromptBtn = document.getElementById("submit-prompt")

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

button.addEventListener("click", function() {
    console.log("Button was clicked!");
    //function logic goes here 
    alert("Hello test");
})

submitPromptBtn.addEventListener("click", function()) {
    var userInput = document.getElementById("prompt-input").value
}
