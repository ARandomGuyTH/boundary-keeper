const button = document.getElementById("button1");

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
