import ollama
import tensorflow as tf
import classifier 

MODEL_PATH = "classifierModel.keras"
OLLAMA_MODEL = "llama3.2" # change this to anything 

classifier = tf.keras.models.load_model(MODEL_PATH)
print("Classifier loaded.\n")

def askOllama(question, model=OLLAMA_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={
            "num_predict": 150  # limit no. of words in a response
        }
    )
    return response["message"]["content"]


def classify(text):
    prediction = classifier.predict(tf.constant([text]), verbose=0)
    return float(prediction[0][0])


def interpret(score):
    if score >= 0.7:
        return "GENERAL"
    elif score <= 0.3:
        return "PERSONAL"
    else:
        return "GREY-AREA"
        


def main():
    print("Ask the AI for financial advice, and we'll classify the response.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() == "quit":
            break

        response = askOllama(question)
        print(f"{OLLAMA_MODEL}: {response}\n")

        score = classify(response)
        label = interpret(score)

        print(f"Classifier score: {score:.2f}")
        print(f"Label: {label}\n")


if __name__ == "__main__":
    main()