import ollama
import tensorflow as tf
import classifier 
import flask
import numpy as np
import inputClassifier

app = flask.Flask(__name__, static_folder="front-end", static_url_path="")

MODEL_PATH = "classifierModel.keras"
INPUT_MODEL_PATH = "inputClassifierModel.keras"
OLLAMA_MODEL = "llama3.2" # change this to anything 

classifier = tf.keras.models.load_model(MODEL_PATH)
inputClassifier = tf.keras.models.load_model(MODEL_PATH)

print("Classifiers loaded.\n")

@app.route("/") #on enter website return the web page
def index():
    return app.send_static_file("index.html")

#FLASK WRAPPED CODE FOR WEB BACKEND
@app.route("/askLLM")
def webAskLLM():
    prompt = flask.request.args.get("prompt", "").strip()    

    return flask.jsonify({"result": askOllama(prompt)})

@app.route("/classify")
def webClassify():
    text = flask.request.args.get("text").strip()
    return flask.jsonify({"result": classify(text)})

def askOllama(question, model=OLLAMA_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={
            "num_predict": 100  # limit no. of words in a response
        }
    )
    return response["message"]["content"]

def classify(text):
    prediction = classifier.predict(tf.constant([text]))
  
    if prediction[0][0] >= prediction[0][1] and prediction[0][0] >= prediction[0][2]:
        return "PERSONAL"
    elif prediction[0][1] >= prediction[0][0] and prediction[0][1] >= prediction[0][2]:
        return "GREY AREA"
    else:
        return "GENERAL"


def main():
    print("Ask the AI for financial advice, and we'll classify the response.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() == "quit":
            break
        
        response = askOllama(question)
        print(f"{OLLAMA_MODEL}: {response}\n")
        print(f"Classification: {classify(response)}")


        


if __name__ == "__main__":
    # main()
    app.run(port=5001)