import ollama
import tensorflow as tf
import classifier 
import flask
import numpy as np
import inputClassifier

app = flask.Flask(__name__, static_folder="front-end", static_url_path="")

INPUT_LABELS = ["DIRECT SOLICITATION", "EMOTIONAL PRESSURE", "HYPOTHETICAL", "FALSE AUTHORITY",
                "ENDORSEMENT BAIT", "JAILBREAK/RULE BYPASS", "INDIRECT", "NEUTRAL"]
RESPONSE_LABELS = ["PERSONAL", "GREY AREA", "GENERAL"]

MODEL_PATH = "classifierModel.keras"
INPUT_MODEL_PATH = "inputClassifierModel.keras"
OLLAMA_MODEL = "llama3.2" # change this to anything 

classifier = tf.keras.models.load_model(MODEL_PATH)
inputClassifier = tf.keras.models.load_model(INPUT_MODEL_PATH)

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
    type1 = int(flask.request.args.get("type").strip())
    result, confidence = classify(text, type1)
    return flask.jsonify({"result": result, "confidence" : confidence})

def askOllama(question, model="gemma2:2b"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question + " Please provide a short answer."}],
        options={
            "num_predict": 300  # limit no. of words in a response
        }
    )
    return response["message"]["content"]

def classify(text, type1):
    labels, model = (INPUT_LABELS, inputClassifier) if type1 == 0 else (RESPONSE_LABELS, classifier)
    probs = model.predict(tf.constant([text]), verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])   # actual probability, e.g. 0.92
    return labels[idx], confidence

def main():
    print("Ask the AI for financial advice, and we'll classify the response.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() == "quit":
            break
        
        response = askOllama(question)
        print(f"{OLLAMA_MODEL}: {response}\n")
        print(f"Classification: {classify(response, 1)}")


        


if __name__ == "__main__":
    # main()
    app.run(port=5001)