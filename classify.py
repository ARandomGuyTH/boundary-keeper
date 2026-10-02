import tensorflow as tf
import classifier  

def classify(text):
    model = tf.keras.models.load_model("classifierModel.keras")
    prediction = model.predict(tf.constant([text]))
    return prediction

if __name__ == "__main__":
    test_text = "Many people make sure they have an emergency buffer before investing"
    prediction = classify(test_text)
    print(f"Prediction for '{test_text}': {prediction}")