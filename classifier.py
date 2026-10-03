import tensorflow as tf
import os
import pandas as pd
import re
import string
import json
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow.keras

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data): # standardise data
    lowercase = tf.strings.lower(input_data) # lowercase everything
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ') # replace html breaks w/ spaces
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), # remove all punctuation with regex
                                    '')


class Classifier:
    def __init__(self):
        self.max_features = 1500
        self.sequence_length = 250
        self.seed = 42
        self.embedding_dim = 16
        self.batch_size = 32

        training, testing = self.load_data()

        # Hold out 20% of the training data for validation
        val = training.sample(frac=0.2, random_state=self.seed)
        newTraining = training.drop(val.index)


        self.vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length)

        # Build the vocab from the training text
        self.vectorize_layer.adapt(newTraining['Text'].astype(str).values)
        
           
        self.train_ds = self.process_data(newTraining) 
        self.test_ds = self.process_data(testing)
        self.val_ds = self.process_data(val)
    
    def create_model(self):
        self.model = tf.keras.Sequential([ #create the model
        layers.Embedding(self.max_features, self.embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')])

        self.model.compile(loss=losses.SparseCategoricalCrossentropy(), #configure loss function and optimiser
                    # optimizer='adam', # default learning rate, 0.001
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                    metrics=['accuracy'])#
    
    def export_model(self, export_path):
        export_model = tf.keras.Sequential([ #exported model vectorises input then parses it into the trained model
        tf.keras.Input(shape=(), dtype=tf.string),
        self.vectorize_layer,
        self.model,
        ])

        export_model(tf.constant(["build the model"]))  # builds it

        # Save the model to the specified path
        export_model.save(export_path)
    
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_ds)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

    def save_outputs(self, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)

        #save training data for charts
        with open(f"{out_dir}/history.json", "w") as f:
            json.dump(self.history.history, f)

        # Test-set predictions, for the confusion matrix and per-class metrics
        y_true = np.concatenate([labels.numpy() for _, labels in self.test_ds]) #pulls out labels from each line
        y_pred = np.argmax(self.model.predict(self.test_ds), axis=1) #runs trained model over each and picks largest probability 
        #compare y_true and y_pred to form confusion matrix 
        np.savez(f"{out_dir}/predictions.npz", y_true=y_true, y_pred=y_pred)


    def train_model(self):
        epochs = 90
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs)

    def process_data(self, df):
        """Turn a dataframe into a dataset of (vectorized_text, label) batches."""
        texts = df['Text'].astype(str).values
        labels = df['Label'].astype('int').values
 
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        ds = ds.batch(self.batch_size)
        ds = ds.map(self.vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
    def vectorize_text(self, text, label):
        # text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label

    def load_data(self):
        #load the training and testing data from CSV files
        training = pd.read_csv("training.csv", names=['Text', 'Label'], comment='#')
        testing = pd.read_csv("testing.csv", names=['Text', 'Label'], comment='#')
        
        return training, testing

if __name__ == "__main__":
    classifier = Classifier()
    classifier.create_model()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.save_outputs()
    classifier.export_model("classifierModel.keras")