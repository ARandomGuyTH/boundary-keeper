import tensorflow as tf
import os
import pandas as pd
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses

class Classifier:
    def __init__(self):
        training, testing = self.load_data()
        training_val =training.pop('inBound') #hide the inBound column from the training data
        testing_val = testing.pop('inBound')

        self.max_features = 10000
        self.sequence_length = 250
        self.seed = 42
        self.embedding_dim = 16

        self.vectorize_layer = layers.TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length)
        
        self.train_ds = self.process_data(training)
        self.test_ds = self.process_data(testing)
        self.val_ds = self.process_data(training_val)
    
    def create_model(self):
        self.model = tf.keras.Sequential([ #create the model
        layers.Embedding(self.max_features, self.embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')])

        self.model.compile(loss=losses.BinaryCrossentropy(), #configure loss function and optimiser
                    optimizer='adam',
                    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])#
    
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_ds)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

    def train_model(self):
        epochs = 10
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs)

    def process_data(self, data):
        batch_size = 32

        ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)#convert to dataset and batch the data

        ds = ds.map(self.vectorize_text) #vectorize the text data

        AUTOTUNE = tf.data.AUTOTUNE

        ds = ds.cache().prefetch(buffer_size=AUTOTUNE) #cache for optimal performance
        return ds
            
    def vectorize_text(self, text):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text)

    def custom_standardization(self,input_data): #standardise data
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    def load_data(self):
        #load the training and testing data from CSV files
        training = pd.read_csv("training.csv", names=['Output', 'inBound'], comment='#')
        testing = pd.read_csv("testing.csv", names=['Output', 'inBound'], comment='#')
        
        return training, testing

if __name__ == "__main__":
    classifier = Classifier()
    classifier.create_model()
    classifier.train_model()
    classifier.evaluate_model()