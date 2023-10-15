# import statements:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from keras_tuner import HyperModel, BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from kerastuner import *

df = pd.read_csv("train.csv",encoding= 'unicode_escape')
df = df[['sentiment','text']]

def preprocess(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    text = text.lower()
    return text
corpus = [preprocess(text) for text in df['text']]

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(corpus)
padded_sequences = pad_sequences(sequences, padding='post')

sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
labels = np.array([sentiment_mapping[sentiment] for sentiment in df["sentiment"]])

train_texts, val_texts, train_labels, val_labels = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

class_counts = np.bincount(train_labels)
total_samples = sum(class_counts)
class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts)}

class SentimentHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, hp.Int('embedding_dim', min_value=64, max_value=256, step=32)))
        model.add(Bidirectional(LSTM(hp.Int('lstm_units', min_value=64, max_value=128, step=32), return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(3, activation='softmax'))
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1, sampling='LOG'))
        loss = SparseCategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

hypermodel = SentimentHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=1
)

tuner.search(train_texts, train_labels, epochs=20, validation_data=(val_texts, val_labels),
             class_weight=class_weights, callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = hypermodel.build(best_hps)
best_model.fit(train_texts, train_labels, epochs=10, validation_data=(val_texts, val_labels),
               class_weight=class_weights)
best_model.save("best_model.h5")

def predict_sentiment_with_best_model(user_input, model):
    preprocessed_input = preprocess(user_input)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)
    sentiment_probabilities = model.predict(padded_sequence)[0]
    predicted_sentiment = np.argmax(sentiment_probabilities)
    return list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_sentiment)]

best_model = tuner.get_best_models(num_models=1)[0]

while True:
    user_input = input("Enter your message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment_with_best_model(user_input, best_model)
    print("Predicted sentiment:", sentiment)
