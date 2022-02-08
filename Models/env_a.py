import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Read in and store in dataframe
raw_messages_df = pd.read_csv('../Datasets/Messages.csv')
raw_scores_df = pd.read_csv('../Datasets/Scores.csv')
split = round(len(raw_messages_df)*0.8)

messages_df = raw_messages_df[:split]
scores_df = raw_scores_df[:split]

test_messages = raw_messages_df[split:]
test_scores = raw_scores_df[split:]

# Create dictionaries containing message contents and scores
file_text = {}
for index, row in messages_df.iterrows():
    file_text[row['File Name']] = row['Text']
file_score = {}
for index, row in scores_df.iterrows():
    file_score[row['Merged']] = row['env_a']

sentences = []
scores = []

for key in file_text:
    # I promise I did not want to have to do this but it seems I was left no other choice
    try:
        sentence = file_text.get(key)
        score = file_score.get(key)
        if sentence and score:
            sentences.append(sentence)
            scores.append(score)
    except:
        # print(f"Could not find key {key}")
        pass

# Set model constants
EMBEDDING_DIM = 16
MAX_LENGTH = 250
NUM_EPOCHS = 20
TRUNC_TYPE = 'post'
OOV_TOK = '<OOV>'
VOCAB_SIZE = 50000

# Give each word an integer key
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert words into integer equivalent
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)
np_scores = np.array(scores)

# Yay we get to create a real model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, np_scores, epochs=NUM_EPOCHS)

prediction_seeds = np.array(raw_messages_df)
prediction = model.predict(prediction_seeds)
print(f"prediction: {prediction}")