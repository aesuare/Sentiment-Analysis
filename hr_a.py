import numpy as np
import pandas as pd
from readfiles import readfiles
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Read in and store in dataframe
messages_df = pd.read_csv('/Users/alfredosuarez/Desktop/Files/Sentiment-Analysis/Datasets/Messages.csv')
scores_df = pd.read_csv('/Users/alfredosuarez/Desktop/Files/Sentiment-Analysis/Datasets/Scores.csv')

# Create dictionaries containing message contents and scores
file_text = {}
for index, row in messages_df.iterrows():
    file_text[row['File Name']] = row['Text']
file_score = {}
for index, row in scores_df.iterrows():
    file_score[row['Merged']] = row['hr_a']

sentences = []
scores = []


# The point of this for loop is to make sure that the file has both a score and an actual paragraph
for key in file_text:
    # I promise I did not want to have to do this but it seems I was left no other choice
    try:
        sentence = file_text.get(key)
        score = file_score.get(key)
        if sentence and score:
            sentences.append(sentence)
            int_score = float(score/5)
            scores.append(int_score)
    except:
        # print(f"Could not find key {key}")
        pass


# ===================================================================================================
# ===================================================================================================
"""
At this point, the only things that are good are the sentences and scores lists
They're in order, so if you zip iterrated the two you could see each letter and the subsequent score
"""
# ===================================================================================================
# ===================================================================================================


# Set model constants
EMBEDDING_DIM = 16
MAX_LENGTH = 300
NUM_EPOCHS = 30
TRUNC_TYPE = 'post'
OOV_TOK = '<OOV>'
VOCAB_SIZE = 50000


# Give each word an integer key
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index   # dictionary mapping each word to its integer counterpart

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=300, padding='post')

eighty_percent_mark = int(round(len(padded)*0.8))

# Divide lists into training testing
tr_pad = padded[:eighty_percent_mark]
tr_sco = scores[:eighty_percent_mark]
te_pad = padded[eighty_percent_mark:]
te_sco = scores[eighty_percent_mark:]


# Convert lists into numpy arrays
training_padded = np.array(tr_pad)
training_scores = np.array(tr_sco)
testing_padded = np.array(te_pad)
testing_scores = np.array(te_sco)



# Instantiate model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_padded, training_scores, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_scores))

model.summary()


# test_sentences = readfiles()
# np_test_sentences = np.array(test_sentences)
# test_sequences = tokenizer.texts_to_sequences(np_test_sentences)
# test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')


raw_test_sentences = readfiles()
sentence_names = []
test_sentences = []
for sentence in raw_test_sentences:
    sentence_name = sentence[0]
    actual_sentence = sentence[1]
    
    sentence_names.append(sentence_name)
    test_sentences.append(actual_sentence)

np_test_sentences = np.array(test_sentences)
test_sequences = tokenizer.texts_to_sequences(np_test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

np_predictions = model.predict(test_padded)
predictions = np_predictions.tolist()


def hr_a_predictions(readfiles_result):
    # Create list of sentences
    curr_sentences = []
    for lst in readfiles_result:
        curr_sentences.append(text:= lst[1])

    curr_np_sentences = np.array(curr_sentences)
    curr_sequences = tokenizer.texts_to_sequences(curr_np_sentences)
    curr_padded = pad_sequences(curr_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    curr_np_predictions = model.predict(curr_padded)
    model_predictions = curr_np_predictions.tolist()
    predictions = []
    for prediction in model_predictions:
        prediction_value = prediction[0]
        real_prediction = int(round(prediction_value*7))
        predictions.append(real_prediction)
    return predictions
