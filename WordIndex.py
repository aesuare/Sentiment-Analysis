import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# Read in and store in dataframe
messages_df = pd.read_csv('Datasets/Messages.csv')
scores_df = pd.read_csv('Datasets/Scores.csv')

# Create dictionaries containing message contents and scores
file_text = {}
for index, row in messages_df.iterrows():
    file_text[row['File Name']] = row['Text']
file_score = {}
for index, row in scores_df.iterrows():
    file_score[row['Merged']] = row['env_a']

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
word_index = tokenizer.word_index   # dictionary mapping each word to its integer counterpart