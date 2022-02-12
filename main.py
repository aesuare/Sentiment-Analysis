"""
There's nothing here yet, but this will be where we execute all the logic built throughout
the rest of the repository..
"""
from WordIndex import word_index
from Scores.div_a import DIV_A
import tensorflow as tf

all_int_words = []

for key in DIV_A:
    word_list = key.split()
    int_words = []
    for word in word_list:
        try:
            int_word = word_index[word]
            if int_word:
                int_words.append(int_word)
            else:
                int_words.append(1)
        except:
            pass
    all_int_words.append(int_words)

# Holds equal length vectors for the words of each sentence
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    all_int_words, padding='pre'
)
