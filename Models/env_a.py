from re import L
from WordIndex import word_index, file_score, file_text
import pandas as pd

messages_df = pd.read_csv('../Datasets/Messages.csv')
int_sentences = []



for key in file_text:
    # I promise I did not want to have to do this but it seems I was left no other choice
    try:
        int_sentence = []
        sentence = file_text.get(key)
        score = file_score.get(key)
        if sentence and score:
            int_sentence.append(score)
            sentence_list = sentence.split()
            for word in sentence_list:
                int_word = word_index.get(word)
                if int_word:
                    int_sentence.append(int_word)
                else:
                    int_sentence.append(1)
        if len(int_sentence) > 0:
            int_sentences.append(int_sentence)
    except:
        # print(f"Could not find key {key}")
        pass

for s in int_sentences:
    print(s, "\n")

print(len(int_sentences))