from io import IncrementalNewlineDecoder
import os
from WordIndex import word_index


TEXT_FILES = []


for entry in os.scandir('.'):
    if entry.is_file():
        with open(entry, encoding='iso-8859-1') as curr_file:
        # with open(entry, encoding ='ISO-8859-1') as curr_file:
            file_contents = curr_file.read()
            clean_contents = []

            list_of_words = file_contents.split()

            for word in list_of_words:
                is_word_alpha = word.isalpha()
                if is_word_alpha:
                    clean_contents.append(word.lower())

            paragraph = " ".join(clean_contents)
            TEXT_FILES.append(paragraph)


for paragraph in TEXT_FILES:
    int_words = []
    for word in paragraph:
        int_word = word_index[word]
        if int_word:
            int_words.append(int_word)
        else:
            int_words.append(1)
    print(int_words)