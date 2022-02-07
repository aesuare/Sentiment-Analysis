from readfiles import messages, scores
from pprint import pprint

# print(messages, "\n\n\n", scores)

texts = {}
for index, row in messages.iterrows():
    texts[row['File Name']] = row['Text']

pprint(texts)