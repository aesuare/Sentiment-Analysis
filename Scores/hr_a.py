import pandas as pd

# Read in and store in dataframe
messages_df = pd.read_csv('Datasets/Messages.csv')
scores_df = pd.read_csv('Datasets/Scores.csv')

# Create dictionaries containing message contents and scores
file_text = {}
for index, row in messages_df.iterrows():
    file_text[row['File Name']] = row['Text']
file_score = {}
for index, row in scores_df.iterrows():
    file_score[row['Merged']] = row['hr_a']

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

HR_A = {}
for score, sentence in zip(scores, sentences):
    HR_A[sentence] = score

# The above constant is a dictionary mapping each paragraph to its score