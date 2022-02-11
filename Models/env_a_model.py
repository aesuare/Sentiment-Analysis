import numpy as np
import pandas as pd
import tensorflow as tf
from WordIndex import word_index, file_score, file_text

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
                    int_sentence.append(int(int_word))
                else:
                    int_sentence.append(1)
        if len(int_sentence) > 0:
            int_sentences.append(int_sentence)
    except:
        # print(f"Could not find key {key}")
        pass

# At this point, the int_sentences variable holds all the information we need
# The first number of each list contains that sentence's env_a score
# Every other number contains the integer representation of that word via WordIndex

scores = []
paragraphs = []

for sent in int_sentences:
    # First number is the env_a score
    score = sent[0]
    scores.append(score)

    # Grab all ints from the sentence but the first (score)
    par = sent[1:]
    paragraphs.append(par)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=728, input_shape=[728])])
model.compile(optimizer='adam', loss='mean_squared_error')

SCORES = np.array(scores,dtype=int)
PARAGRAPHS = np.array(paragraphs, dtype=int)



model.fit(SCORES, PARAGRAPHS, epochs=50)
print(model.predict("""2 2017 ANNUAL REPORT
DEAR STAKEHOLDERS,
2017was a challenging year for all of agriculture and for our membersÑa
year that required Land OÕLakes, Inc. to perform at its best in order to be able to do everything we
could to support our membership in this challenging time. This desire to help our membership required all of us at Land OÕLakes to more aggressively leverage our Òvalue-addedÓ approach to doing business, an approach that allows us to capture value from the marketplace and pass that value along to our members. We recognize when we do so, we deliver market access, earnings, products and, increasingly, services to our membership. This model of driving business success to pass along to our membership has been founda- tional to Land OÕLakes, Inc. and our member-owners from our inception in 1921.
Our strategy is based on a deep understanding what our consumers, customers and farmers need to be successful, and then developing innovative, value-added products and services to meet those needs. Being a cooperative literally means working together for the common good and for shared success. Every decision we make to be successful
earnings of $230 million. The business is poised to lead all aspects of precision agriculture and ag technology into its bright future.
Dairy Foods once again captured share and delivered sales of $3.9 billion and pretax earnings of $71 million. In addition, we invested in Vermont Creamery, which will provide a platform for future branded growth. The business was able to offset negative market conditions and deliver a patronage of 34 cents per hundredweight.
In Feed, we delivered another successive record perfor- mance. The business saw sales of $3.7 billion and pretax earnings of $92 million. The business saw strength in the business-to-business platform with Nutra Blend and milk replacer, and strengthened key customer relationships.
Our Land OÕLakes SUSTAIN business, which we created to both put the grower at the center of the sustainability discussion and to help evolve their own practices, is also gaining ground and new customers as it expands its foot- print. WeÕve seen valuable support from government and various associations.
On the International business front, the word was invest- ment. We saw growth in our Villa Crop Protection business in South Africa and an even better-than-expected expansion in our Bidco feed business in Kenya. In addition, we expanded our footprint in China, Mexico and Canada.
Nearly 100 years ago, Land OÕLakes, Inc. was estab-
lished by a group of dairy farmers to aggregate supply, gain bargaining power and get their Òvalue-addedÓ product where it needed to go. It evolved to use the aggregated demand of those owners to buy goods and services together, access- ing better products and services faster and cheaper than they could on their own. Nearly a century later, we have not forgotten our roots or changed our practice of using our collective strength to our membersÕ benefit.
Thank you for your business and your support this year, and the many years leading up to it.
Sincerely,
Chris Policinski President
Chief Executive Officer
Pete Kappelman Chairman of the Board
as a company today is both true to our roots and is done for the bene- fit of our members.
Taken together, in 2017, Land OÕLakes deliv-
ered $365 million in net earnings through its four businesses.
At WinField United, we completed the WinField and United Suppliers merger, grew share
and delivered sales of $5.7 billion and pretax"""
))