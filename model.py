import numpy as np
import pandas as pd
import tensorflow as tf

training_set = pd.read_csv('Datasets/twitter_training.csv')
message_content = training_set[["Tweet Content"]]
sentiment = training_set[["Sentiment"]]

