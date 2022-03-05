from WriteToResults import WriteToExcel
import numpy as np
import tensorflow as tf
from readfiles import readfiles
from env_a import env_a_predictions

print("Reading files...")
lists = readfiles()
print("Finished reading files...")

predictions = env_a_predictions(lists)
print(predictions)