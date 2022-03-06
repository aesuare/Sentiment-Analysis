from WriteToResults import WriteToExcel
from readfiles import readfiles

from cg_a import cg_a_predictions
from hr_a import hr_a_predictions
from env_s import env_s_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Generate predictions for cg_a score and write to excel
predictions = cg_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=8)


# Generate predictions for hr_a score and write to excel
predictions = hr_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=9)


# Generate predictions for env_s score and write to excel
predictions = env_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=10)

