from WriteToResults import WriteToExcel
from readfiles import readfiles

from env_a import env_a_predictions
from comma_a import comma_a_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Update excel sheet with names and contents of files
files = {}
for lst in lists:
    file_name = lst[0]
    file_contents = lst[1]
    files[file_name] = file_contents

WriteToExcel(files, column_number=2)


# Generate predictions for env_a score and write to excel
predictions = env_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=3)


# Generate predictions for comma_a score and write to excel
predictions = comma_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=4)
