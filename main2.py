from WriteToResults import WriteToExcel
from readfiles import readfiles

from emp_a import emp_a_predictions
from div_a import div_a_predictions
from con_a import con_a_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Generate predictions for emp_a score and write to excel
predictions = emp_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=5)


# Generate predictions for div_a score and write to excel
predictions = div_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=6)


# Generate predictions for con_a score and write to excel
predictions = con_a_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=7)

