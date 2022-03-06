from WriteToResults import WriteToExcel
from readfiles import readfiles

from comm_s import comm_s_predictions
from emp_s import emp_s_predictions
from div_s import div_s_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Generate predictions for comm_s score and write to excel
predictions = comm_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=11)


# Generate predictions for comm_s score and write to excel
predictions = emp_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=12)


# Generate predictions for emp_s score and write to excel
predictions = div_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=13)

