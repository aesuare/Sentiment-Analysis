from WriteToResults import WriteToExcel
from readfiles import readfiles

from con_s import con_s_predictions
from cg_s import cg_s_predictions
from hr_s import hr_s_predictions
from csr import csr_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Generate predictions for div_s score and write to excel
predictions = con_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=14)


# Generate predictions for con_s score and write to excel
predictions = cg_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=15)


# Generate predictions for cg_s score and write to excel
predictions = hr_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=16)


# Generate predictions for cg_s score and write to excel
predictions = csr_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=17)