from WriteToResults import WriteToExcel
from readfiles import readfiles

from emp_a import emp_a_predictions
from env_a import env_a_predictions
from comma_a import comma_a_predictions
from emp_a import emp_a_predictions
from div_a import div_a_predictions
from con_a import con_a_predictions
from cg_a import cg_a_predictions
from hr_a import hr_a_predictions
from env_s import env_s_predictions
from comm_s import comm_s_predictions
from emp_s import emp_s_predictions
from div_s import div_s_predictions
from con_s import con_s_predictions
from cg_s import cg_s_predictions
from hr_s import hr_s_predictions
from csr import csr_predictions

# Read files and notify us we've finished reading all text files
print("Reading files...")
lists = readfiles()
print("Finished reading files...")


# Generate predictions for comm_s score and write to excel
predictions = comm_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=11)


# Generate predictions for comm_s score and write to excel
predictions = comm_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=12)


# Generate predictions for emp_s score and write to excel
predictions = emp_s_predictions(lists)
print(predictions)
WriteToExcel(predictions, column_number=13)

