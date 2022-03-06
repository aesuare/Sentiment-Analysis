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
