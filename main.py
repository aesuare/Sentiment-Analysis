from WriteToResults import WriteToExcel
from readfiles import readfiles

env_a_dict = {}

for lst in readfiles():
    print(f"lst: {lst}")
    file_name = lst[0]
    file_text = lst[1]

    env_a_dict[file_name] = file_text

WriteToExcel(env_a_dict, 2)
