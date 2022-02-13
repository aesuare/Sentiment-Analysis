import os
from tkinter.tix import TEXT


TEXT_FILES = []


for entry in os.scandir('.'):
    if entry.is_file():
        with open(entry, encoding='iso-8859-1') as curr_file:
        # with open(entry, encoding ='ISO-8859-1') as curr_file:
            file_contents = curr_file.read()
            # print(file_contents)
            TEXT_FILES.append(file_contents)
            print("====================================================================================")




print(TEXT_FILES)
            