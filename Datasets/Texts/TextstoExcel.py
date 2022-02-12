import os
for entry in os.scandir('.'):
    if entry.is_file():
        # with open(entry, 'rb') as curr_file:
        with open(entry, encoding ='ISO-8859-1') as curr_file:
            print(curr_file.read())