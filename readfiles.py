import os

def readfiles():
    # Folder Path
    path = "/Users/alfredosuarez/Desktop/Files/Sentiment-Analysis/Datasets/Texts"

    paragraphs = []

    # Change the directory
    os.chdir(path)

    # Read text File
    def read_text_file(file_path, lst):
        with open(file_path, 'r', encoding="iso-8859-1") as f:
            raw_file_name = f.name
            raw_file_name_list = raw_file_name.split("/")
            curr_file_name = raw_file_name_list[-1]
            paragraph = f.read()
            paragraph_list = paragraph.split()
            alnum_paragraph = []
            for word in paragraph_list:
                is_word_alphanumeric = word.isalpha()
                if is_word_alphanumeric:
                    alnum_paragraph.append(word.lower())
            cleaned_paragraph = " ".join(alnum_paragraph)
            return_contents = [curr_file_name, cleaned_paragraph]
            lst.append(return_contents)


    # Iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            # call read text file function
            read_text_file(file_path, paragraphs)

    return paragraphs

print(readfiles())